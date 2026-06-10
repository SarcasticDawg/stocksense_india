from flask import Flask, render_template, request, jsonify, redirect
import sys
import os
import numpy as np
import concurrent.futures
import time
import threading
import pandas as pd
import gc
import traceback
import pytz
from datetime import datetime
from pymongo import MongoClient

# Add engines to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'engines'))

import data_engine
import predictor
import sentiment
import sector
import macro
import fno
import market_context
import meta_model
import backtester
import conviction

app = Flask(__name__)

# Config
USE_LEGACY_WEIGHTS = os.environ.get('USE_LEGACY_WEIGHTS', 'false').lower() == 'true'
MONGODB_URI = os.getenv("MONGODB_URI", "")

# Global state (Initialized lazily in background)
_stock_predictor = None
_sentiment_analyzer = None
_meta_model = None
_mongodb_collection = None
_models_loading = False
_training_lock = threading.Lock()

def load_engines_task():
    """
    Background task to initialize heavy models and DB connection.
    This allows Gunicorn/Render to start the server instantly.
    """
    global _stock_predictor, _sentiment_analyzer, _meta_model, _mongodb_collection, _models_loading
    
    if _models_loading: return
    _models_loading = True
    
    print("--- [Background] Initializing Market-Adaptive Ensemble & DB... ---")
    
    # 1. MongoDB Setup
    if MONGODB_URI:
        try:
            client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
            db = client.stocksense
            _mongodb_collection = db.stocksense_results
            # Just a quick check, don't let it block indefinitely
            client.admin.command('ping')
            print("--- [Background] MongoDB Connected Successfully ---")
        except Exception as e:
            print(f"--- [Background] MongoDB Connection Error: {e} ---")
            _mongodb_collection = None
    else:
        print("--- [Background] WARNING: MONGODB_URI not set. Night Mode will be disabled. ---")

    # 2. Heavy Engine Loading
    try:
        if _sentiment_analyzer is None: _sentiment_analyzer = sentiment.SentimentAnalyzer()
        if _meta_model is None: _meta_model = meta_model.MetaModel()
        
        mode = get_dashboard_mode()
        if mode == "LIVE":
            if _stock_predictor is None: _stock_predictor = predictor.StockPredictor()
            print("--- [Background] All Engines (LIVE MODE) Ready. ---")
        else:
            print("--- [Background] Night Mode: Skipping heavy ML models to save memory. ---")
            
    except Exception as e:
        print(f"--- [Background] Error loading models: {e} ---")
        
    _models_loading = False

# Start initialization thread IMMEDIATELY at the top level
# This ensures it runs even when imported by Gunicorn
threading.Thread(target=load_engines_task, daemon=True).start()

def get_models():
    """Helper to get models, ensuring they are loaded."""
    global _stock_predictor, _sentiment_analyzer, _meta_model
    if _stock_predictor is None or _sentiment_analyzer is None or _meta_model is None:
        # If they aren't ready yet, the dashboard will handle it gracefully
        pass
    return _stock_predictor, _sentiment_analyzer, _meta_model

def safe_train(pred_engine, symbol):
    with _training_lock:
        print(f"--- [Background] Training for {symbol} ---")
        try:
            pred_engine.train(symbol)
        except Exception as e:
            print(f"Background Training Error: {e}")
        gc.collect()

def compute_combined_signal(symbol, pred_data, sentiment_metadata, macro_data, sector_data, conviction_data, meta_engine):
    """
    Robust signal computation with Contextual Sentiment Weighting.
    """
    # 1. Prepare AI Scores (Blended LightGBM + XGBoost)
    lgbm_change = pred_data.get('price_change_pct', 0) if isinstance(pred_data, dict) else 0
    lgbm_score = np.clip(lgbm_change / 5.0, -1, 1)
    
    xgb_prob = pred_data.get('trend_prob_up', 0.5) if isinstance(pred_data, dict) else 0.5
    xgb_score = (xgb_prob - 0.5) * 2 
    
    ai_score = (lgbm_score * 0.6) + (xgb_score * 0.4)
    
    inst = macro_data.get('INSTITUTIONAL', {})
    cash_net = 0.0
    if isinstance(inst, dict) and isinstance(inst.get('cash'), dict):
        cash_net = float(inst['cash'].get('fii', 0)) + float(inst['cash'].get('dii', 0))
    inst_flow_score = np.clip(cash_net / 2500.0, -1, 1)
    
    pcr_val = 1.0
    if isinstance(pred_data, dict) and isinstance(pred_data.get('pcr_data'), dict):
        pcr_val = float(pred_data['pcr_data'].get('pcr', 1.0))
    pcr_score = np.clip((pcr_val - 0.9) * 2.0, -1, 1)
    
    conv_score = 0.0
    if isinstance(conviction_data, dict):
        conv_val = conviction_data.get('latest_pct', 30)
        conv_score = np.clip((conv_val - 40) / 20.0, -1, 1)

    sector_score = 0.0
    if isinstance(sector_data, dict):
        peers = sector_data.get('peers', [])
        clean_sym = symbol.replace(".NS", "")
        symbol_rank = 0
        for i, p in enumerate(peers):
            if isinstance(p, dict) and p.get('symbol') == clean_sym:
                symbol_rank = i
                break
        if len(peers) > 0:
            sector_score = (1.0 - (symbol_rank / len(peers))) * 2 - 1

    regime_data = macro_data.get('REGIME', {})
    regime_val = 1 if regime_data.get('regime') == 'BULL' else (-1 if regime_data.get('regime') == 'BEAR' else 0)

    # 2. Determine Contextual Sentiment Weight
    sent_score = sentiment_metadata.get('total_score', 0.0)
    mentions = sentiment_metadata.get('mentions', {})
    
    # 6-Circumstance Logic
    sent_weight = 0.20
    sent_mode = "Full"
    
    if mentions.get('corp', 0) >= 1:
        sent_weight = 0.20
        sent_mode = "Full (Official)"
    elif mentions.get('news', 0) >= 3:
        sent_weight = 0.10
        sent_mode = "Partial (News)"
    elif mentions.get('social', 0) >= 3: # Updated for Screener only
        sent_weight = 0.05
        sent_mode = "Minimal (Social)"
    else:
        sent_weight = 0.0
        sent_mode = "None (Insufficient)"

    current_features = {
        'ai_price': float(ai_score),
        'sentiment': float(sent_score),
        'inst_flow': float(inst_flow_score),
        'pcr': float(pcr_score),
        'conviction': float(conv_score),
        'sector': float(sector_score),
        'regime': float(regime_val)
    }

    # 3. Compute Total Score
    if USE_LEGACY_WEIGHTS or meta_engine.model is None:
        w_ai, w_sent, w_inst, w_conv, w_sect = 0.30, sent_weight, 0.20, 0.15, 0.15
        
        # Non-F&O Fallback
        has_fo = mentions.get('corp', 0) > 0 or pcr_val != 1.0 
        if not has_fo:
            w_inst += 0.05
            w_ai += 0.10
            
        total_w = w_ai + w_sent + w_inst + w_conv + w_sect
        w_ai /= total_w
        w_sent /= total_w
        w_inst /= total_w
        w_conv /= total_w
        w_sect /= total_w
        
        total_score = (
            (current_features['ai_price'] * w_ai) +
            (current_features['sentiment'] * w_sent) +
            (current_features['inst_flow'] * w_inst) +
            (current_features['conviction'] * w_conv) +
            (current_features['sector'] * w_sect)
        )
        method = "Contextual Manual Weights"
    else:
        total_score = meta_engine.predict(current_features)
        method = "Machine-Learned Ensemble"

    meta_engine.cache_features(symbol, current_features)
    
    verdict = "HOLD"
    if total_score > 0.3: verdict = "BUY"
    elif total_score < -0.3: verdict = "SELL"
    
    return {
        'verdict': verdict,
        'score': round(float(total_score), 2),
        'method': method,
        'sent_mode': sent_mode,
        'breakdown': {
            'price': round(float(ai_score), 2),
            'sentiment': round(float(sent_score), 2),
            'institutional': round(float(inst_flow_score), 2),
            'conviction': round(float(conv_score), 2),
            'sector': round(float(sector_score), 2)
        }
    }

analysis_cache = {}

def get_mongodb_collection():
    """Helper to safely get the MongoDB collection, even if loaded late."""
    global _mongodb_collection
    return _mongodb_collection

@app.route('/')
def home():
    try:
        indices = data_engine.get_market_indices()
        regime = market_context.get_market_regime()
        breadth = market_context.get_market_breadth()
        index_summary = {}
        for name, df in indices.items():
            if df is not None and not df.empty:
                latest = df['Close'].iloc[-1]
                prev = df['Close'].iloc[0]
                change = ((latest - prev) / prev) * 100
                index_summary[name] = {"value": round(float(latest), 2), "change": round(float(change), 2)}
            else:
                index_summary[name] = {"value": "N/A", "change": 0.0}
        try:
            stocks_df = pd.read_csv('data/nse_stocks.csv')
            stock_list = stocks_df.sort_values(by='Symbol').to_dict('records')
        except:
            stock_list = []
        return render_template('home.html', indices=index_summary, stock_list=stock_list, regime=regime, breadth=breadth, app_name="StockIntel")
    except Exception as e:
        print(f"Home route error: {e}")
        return render_template('error.html', message="System error on home page.")

@app.route('/stock/<symbol>')
def stock_detail(symbol):
    try:
        if not symbol.endswith(".NS"): symbol += ".NS"
        
        mode = get_dashboard_mode()
        print(f"--- [Dashboard] Mode: {mode} ---")
        
        pred_engine, sent_engine, meta_engine = get_models()

        if pred_engine is None or sent_engine is None or meta_engine is None:
            return render_template('error.html', message="System is still warming up. Models are loading in the background. Please refresh in 30 seconds.")

        if mode == "NIGHT":
            # 1. Try to fetch morning_prediction (freshest) or nightly_dump
            mongo_col = get_mongodb_collection()
            latest_data = None
            if mongo_col is not None:
                latest_data = mongo_col.find_one(
                    {"symbol": symbol, "type": {"$in": ["morning_prediction", "nightly_dump"]}},
                    sort=[("timestamp", -1)]
                )
            
            if latest_data:
                print(f"Found pre-computed data for {symbol} in MongoDB.")
                
                # In NIGHT MODE, we only run sentiment fresh
                # But to follow the request "only runs sentiment engine fresh on user search",
                # we need to combine nightly data with fresh sentiment.
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    future_raw = executor.submit(data_engine.get_all_raw_data, symbol)
                    future_chart = executor.submit(data_engine.get_chart_data, symbol)
                    
                    raw_data = future_raw.result()
                    chart_data = future_chart.result()
                
                nightly = latest_data['data'] if latest_data['type'] == "nightly_dump" else latest_data['data']['nightly_data']
                
                # Fresh Sentiment
                news_items = raw_data.get('news', [])
                social_items = raw_data.get('social', [])
                corp_items = raw_data.get('corporate', [])

                news_sent = sent_engine.analyze_batch(news_items, is_news=True)
                social_sent = sent_engine.analyze_batch(social_items)
                corp_sent = sent_engine.analyze_batch(corp_items, is_corporate=True)
                total_sent = (news_sent * 0.3) + (social_sent * 0.2) + (corp_sent * 0.5)

                sent_metadata = {
                    'total_score': total_sent,
                    'mentions': {
                        'news': len(news_items), 
                        'social': len(social_items), 
                        'corp': len(corp_items),
                        'total': len(news_items) + len(social_items) + len(corp_items)
                    }
                }

                # Combined Signal with pre-computed parts
                signal = compute_combined_signal(
                    symbol, 
                    nightly['pred_data'], 
                    sent_metadata, 
                    nightly['macro_data'], 
                    nightly['sector_data'], 
                    nightly['conv_data'], 
                    meta_engine
                )
                
                # Unified Feed
                unified_feed = []
                for n in news_items: unified_feed.append({'title': n['title'], 'link': n.get('link', '#'), 'date': n['date'], 'source': n['source'], 'type': 'News'})
                for s in social_items: unified_feed.append({'title': s['title'], 'link': s.get('link', '#'), 'date': s['date'], 'source': s['source'], 'type': 'Social'})
                for c in corp_items: unified_feed.append({'title': c['title'], 'link': c.get('link', '#'), 'date': c['date'], 'source': 'NSE/BSE', 'type': 'Announcement'})

                render_params = {
                    'symbol': symbol, 'prediction': nightly['pred_data'], 'pcr': nightly['pcr_data'], 'conviction': nightly['conv_data'],
                    'macro': nightly['macro_data'], 'sector': nightly['sector_data'], 'signal': signal,
                    'backtest': nightly['bt_data'], 'unified_feed': unified_feed,
                    'sent_data': {
                        'news': news_sent, 'social': social_sent, 'corporate': corp_sent, 'total': total_sent, 
                        'mentions': sent_metadata['mentions']
                    },
                    'chart_data': chart_data, 'app_name': "StockIntel",
                    'mode': 'NIGHT (Cached)'
                }
                return render_template('stock.html', **render_params)
            else:
                print(f"--- [Dashboard] No MongoDB data found for {symbol}. Falling back to LIVE MODE. ---")

        # Fallback to LIVE MODE (on-demand fresh engines)
        if symbol in analysis_cache:
            ts, data = analysis_cache[symbol]
            if time.time() - ts < 1800: return render_template('stock.html', **data)

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            future_models = executor.submit(get_models)
            future_raw = executor.submit(data_engine.get_all_raw_data, symbol)
            future_macro = executor.submit(macro.get_macro_data)
            future_regime = executor.submit(market_context.get_market_regime)
            future_sector = executor.submit(sector.get_sector_analysis, symbol)
            future_chart = executor.submit(data_engine.get_chart_data, symbol)
            future_pcr = executor.submit(fno.get_stock_pcr, symbol)
            future_conv = executor.submit(conviction.get_delivery_data, symbol)
            future_bt = executor.submit(backtester.StrategyBacktester().run_backtest, symbol)

            raw_data = future_raw.result()
            macro_data = future_macro.result() or {}
            macro_data['REGIME'] = future_regime.result()
            sector_data = future_sector.result() or {}
            chart_data = future_chart.result() or {"labels":[], "prices":[]}
            pcr_data = future_pcr.result() or {"pcr": 1.0, "sentiment": "Neutral"}
            conv_data = future_conv.result()
            bt_data = future_bt.result()
            pred_engine, sent_engine, meta_engine = future_models.result()
            
        if not raw_data or raw_data.get('price_data') is None:
            return render_template('error.html', message=f"Could not fetch data for {symbol}")

        cur_price = round(float(raw_data['price_data']['Close'].iloc[-1]), 2)
        lgbm_path = os.path.join('models', f'lgbm_{symbol}.joblib')
        
        if not os.path.exists(lgbm_path):
            threading.Thread(target=safe_train, args=(pred_engine, symbol)).start()
            pred_data = {'current_price': cur_price, 'predicted_price': "Training...", 'price_change_pct': 0.0, 'trend_prob_up': 0.5}
        else:
            pred_data = pred_engine.get_prediction(symbol) or {'current_price': cur_price, 'predicted_price': "Error", 'price_change_pct': 0.0, 'trend_prob_up': 0.5}

        pred_data['pcr_data'] = pcr_data
        
        # --- Advanced Sentiment Engine v4.0 ---
        news_items = raw_data.get('news', [])
        social_items = raw_data.get('social', [])
        corp_items = raw_data.get('corporate', [])

        news_sent = sent_engine.analyze_batch(news_items, is_news=True)
        social_sent = sent_engine.analyze_batch(social_items)
        corp_sent = sent_engine.analyze_batch(corp_items, is_corporate=True)
        
        # Blended Sentiment with official source dominance
        total_sent = (news_sent * 0.3) + (social_sent * 0.2) + (corp_sent * 0.5)

        sent_metadata = {
            'total_score': total_sent,
            'mentions': {
                'news': len(news_items), 
                'social': len(social_items), 
                'corp': len(corp_items),
                'total': len(news_items) + len(social_items) + len(corp_items)
            }
        }

        # Unified Feed
        unified_feed = []
        for n in news_items: unified_feed.append({'title': n['title'], 'link': n.get('link', '#'), 'date': n['date'], 'source': n['source'], 'type': 'News'})
        for s in social_items: unified_feed.append({'title': s['title'], 'link': s.get('link', '#'), 'date': s['date'], 'source': s['source'], 'type': 'Social'})
        for c in corp_items: unified_feed.append({'title': c['title'], 'link': c.get('link', '#'), 'date': c['date'], 'source': 'NSE/BSE', 'type': 'Announcement'})

        # Final Signal
        signal = compute_combined_signal(symbol, pred_data, sent_metadata, macro_data, sector_data, conv_data, meta_engine)
        
        render_params = {
            'symbol': symbol, 'prediction': pred_data, 'pcr': pcr_data, 'conviction': conv_data,
            'macro': macro_data, 'sector': sector_data, 'signal': signal,
            'backtest': bt_data, 'unified_feed': unified_feed,
            'sent_data': {
                'news': news_sent, 'social': social_sent, 'corporate': corp_sent, 'total': total_sent, 
                'mentions': sent_metadata['mentions']
            },
            'chart_data': chart_data, 'app_name': "StockIntel"
        }
        # Clear cache for this stock if it has old structure
        if pred_data.get('predicted_price') != "Training...":
            analysis_cache[symbol] = (time.time(), render_params)
        return render_template('stock.html', **render_params)
    except Exception as e:
        print(f"Stock route error for {symbol}: {e}")
        traceback.print_exc()
        return render_template('error.html', message=f"Internal Error processing {symbol}")

@app.route('/api/chart/<symbol>')
def api_chart(symbol):
    period = request.args.get('period', '3mo')
    chart_data = data_engine.get_chart_data(symbol, period=period)
    return jsonify(chart_data)

@app.route('/stock_search')
def stock_search():
    symbol = request.args.get('symbol', '').upper()
    return redirect(f'/stock/{symbol}') if symbol else redirect('/')

if __name__ == '__main__':
    # Thread is already started at the top level for Gunicorn, 
    # but we keep it here for local development runs.
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
