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

# Global state
_stock_predictor = None
_sentiment_analyzer = None
_meta_model = None
_models_loading = False
_training_lock = threading.Lock()

# Config
USE_LEGACY_WEIGHTS = os.environ.get('USE_LEGACY_WEIGHTS', 'false').lower() == 'true'

def load_models_task():
    global _stock_predictor, _sentiment_analyzer, _meta_model, _models_loading
    if _stock_predictor is None or _sentiment_analyzer is None or _meta_model is None:
        if _models_loading: return
        _models_loading = True
        print("--- [Server] Initializing Market-Adaptive Ensemble... ---")
        try:
            if _stock_predictor is None: _stock_predictor = predictor.StockPredictor()
            if _sentiment_analyzer is None: _sentiment_analyzer = sentiment.SentimentAnalyzer()
            if _meta_model is None: _meta_model = meta_model.MetaModel()
        except Exception as e:
            print(f"Error loading models: {e}")
        _models_loading = False
        print("--- [Server] Ensemble Ready. ---")

def get_models():
    global _stock_predictor, _sentiment_analyzer, _meta_model
    if _stock_predictor is None or _sentiment_analyzer is None or _meta_model is None:
        load_models_task()
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
    elif mentions.get('reddit', 0) >= 5:
        sent_weight = 0.05
        sent_mode = "Minimal (Social)"
    else:
        sent_weight = 0.0
        sent_mode = "None (Insufficient)"

    # 3. F&O Data Availability Check
    has_fo = False
    inst = macro_data.get('INSTITUTIONAL', {})
    if isinstance(inst, dict):
        # Check if FII Futures or Options views are present (not 'Neutral' or empty)
        f_view = inst.get('fii_future', {}).get('view', 'Neutral')
        o_view = inst.get('fii_option', {}).get('view', 'Neutral')
        if f_view != 'Neutral' or o_view != 'Neutral' or pcr_val != 1.0:
            has_fo = True

    current_features = {
        'ai_price': float(ai_score),
        'sentiment': float(sent_score),
        'inst_flow': float(inst_flow_score),
        'pcr': float(pcr_score),
        'conviction': float(conv_score),
        'sector': float(sector_score),
        'regime': float(regime_val)
    }

    # 4. Compute Total Score
    if USE_LEGACY_WEIGHTS or meta_engine.model is None:
        # Redistribution Logic for Manual Weights
        # Baseline: AI (0.30), Sentiment (0.20), Inst (0.20), Conv (0.15), Sect (0.15)
        # PCR is normally 15% of the 20% Inst slice, but here we treat it as 0.15 of total in legacy.
        
        # Initial target weights
        w_ai, w_sent, w_inst, w_conv, w_sect = 0.30, sent_weight, 0.20, 0.15, 0.15
        
        # If no F&O data, redistribute PCR weight (part of w_inst and w_ai)
        if not has_fo:
            # Shift weight from F&O components to Cash Flow and AI
            w_inst += 0.05 # Increase Cash Flow importance
            w_ai += 0.10   # Increase AI importance
            # We skip PCR calculation in total_score below by effectively setting it to 0 or 
            # reducing the institutional slice to just Cash.
        
        # Normalize weights to ensure sum = 1.0
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
        if not has_fo: method += " (Non-F&O Fallback)"
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
        
        # Enhanced Sentiment Processing
        news_texts = [n['text'] for n in raw_data.get('news', [])]
        reddit_texts = [p['title'] + " " + " ".join(p['comments']) for p in raw_data.get('reddit', [])]
        reddit_scores = [p['score'] for p in raw_data.get('reddit', [])]
        corp_texts = [c['title'] for c in raw_data.get('corporate', [])]

        news_sent = sent_engine.analyze_batch(news_texts, is_news=True) if news_texts else 0.0
        reddit_sent = sent_engine.analyze_batch(reddit_texts, weights=reddit_scores) if reddit_texts else 0.0
        corp_sent = sent_engine.analyze_batch(corp_texts, is_corporate=True) if corp_texts else 0.0
        total_sent = (news_sent * 0.3) + (reddit_sent * 0.2) + (corp_sent * 0.5)

        sent_metadata = {
            'total_score': total_sent,
            'mentions': {'news': len(news_texts), 'reddit': len(reddit_texts), 'corp': len(corp_texts)}
        }

        # Unified Feed
        unified_feed = []
        for n in raw_data.get('news', []): unified_feed.append({'title': n['title'], 'link': n.get('link', '#'), 'date': n['date'], 'source': n['source'], 'type': 'News'})
        for r in raw_data.get('reddit', []): unified_feed.append({'title': r['title'], 'link': f"https://www.reddit.com/r/{r['subreddit']}", 'date': 'Social', 'source': f"r/{r['subreddit']}", 'type': 'Social'})
        for c in raw_data.get('corporate', []): unified_feed.append({'title': c['title'], 'link': '#', 'date': c['date'], 'source': 'NSE/BSE', 'type': 'Announcement'})

        # Final Signal with Contextual Weighting
        signal = compute_combined_signal(symbol, pred_data, sent_metadata, macro_data, sector_data, conv_data, meta_engine)
        
        render_params = {
            'symbol': symbol, 'prediction': pred_data, 'pcr': pcr_data, 'conviction': conv_data,
            'macro': macro_data, 'sector': sector_data, 'signal': signal,
            'backtest': bt_data, 'unified_feed': unified_feed,
            'sentiment': {'news': news_sent, 'reddit': reddit_sent, 'corporate': corp_sent, 'total': total_sent, 'mentions': sent_metadata['mentions']},
            'chart_data': chart_data, 'app_name': "StockIntel"
        }
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
    threading.Thread(target=load_models_task).start()
    app.run(debug=True, port=8080, use_reloader=False)
