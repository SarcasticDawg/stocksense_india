from flask import Flask, render_template, request, jsonify, redirect
import sys
import os
import threading
import gc
import traceback
import pytz
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add engines to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'engines'))

app = Flask(__name__)

@app.route('/health')
def health():
    return "OK", 200

# Config
USE_LEGACY_WEIGHTS = os.environ.get('USE_LEGACY_WEIGHTS', 'false').lower() == 'true'
MONGODB_URI = os.getenv("MONGODB_URI", "")

# Global state
_sentiment_analyzer = None
_meta_model = None
_mongodb_collection = None
_models_loading = False

def get_dashboard_mode():
    return "NIGHT"

def load_engines_task():
    global _sentiment_analyzer, _meta_model, _mongodb_collection, _models_loading
    if _models_loading: return
    _models_loading = True
    
    print("--- [Background] Initializing DB... ---")
    if MONGODB_URI:
        try:
            client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
            db = client.stocksense
            _mongodb_collection = db.stocksense_results
            print("--- [Background] MongoDB Connected. ---")
        except Exception as e:
            print(f"--- [Background] MongoDB Error: {e} ---")
            _mongodb_collection = None
            
    _models_loading = False

# Start DB connection in background
threading.Thread(target=load_engines_task, daemon=True).start()

def get_mongodb_collection():
    global _mongodb_collection
    return _mongodb_collection

def compute_combined_signal(symbol, pred_data, sentiment_metadata, macro_data, sector_data, conviction_data, meta_engine):
    import numpy as np
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

    sent_score = sentiment_metadata.get('total_score', 0.0)
    mentions = sentiment_metadata.get('mentions', {})
    
    sent_weight = 0.20
    sent_mode = "Full"
    
    if mentions.get('corp', 0) >= 1:
        sent_weight = 0.20
        sent_mode = "Full (Official)"
    elif mentions.get('news', 0) >= 3:
        sent_weight = 0.10
        sent_mode = "Partial (News)"
    elif mentions.get('social', 0) >= 3:
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
        'sector': float(sector_score),
        'regime': float(regime_val)
    }

    if meta_engine is None or meta_engine.model is None:
        total_score = (ai_score * 0.4) + (sent_score * sent_weight) + (inst_flow_score * 0.2)
        method = "Fallback Weights"
    else:
        total_score = meta_engine.predict(current_features)
        method = "ML Ensemble"

    verdict = "HOLD"
    if total_score > 0.3: verdict = "BUY"
    elif total_score < -0.3: verdict = "SELL"
    
    return {
        'verdict': verdict, 'score': round(float(total_score), 2), 'method': method,
        'sent_mode': sent_mode, 'breakdown': {
            'price': round(float(ai_score), 2), 'sentiment': round(float(sent_score), 2),
            'institutional': round(float(inst_flow_score), 2), 'conviction': round(float(conv_score), 2),
            'sector': round(float(sector_score), 2)
        }
    }

@app.route('/')
def home():
    try:
        import data_engine, market_context, pandas as pd
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
        
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'nse_stocks.csv')
        stocks_df = pd.read_csv(csv_path)
        stock_list = stocks_df.sort_values(by='Symbol').to_dict('records')
        
        return render_template('home.html', indices=index_summary, stock_list=stock_list, regime=regime, breadth=breadth, app_name="StockIntel")
    except Exception as e:
        print(f"Home error: {e}")
        return render_template('error.html', message="System error on home page.")

@app.route('/stock/<symbol>')
def stock_detail(symbol):
    try:
        if not symbol.endswith(".NS"): symbol += ".NS"
        mongo_col = get_mongodb_collection()
        
        if mongo_col is None:
            return render_template('error.html', message="Connecting to database... Please refresh.")

        latest_data = mongo_col.find_one({"symbol": symbol, "type": "nightly_dump"}, sort=[("timestamp", -1)])
        
        if latest_data:
            nightly = latest_data['data']
            news_items = nightly.get('raw_data_news', [])
            social_items = nightly.get('raw_data_social', [])
            corp_items = nightly.get('raw_data_corporate', [])
            sent_metadata = nightly.get('sentiment_metadata', {})
            
            signal = compute_combined_signal(symbol, nightly['pred_data'], sent_metadata, nightly['macro_data'], nightly['sector_data'], nightly['conv_data'], None)
            
            unified_feed = []
            for n in news_items: unified_feed.append({'title': n['title'], 'link': n.get('link', '#'), 'date': n['date'], 'source': n['source'], 'type': 'News'})
            for s in social_items: unified_feed.append({'title': s['title'], 'link': s.get('link', '#'), 'date': s['date'], 'source': s['source'], 'type': 'Social'})
            for c in corp_items: unified_feed.append({'title': c['title'], 'link': c.get('link', '#'), 'date': c['date'], 'source': 'NSE/BSE', 'type': 'Announcement'})

            return render_template('stock.html', 
                symbol=symbol, prediction=nightly.get('pred_data') or {}, 
                pcr=nightly.get('pcr_data') or {"pcr": "N/A"}, conviction=nightly.get('conv_data') or {"latest_pct": 0},
                macro=nightly.get('macro_data') or {}, sector=nightly.get('sector_data') or {"sector": "Unknown"},
                signal=signal, backtest=nightly.get('bt_data') or {"win_rate": 0},
                unified_feed=unified_feed, sent_data={'total': sent_metadata.get('total_score', 0), 'mentions': sent_metadata.get('mentions', {})},
                chart_data=nightly.get('chart_data', {"labels":[], "prices":[]}), mode='NIGHT (Cached)'
            )
        else:
            return render_template('error.html', message=f"No data found for {symbol} in Database.")
            
    except Exception as e:
        traceback.print_exc()
        return render_template('error.html', message=f"Internal Error: {e}")

@app.route('/stock_search')
def stock_search():
    symbol = request.args.get('symbol', '').upper()
    return redirect(f'/stock/{symbol}') if symbol else redirect('/')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
