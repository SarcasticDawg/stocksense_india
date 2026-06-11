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

@app.route('/diag')
def diag():
    uri_exists = bool(os.getenv("MONGODB_URI"))
    col = get_mongodb_collection()
    return jsonify({
        "uri_set": uri_exists,
        "connected": col is not None,
        "mode": get_dashboard_mode()
    })

# Config
MONGODB_URI = os.getenv("MONGODB_URI", "")

# Global state
_mongodb_collection = None
_models_loading = False

def get_dashboard_mode():
    return "NIGHT"

def load_engines_task():
    global _mongodb_collection, _models_loading
    if _models_loading: return
    _models_loading = True
    
    print("--- [Background] Connecting to MongoDB... ---")
    if MONGODB_URI:
        try:
            # We only need the MongoDB connection. NO ML MODELS.
            client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
            db = client.stocksense
            _mongodb_collection = db.stocksense_results
            print("--- [Background] MongoDB Connected Successfully. ---")
        except Exception as e:
            print(f"--- [Background] MongoDB Connection Failed: {e} ---")
            _mongodb_collection = None
    else:
        print("--- [Background] WARNING: MONGODB_URI not set. ---")
            
    _models_loading = False

# Start DB connection in background
threading.Thread(target=load_engines_task, daemon=True).start()

def get_mongodb_collection():
    global _mongodb_collection
    if _mongodb_collection is None:
        # Fallback: Attempt immediate connection if background thread hasn't finished
        if MONGODB_URI:
            try:
                client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=2000)
                db = client.stocksense
                _mongodb_collection = db.stocksense_results
            except:
                pass
    return _mongodb_collection

@app.route('/')
def home():
    try:
        import pandas as pd
        mongo_col = get_mongodb_collection()
        
        # 1. Load static stock list from CSV
        try:
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'nse_stocks.csv')
            stocks_df = pd.read_csv(csv_path)
            stock_list = stocks_df.sort_values(by='Symbol').to_dict('records')
        except:
            stock_list = []

        # 2. Get Market Context and Indices from MongoDB
        regime = {"regime": "NIGHT", "volatility": "Normal"}
        indices = {"NIFTY 50": {"value": "N/A", "change": 0}, "SENSEX": {"value": "N/A", "change": 0}}

        if mongo_col is not None:
            # Get latest Market Indices
            latest_indices = mongo_col.find_one({"type": "market_indices"}, sort=[("timestamp", -1)])
            if latest_indices:
                indices = latest_indices['data']

            # Get latest RELIANCE dump for regime
            latest_ref = mongo_col.find_one({"symbol": "RELIANCE.NS", "type": "nightly_dump"}, sort=[("timestamp", -1)])
            if latest_ref:
                macro_data = latest_ref['data'].get('macro_data', {})
                regime = macro_data.get('REGIME', regime)
        
        return render_template('home.html', 
            indices=indices, 
            stock_list=stock_list, 
            regime=regime, 
            breadth=None, 
            app_name="StockIntel"
        )
    except Exception as e:
        print(f"Home error: {e}")
        return render_template('error.html', message="System error or Database connecting...")

@app.route('/stock/<symbol>')
def stock_detail(symbol):
    try:
        import numpy as np
        if not symbol.endswith(".NS"): symbol += ".NS"
        mongo_col = get_mongodb_collection()
        
        if mongo_col is None:
            return render_template('error.html', message="Connecting to database... Please refresh.")

        # STRICT DB ONLY: Pull from nightly_dump
        latest_data = mongo_col.find_one({"symbol": symbol, "type": "nightly_dump"}, sort=[("timestamp", -1)])
        
        if latest_data:
            nightly = latest_data['data']
            
            # Check for fresher sentiment update (from 3hr updater)
            sentiment_update = mongo_col.find_one(
                {"symbol": symbol, "type": "sentiment_update"},
                sort=[("timestamp", -1)]
            )
            
            if sentiment_update and sentiment_update['timestamp'] > latest_data['timestamp']:
                print(f"Using fresher sentiment for {symbol}")
                sent_source = sentiment_update['data']
            else:
                sent_source = nightly
            
            news_items = sent_source.get('raw_data_news', [])
            social_items = sent_source.get('raw_data_social', [])
            corp_items = sent_source.get('raw_data_corporate', [])
            sent_metadata = sent_source.get('sentiment_metadata', {})
            
            # Pull pre-computed signal from the dump (Restores ML Intelligence)
            signal = nightly.get('signal', {
                'verdict': 'HOLD', 'score': 0.0, 'method': 'Legacy Fallback', 
                'breakdown': {'price': 0, 'sentiment': 0, 'institutional': 0, 'sector': 0}
            })
            
            unified_feed = []
            for n in news_items: unified_feed.append({'title': n['title'], 'link': n.get('link', '#'), 'date': n['date'], 'source': n['source'], 'type': 'News'})
            for s in social_items: unified_feed.append({'title': s['title'], 'link': s.get('link', '#'), 'date': s['date'], 'source': s['source'], 'type': 'Social'})
            for c in corp_items: unified_feed.append({'title': c['title'], 'link': c.get('link', '#'), 'date': c['date'], 'source': 'NSE/BSE', 'type': 'Announcement'})

            return render_template('stock.html', 
                symbol=symbol, prediction=pred_data, 
                pcr=nightly.get('pcr_data') or {"pcr": "N/A"}, conviction=nightly.get('conv_data') or {"latest_pct": 0},
                macro=nightly.get('macro_data') or {}, sector=nightly.get('sector_data') or {"sector": "Unknown"},
                signal=signal, backtest=nightly.get('bt_data') or {"win_rate": 0},
                unified_feed=unified_feed, sent_data={'total': sent_metadata.get('total_score', 0), 'mentions': sent_metadata.get('mentions', {})},
                chart_data=nightly.get('chart_data', {"labels":[], "prices":[]}), mode='DB-ONLY (Stable)'
            )
        else:
            return render_template('error.html', message=f"No data found for {symbol} in Database.")
            
    except Exception as e:
        traceback.print_exc()
        return render_template('error.html', message=f"Internal Error: {e}")

@app.route('/api/chart/<symbol>')
def api_chart(symbol):
    """STRICT DB ONLY: Pull chart data from MongoDB."""
    try:
        if not symbol.endswith(".NS"): symbol += ".NS"
        mongo_col = get_mongodb_collection()
        if mongo_col:
            doc = mongo_col.find_one({"symbol": symbol, "type": "nightly_dump"}, sort=[("timestamp", -1)])
            if doc:
                return jsonify(doc['data'].get('chart_data', {"labels":[], "prices":[]}))
    except:
        pass
    return jsonify({"labels":[], "prices":[]})

@app.route('/stock_search')
def stock_search():
    symbol = request.args.get('symbol', '').upper()
    return redirect(f'/stock/{symbol}') if symbol else redirect('/')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
