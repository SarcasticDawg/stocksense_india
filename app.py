from flask import Flask, render_template, request, jsonify, redirect
import sys
import os
import threading
import gc
import traceback
import pytz
from datetime import datetime
import json # New import for JSON handling
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add engines to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'engines'))

app = Flask(__name__)

# --- JSON File Paths (copied from batch_runner.py) ---
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
NIGHTLY_DUMP_PATH = os.path.join(DATA_DIR, 'nightly_dump.json')
SENTIMENT_UPDATE_PATH = os.path.join(DATA_DIR, 'sentiment_update.json')
MARKET_INDICES_PATH = os.path.join(DATA_DIR, 'market_indices.json')

# --- Helper functions for JSON I/O (copied from batch_runner.py) ---
def load_json_data(file_path, default_value={}):
    if not os.path.exists(DATA_DIR):
        # In a web app context, this directory should exist if batch_runner ran
        print(f"WARNING: Data directory {DATA_DIR} not found.")
        return default_value
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"WARNING: JSONDecodeError in {file_path}. Returning default.")
                return default_value
    return default_value

# Config (MONGODB_URI removed)

# Global state (MongoDB removed)
_models_loading = False # Still relevant for background tasks if any

def get_dashboard_mode():
    # This logic was previously based on current time; now it's fixed to NIGHT for stability
    return "NIGHT"

# MongoDB connection code removed (load_engines_task, get_mongodb_collection removed)
# The MongoDB connection is now handled by directly loading JSON files.

@app.route('/diag')
def diag():
    # Diagnostic route adapted for JSON files
    nightly_exists = os.path.exists(NIGHTLY_DUMP_PATH)
    sentiment_exists = os.path.exists(SENTIMENT_UPDATE_PATH)
    indices_exists = os.path.exists(MARKET_INDICES_PATH)
    
    nightly_data = load_json_data(NIGHTLY_DUMP_PATH)
    sentiment_data = load_json_data(SENTIMENT_UPDATE_PATH)
    market_indices_data = load_json_data(MARKET_INDICES_PATH)

    return jsonify({
        "data_dir_exists": os.path.exists(DATA_DIR),
        "nightly_dump_file_exists": nightly_exists,
        "sentiment_update_file_exists": sentiment_exists,
        "market_indices_file_exists": indices_exists,
        "nightly_dump_records": len(nightly_data) if isinstance(nightly_data, dict) else 0,
        "sentiment_update_records": len(sentiment_data) if isinstance(sentiment_data, dict) else 0,
        "market_indices_data": market_indices_data,
        "mode": get_dashboard_mode()
    })

@app.route('/')
def home():
    try:
        import pandas as pd # Used for stocks_df
        
        # 1. Load static stock list from CSV
        try:
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'nse_stocks.csv')
            stocks_df = pd.read_csv(csv_path)
            stock_list = stocks_df.sort_values(by='Symbol').to_dict('records')
        except Exception as e:
            print(f"Error loading stock list CSV: {e}")
            stock_list = []

        # 2. Get Market Context and Indices from JSON files
        regime = {"regime": "Analyzing...", "volatility": "Normal"}
        indices = {"NIFTY 50": {"value": "N/A", "change": 0}, "SENSEX": {"value": "N/A", "change": 0}}
        
        market_indices_data = load_json_data(MARKET_INDICES_PATH)
        if market_indices_data:
            # Create a case-insensitive map for the ticker
            indices = {}
            for k, v in market_indices_data.items():
                indices[k.upper()] = v
        
        # Get latest RELIANCE dump for regime (from nightly_dump.json)
        nightly_dump_data = load_json_data(NIGHTLY_DUMP_PATH)
        reliance_dump = nightly_dump_data.get("RELIANCE.NS", {})
        if reliance_dump:
            # The regime is stored inside macro_data in the JSON
            macro_data = reliance_dump.get('data', {}).get('macro_data', {})
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
        traceback.print_exc() # Print full traceback for debugging
        return render_template('error.html', message="System error or Data not available. Please try again.")

@app.route('/stock/<symbol>')
def stock_detail(symbol):
    try:
        import numpy as np # Used for signal computation
        if not symbol.endswith(".NS"): symbol += ".NS"
        
        nightly_dump_data = load_json_data(NIGHTLY_DUMP_PATH)
        sentiment_update_data = load_json_data(SENTIMENT_UPDATE_PATH)

        # STRICT DB ONLY: Pull from nightly_dump.json
        latest_data_entry = nightly_dump_data.get(symbol)
        
        if latest_data_entry:
            nightly = latest_data_entry['data']
            
            # Check for fresher sentiment update (from sentiment_update.json)
            fresher_sentiment_entry = sentiment_update_data.get(symbol)
            
            sent_source = nightly # Default to nightly sentiment
            if fresher_sentiment_entry and fresher_sentiment_entry['timestamp'] > latest_data_entry['timestamp']:
                print(f"Using fresher sentiment for {symbol}")
                sent_source = fresher_sentiment_entry['data']
            
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

            sent_display_data = {
                'total': sent_metadata.get('total_score', 0),
                'news': sent_metadata.get('news_score', 0),
                'social': sent_metadata.get('social_score', 0),
                'corporate': sent_metadata.get('corp_score', 0),
                'mentions': sent_metadata.get('mentions', {})
            }

            return render_template('stock.html', 
                symbol=symbol, prediction=nightly.get('pred_data', {}), 
                pcr=nightly.get('pcr_data') or {"pcr": "N/A"}, conviction=nightly.get('conv_data') or {"latest_pct": 0},
                macro=nightly.get('macro_data') or {}, sector=nightly.get('sector_data') or {"sector": "Unknown"},
                signal=signal, backtest=nightly.get('bt_data') or {"win_rate": 0},
                unified_feed=unified_feed, sent_data=sent_display_data,
                chart_data=nightly.get('chart_data', {"labels":[], "prices":[]}), mode='DB-ONLY (Stable)'
            )
        else:
            return render_template('error.html', message=f"No data found for {symbol}. Data files might be missing or empty.")
            
    except Exception as e:
        traceback.print_exc()
        return render_template('error.html', message=f"Internal Error: {e}")

@app.route('/api/chart/<symbol>')
def api_chart(symbol):
    """STRICT JSON ONLY: Pull chart data from nightly_dump.json."""
    try:
        if not symbol.endswith(".NS"): symbol += ".NS"
        nightly_dump_data = load_json_data(NIGHTLY_DUMP_PATH)
        doc = nightly_dump_data.get(symbol)
        if doc:
            return jsonify(doc['data'].get('chart_data', {"labels":[], "prices":[]}))
    except Exception as e:
        print(f"API Chart error: {e}")
        traceback.print_exc()
    return jsonify({"labels":[], "prices":[]})

@app.route('/stock_search')
def stock_search():
    symbol = request.args.get('symbol', '').upper()
    return redirect(f'/stock/{symbol}') if symbol else redirect('/')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
