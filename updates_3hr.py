import os
import sys
import pandas as pd
import json # New import for JSON handling
from datetime import datetime
import pytz

# Add engines to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'engines'))

import data_engine
import sentiment
import scraper

# --- JSON File Paths ---
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
NIGHTLY_DUMP_PATH = os.path.join(DATA_DIR, 'nightly_dump.json') # To read existing stock data
SENTIMENT_UPDATE_PATH = os.path.join(DATA_DIR, 'sentiment_update.json')

# --- Helper functions for JSON I/O (copied from batch_runner.py) ---
def load_json_data(file_path, default_value={}):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return default_value
    return default_value

def save_json_data(file_path, data):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

# MongoDB Setup (Removed)

def run_3hr_update():
    IST = pytz.timezone('Asia/Kolkata')
    now_ist = datetime.now(IST)

    print(f"--- [3-Hour Updater] Starting at {now_ist.strftime('%Y-%m-%d %H:%M:%S')} IST ---")

    try:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'nse_stocks.csv')
        stocks_df = pd.read_csv(csv_path)
        symbols = stocks_df['Symbol'].tolist()[:50]
    except Exception as e:
        print(f"Error loading stock list: {e}")
        return

    sent_analyzer = sentiment.SentimentAnalyzer()
    
    # Load existing sentiment data
    current_sentiment_data = load_json_data(SENTIMENT_UPDATE_PATH)

    for sym in symbols:
        symbol = f"{sym}.NS"
        print(f"Refreshing sentiment for {symbol}...")

        try:
            raw_data = data_engine.get_all_raw_data(symbol)
            news_items = raw_data.get('news', [])
            social_items = raw_data.get('social', [])
            corp_items = raw_data.get('corporate', [])
            corp_titles = [c['title'] for c in corp_items]

            news_sent = sent_analyzer.analyze_batch(news_items, is_news=True) if news_items else 0
            social_sent = sent_analyzer.analyze_batch(social_items) if social_items else 0
            corp_sent = sent_analyzer.analyze_batch(corp_titles, is_corporate=True) if corp_titles else 0
            
            total_sent = (news_sent * 0.4) + (social_sent * 0.2) + (corp_sent * 0.4)

            sentiment_metadata = {
                'total_score': total_sent,
                'news_score': news_sent,
                'social_score': social_sent,
                'corp_score': corp_sent,
                'mentions': {
                    'news': len(news_items), 
                    'social': len(social_items), 
                    'corp': len(corp_items),
                    'total': len(news_items) + len(social_items) + len(corp_items)
                }
            }
            
            # Save to SENTIMENT_UPDATE_PATH (latest sentiment data)
            current_sentiment_data[symbol] = {
                "timestamp": now_ist.isoformat(),
                "data": {
                    "raw_data_news": news_items,
                    "raw_data_social": social_items,
                    "raw_data_corporate": corp_items,
                    "sentiment_metadata": sentiment_metadata
                }
            }

        except Exception as e:
            print(f"Error refreshing {symbol}: {e}")

    # Save the updated sentiment data after processing all symbols
    save_json_data(SENTIMENT_UPDATE_PATH, current_sentiment_data)

    print("--- [3-Hour Updater] Completed ---")

if __name__ == "__main__":
    run_3hr_update()
