import os
import sys
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
import pytz

# Add engines to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'engines'))

import data_engine
import sentiment
import scraper

# MongoDB Setup
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGODB_URI)
db = client.stocksense
collection = db.stocksense_results

def run_3hr_update():
    IST = pytz.timezone('Asia/Kolkata')
    now_ist = datetime.now(IST)

    print(f"--- [3-Hour Updater] Starting at {now_ist.strftime('%Y-%m-%d %H:%M:%S')} IST ---")

    try:
        stocks_df = pd.read_csv('data/nse_stocks.csv')
        symbols = stocks_df['Symbol'].tolist()[:50]
    except Exception as e:
        print(f"Error loading stock list: {e}")
        return

    sent_analyzer = sentiment.SentimentAnalyzer()

    for sym in symbols:
        symbol = f"{sym}.NS"
        print(f"Refreshing sentiment for {symbol}...")

        try:
            raw_data = data_engine.get_all_raw_data(symbol)
            news_items = raw_data.get('news', [])
            social_items = raw_data.get('social', [])
            corp_items = raw_data.get('corporate', [])

            news_sent = sent_analyzer.analyze_batch(news_items, is_news=True)
            social_sent = sent_analyzer.analyze_batch(social_items)
            corp_sent = sent_analyzer.analyze_batch(corp_items, is_corporate=True)
            
            total_sent = (news_sent * 0.3) + (social_sent * 0.2) + (corp_sent * 0.5)

            sentiment_metadata = {
                'total_score': total_sent,
                'mentions': {
                    'news': len(news_items), 
                    'social': len(social_items), 
                    'corp': len(corp_items),
                    'total': len(news_items) + len(social_items) + len(corp_items)
                }
            }

            update_doc = {
                "symbol": symbol,
                "timestamp": now_ist.isoformat(),
                "type": "sentiment_update",
                "data": {
                    "raw_data_news": news_items,
                    "raw_data_social": social_items,
                    "raw_data_corporate": corp_items,
                    "sentiment_metadata": sentiment_metadata
                }
            }
            
            collection.insert_one(update_doc)

        except Exception as e:
            print(f"Error refreshing {symbol}: {e}")

    print("--- [3-Hour Updater] Completed ---")

if __name__ == "__main__":
    run_3hr_update()
