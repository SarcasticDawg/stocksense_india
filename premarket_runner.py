import os
import sys
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
import pytz

# Add engines to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'engines'))

import scraper
import sentiment
import meta_model
import data_engine

# MongoDB Setup
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGODB_URI)
db = client.stocksense
collection = db.stocksense_results

def run_premarket():
    IST = pytz.timezone('Asia/Kolkata')
    now_ist = datetime.now(IST)
    today_str = now_ist.strftime('%Y-%m-%d')

    print(f"--- [Premarket Runner] Starting at {now_ist.strftime('%Y-%m-%d %H:%M:%S')} IST ---")

    # 1. Load Nifty 50 symbols
    try:
        stocks_df = pd.read_csv('data/nse_stocks.csv')
        symbols = stocks_df['Symbol'].tolist()[:50]
    except Exception as e:
        print(f"Error loading stock list: {e}")
        return

    sent_analyzer = sentiment.SentimentAnalyzer()
    meta = meta_model.MetaModel()

    for sym in symbols:
        symbol = f"{sym}.NS"
        print(f"Refreshing sentiment for {symbol}...")

        try:
            # 2. Find last nightly dump
            last_dump = collection.find_one(
                {"symbol": symbol, "type": "nightly_dump"},
                sort=[("timestamp", -1)]
            )

            if not last_dump:
                print(f"No nightly dump found for {symbol}. Skipping.")
                continue

            # 3. Fetch Fresh Sentiment
            raw_scraped = scraper.get_all_scraped_data(sym)
            news_texts = [n['text'] for n in raw_scraped['news']]
            social_texts = [s['text'] for s in raw_scraped['social']]
            
            news_score = sent_analyzer.analyze_batch(news_texts) if news_texts else 0
            social_score = sent_analyzer.analyze_batch(social_texts) if social_texts else 0
            
            total_sent = (news_score * 0.7) + (social_score * 0.3)
            
            mentions = {
                'corp': len(last_dump['data'].get('raw_data_corporate', [])),
                'news': len(raw_scraped['news']),
                'social': len(raw_scraped['social'])
            }
            
            updated_sentiment = {
                'total_score': total_sent,
                'mentions': mentions
            }

            # 4. Recalculate Combined Signal (we need this to save 'features' for tomorrow's retraining)
            # Reconstruct current features for Meta-Model
            pred_data = last_dump['data']['pred_data']
            macro_data = last_dump['data']['macro_data']
            sector_data = last_dump['data']['sector_data']
            conv_data = last_dump['data']['conv_data']
            pcr_data = last_dump['data']['pcr_data']
            
            # Logic mirrored from app.py:compute_combined_signal
            lgbm_change = pred_data.get('price_change_pct', 0)
            lgbm_score = (lgbm_change / 5.0) # Simplified normalization
            xgb_prob = pred_data.get('trend_prob_up', 0.5)
            xgb_score = (xgb_prob - 0.5) * 2
            ai_score = (lgbm_score * 0.6) + (xgb_score * 0.4)
            
            inst = macro_data.get('INSTITUTIONAL', {})
            cash_net = float(inst.get('cash', {}).get('fii', 0)) + float(inst.get('cash', {}).get('dii', 0))
            inst_flow_score = (cash_net / 2500.0)
            
            pcr_val = float(pcr_data.get('pcr', 1.0))
            pcr_score = (pcr_val - 0.9) * 2.0
            
            conv_val = conv_data.get('latest_pct', 30)
            conv_score = (conv_val - 40) / 20.0
            
            regime_data = macro_data.get('REGIME', {})
            regime_val = 1 if regime_data.get('regime') == 'BULL' else (-1 if regime_data.get('regime') == 'BEAR' else 0)

            # Note: sector_score is complex to re-calc exactly here without full peer list, 
            # we'll use the one from nightly dump as proxy.
            
            current_features = {
                'ai_price': float(ai_score),
                'sentiment': float(total_sent),
                'inst_flow': float(inst_flow_score),
                'pcr': float(pcr_score),
                'conviction': float(conv_score),
                'sector': 0.0, # Placeholder
                'regime': float(regime_val)
            }

            # 5. Save as morning_prediction
            morning_doc = {
                "symbol": symbol,
                "timestamp": now_ist.isoformat(),
                "type": "morning_prediction",
                "features": current_features, # For tomorrow's meta-training
                "data": {
                    "price_data_current": pred_data.get('current_price'),
                    "sentiment_metadata": updated_sentiment,
                    "nightly_data": last_dump['data'] # Keep reference to all other pre-computed data
                }
            }
            
            collection.insert_one(morning_doc)

        except Exception as e:
            print(f"Error refreshing {symbol}: {e}")

    print("--- [Premarket Runner] Completed ---")

if __name__ == "__main__":
    run_premarket()
