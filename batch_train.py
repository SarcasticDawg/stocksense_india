import os
import sys
import pandas as pd
import yfinance as yf
from pymongo import MongoClient
from datetime import datetime, timedelta
import pytz
import time

# Add engines to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'engines'))

import predictor
import meta_model

def run_batch_training():
    IST = pytz.timezone('Asia/Kolkata')
    now_ist = datetime.now(IST)
    
    print("====================================================")
    print(f"StockSense India - AI Training Pipeline v4.0")
    print(f"Started at: {now_ist.strftime('%Y-%m-%d %H:%M:%S')} IST")
    print("====================================================\n")

    # MongoDB Setup for Meta-Model
    MONGODB_URI = os.getenv("MONGODB_URI")
    client = None
    collection = None
    if MONGODB_URI:
        client = MongoClient(MONGODB_URI)
        db = client.stocksense
        collection = db.stocksense_results

    # 1. RETRAIN INDIVIDUAL STOCK MODELS (LGBM/XGB)
    try:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'nse_stocks.csv')
        stocks_df = pd.read_csv(csv_path)
        symbols = stocks_df['Symbol'].tolist()[:50] # Focus on Nifty 50
    except Exception as e:
        print(f"Error loading stock list: {e}")
        return

    ai_predictor = predictor.StockPredictor()
    
    total = len(symbols)
    print(f"--- [Phase 1] Training individual stock models for {total} stocks ---")
    for i, sym in enumerate(symbols):
        symbol = f"{sym}.NS"
        print(f"[{i+1}/{total}] Training {symbol}...")
        try:
            ai_predictor.train(symbol)
        except Exception as e:
            print(f"Error training {symbol}: {e}")
        time.sleep(1) # Safety delay

    # 2. RETRAIN META-MODEL (THE BRAIN)
    if collection is not None:
        print("\n--- [Phase 2] Retraining Meta-Model from MongoDB history ---")
        meta = meta_model.MetaModel()
        
        # Look back 2 days to compare old predictions with today's actual close
        lookback_date = (now_ist - timedelta(days=2)).strftime('%Y-%m-%d')
        last_dumps = list(collection.find({
            "timestamp": {"$regex": f"^{lookback_date}"},
            "type": "nightly_dump"
        }))

        if last_dumps:
            print(f"Found {len(last_dumps)} historical records. Backfilling outcomes...")
            features_list = []
            outcomes_list = []
            
            for dump in last_dumps:
                symbol = dump['symbol']
                try:
                    stock = yf.Ticker(symbol)
                    hist = stock.history(period="1d")
                    if not hist.empty:
                        actual_close = hist['Close'].iloc[-1]
                        prev_close = dump['data']['pred_data'].get('current_price', 0)
                        
                        if prev_close > 0:
                            outcome = 1 if actual_close > prev_close else 0
                            if 'features' in dump:
                                features_list.append(dump['features'])
                                outcomes_list.append(outcome)
                except Exception as e:
                    print(f"Error backfilling outcome for {symbol}: {e}")
            
            if len(features_list) > 10:
                meta.train(pd.DataFrame(features_list), pd.Series(outcomes_list))
                print("--- SUCCESS: Meta-Model weights optimized. ---")
            else:
                print("Notice: Not enough historical data to retrain Meta-Model yet.")
    else:
        print("\n--- [Phase 2] Skipped: MONGODB_URI not set for Meta-Model training. ---")

    print("\n====================================================")
    print(f"Training Pipeline Complete at: {datetime.now(IST).strftime('%H:%M:%S')} IST")
    print("====================================================")

if __name__ == "__main__":
    run_batch_training()
