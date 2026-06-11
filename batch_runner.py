import os
import sys
import pandas as pd
import yfinance as yf
from pymongo import MongoClient
from datetime import datetime, timedelta
import pytz
import joblib

# Add engines to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'engines'))

import predictor
import meta_model
import data_engine
import macro
import market_context
import sector
import fno
import conviction
import scraper
import sentiment
import backtester

# MongoDB Setup
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGODB_URI)
db = client.stocksense
collection = db.stocksense_results

def run_batch():
    IST = pytz.timezone('Asia/Kolkata')
    now_ist = datetime.now(IST)
    today_str = now_ist.strftime('%Y-%m-%d')
    yesterday_str = (now_ist - timedelta(days=1)).strftime('%Y-%m-%d')

    print(f"--- [Batch Runner] Starting at {now_ist.strftime('%Y-%m-%d %H:%M:%S')} IST ---")

    # 1. Backfill Outcomes & Retrain Meta-Model
    meta = meta_model.MetaModel()
    
    # Try to find yesterday's 'morning_prediction' to check accuracy
    # morning_prediction was saved by premarket_runner
    last_predictions = list(collection.find({
        "timestamp": {"$regex": f"^{yesterday_str}"},
        "type": "morning_prediction"
    }))

    if last_predictions:
        print(f"Found {len(last_predictions)} yesterday predictions. Backfilling outcomes...")
        features_list = []
        outcomes_list = []
        
        for pred in last_predictions:
            symbol = pred['symbol']
            try:
                # Fetch actual closing price for 'today'
                stock = yf.Ticker(symbol)
                hist = stock.history(period="1d")
                if not hist.empty:
                    actual_close = hist['Close'].iloc[-1]
                    prev_close = pred['data']['price_data_current'] # Price at premarket
                    
                    outcome = 1 if actual_close > prev_close else 0
                    
                    # Store features for retraining
                    features_list.append(pred['features'])
                    outcomes_list.append(outcome)
                    
                    # Update the record in Mongo
                    collection.update_one(
                        {"_id": pred["_id"]},
                        {"$set": {"actual_outcome": outcome, "actual_close": actual_close}}
                    )
            except Exception as e:
                print(f"Error backfilling {symbol}: {e}")
        
        if len(features_list) > 5:
            meta.train(pd.DataFrame(features_list), pd.Series(outcomes_list))

    # 2. Fetch All Data for Nifty 50
    try:
        # Save Global Market Indices first
        print("--- [Batch Runner] Saving Global Market Indices... ---")
        indices_raw = data_engine.get_market_indices()
        indices_summary = {}
        for name, df in indices_raw.items():
            if df is not None and not df.empty:
                latest = df['Close'].iloc[-1]
                prev = df['Close'].iloc[0]
                change = ((latest - prev) / prev) * 100
                indices_summary[name] = {"value": round(float(latest), 2), "change": round(float(change), 2)}
        
        collection.insert_one({
            "type": "market_indices",
            "timestamp": now_ist.isoformat(),
            "data": indices_summary
        })

        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'nse_stocks.csv')
        stocks_df = pd.read_csv(csv_path)
        symbols = stocks_df['Symbol'].tolist()[:50] # Nifty 50
    except Exception as e:
        print(f"Error loading stock list: {e}")
        return

    ai_predictor = predictor.StockPredictor()
    sent_analyzer = sentiment.SentimentAnalyzer()
    retry_queue = []

    for sym in symbols:
        symbol = f"{sym}.NS"
        print(f"Processing {symbol}...")
        
        try:
            # Re-train models locally as per existing batch_train logic
            ai_predictor.train(symbol)
            
            # Fetch All Fresh Data
            raw_data = data_engine.get_all_raw_data(symbol)
            macro_data = macro.get_macro_data()
            regime_data = market_context.get_market_regime()
            macro_data['REGIME'] = regime_data
            
            sector_data = sector.get_sector_analysis(symbol)
            pcr_data = fno.get_stock_pcr(symbol)
            conv_data = conviction.get_delivery_data(symbol)
            chart_data = data_engine.get_chart_data(symbol)
            bt_data = backtester.StrategyBacktester().run_backtest(symbol)
            
            # Prediction
            pred_data = ai_predictor.get_prediction(symbol)
            
            # Monitor for failures to retry later
            if pcr_data is None or conv_data is None:
                retry_queue.append({
                    'symbol': symbol,
                    'clean_sym': sym,
                    'pcr_missing': pcr_data is None,
                    'conv_missing': conv_data is None
                })

            # Prepare Sentiment Metadata
            news_texts = [n['text'] for n in raw_data['news']]
            social_texts = [s['text'] for s in raw_data['social']]
            
            news_score = sent_analyzer.analyze_batch(news_texts) if news_texts else 0
            social_score = sent_analyzer.analyze_batch(social_texts) if social_texts else 0
            
            total_sent = (news_score * 0.7) + (social_score * 0.3)
            
            mentions = {
                'corp': len(raw_data.get('corporate', [])),
                'news': len(raw_data['news']),
                'social': len(raw_data['social'])
            }
            
            sentiment_metadata = {
                'total_score': total_sent,
                'mentions': mentions
            }

            result_doc = {
                "symbol": symbol,
                "timestamp": now_ist.isoformat(),
                "type": "nightly_dump",
                "data": {
                    "raw_data_news": raw_data['news'],
                    "raw_data_social": raw_data['social'],
                    "raw_data_corporate": raw_data['corporate'],
                    "macro_data": macro_data,
                    "sector_data": sector_data,
                    "pcr_data": pcr_data,
                    "conv_data": conv_data,
                    "chart_data": chart_data,
                    "bt_data": bt_data,
                    "pred_data": pred_data,
                    "sentiment_metadata": sentiment_metadata
                }
            }
            
            collection.insert_one(result_doc)
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    # 3. Self-Healing Retry Loop
    if retry_queue:
        print(f"\n--- [Batch Runner] Healing {len(retry_queue)} stocks with missing data... ---")
        time.sleep(30) # Cool down period to avoid IP blocks
        
        for item in retry_queue:
            symbol = item['symbol']
            try:
                updates = {}
                if item['pcr_missing']:
                    pcr = fno.get_stock_pcr(symbol)
                    if pcr: 
                        updates["data.pcr_data"] = pcr
                        print(f"Healed PCR for {symbol}")
                
                if item['conv_missing']:
                    conv = conviction.get_delivery_data(symbol)
                    if conv: 
                        updates["data.conv_data"] = conv
                        print(f"Healed Delivery for {symbol}")
                
                if updates:
                    collection.update_one(
                        {"symbol": symbol, "type": "nightly_dump", "timestamp": now_ist.isoformat()},
                        {"$set": updates}
                    )
            except Exception as e:
                print(f"Retry failure for {symbol}: {e}")

    print("--- [Batch Runner] Completed ---")

if __name__ == "__main__":
    run_batch()
