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
    
    # Try to find 'nightly_dump' from 2 days ago to check accuracy against today
    # (Since yesterday's dump already has today's close price context if run late, 
    # we look for the one before that to see if the prediction was right)
    lookback_date = (now_ist - timedelta(days=2)).strftime('%Y-%m-%d')
    last_dumps = list(collection.find({
        "timestamp": {"$regex": f"^{lookback_date}"},
        "type": "nightly_dump"
    }))

    if last_dumps:
        print(f"Found {len(last_dumps)} historical dumps. Backfilling outcomes for Meta-Model...")
        features_list = []
        outcomes_list = []
        
        for dump in last_dumps:
            symbol = dump['symbol']
            try:
                # Fetch actual closing price for 'today'
                stock = yf.Ticker(symbol)
                hist = stock.history(period="1d")
                if not hist.empty:
                    actual_close = hist['Close'].iloc[-1]
                    # Compare with the price at the time of that dump
                    prev_close = dump['data']['pred_data'].get('current_price', 0)
                    
                    if prev_close > 0:
                        outcome = 1 if actual_close > prev_close else 0
                        
                        # Use the features that were saved in that dump
                        if 'features' in dump:
                            features_list.append(dump['features'])
                            outcomes_list.append(outcome)
            except Exception as e:
                print(f"Error backfilling {symbol}: {e}")
        
        if len(features_list) > 10:
            meta.train(pd.DataFrame(features_list), pd.Series(outcomes_list))
        else:
            print("--- [Meta-Model] Not enough historical data for retraining yet. ---")

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

            # --- Meta-Model Feature Extraction & Final Signal ---
            # Extract features exactly as MetaModel expects
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
            
            pcr_val = float(pcr_data.get('pcr', 1.0)) if pcr_data else 1.0
            pcr_score = np.clip((pcr_val - 0.9) * 2.0, -1, 1)
            
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

            regime_val = 1 if macro_data.get('REGIME', {}).get('regime') == 'BULL' else (-1 if macro_data.get('REGIME', {}).get('regime') == 'BEAR' else 0)

            features = {
                'ai_price': float(ai_score),
                'sentiment': float(total_sent),
                'inst_flow': float(inst_flow_score),
                'pcr': float(pcr_score),
                'sector': float(sector_score),
                'regime': float(regime_val)
            }

            # Final Signal (using trained meta-model weights)
            final_signal = meta.get_signal(features)
            
            verdict = "HOLD"
            if final_signal > 0.3: verdict = "BUY"
            elif final_signal < -0.3: verdict = "SELL"

            result_doc = {
                "symbol": symbol,
                "timestamp": now_ist.isoformat(),
                "type": "nightly_dump",
                "features": features, # Saved for future training
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
                    "sentiment_metadata": sentiment_metadata,
                    "signal": {
                        "verdict": verdict,
                        "score": round(float(final_signal), 2),
                        "method": "ML Ensemble (Pre-Computed)",
                        "breakdown": {
                            "price": round(float(ai_score), 2),
                            "sentiment": round(float(total_sent), 2),
                            "institutional": round(float(inst_flow_score), 2),
                            "sector": round(float(sector_score), 2)
                        }
                    }
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
