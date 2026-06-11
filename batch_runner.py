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
import data_engine
import macro
import market_context
import sector
import fno
import conviction
import scraper
import sentiment
import backtester

# Fix: Resolved pymongo NotImplementedError (2026-06-11)

# MongoDB Setup
MONGODB_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGODB_URI) if MONGODB_URI else None
db = client.stocksense if client else None
collection = db.stocksense_results if db is not None else None

def run_batch():
    IST = pytz.timezone('Asia/Kolkata')
    now_ist = datetime.now(IST)
    
    print("====================================================")
    print(f"StockSense India - Execution Runner v4.0")
    print(f"Started at: {now_ist.strftime('%Y-%m-%d %H:%M:%S')} IST")
    print("====================================================\n")

    if collection is None:
        print("CRITICAL ERROR: MONGODB_URI not set. Execution halted.")
        return

    # 1. Load Market Context & Indices
    try:
        print("--- [Worker] Saving Global Market Indices... ---")
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
    except Exception as e:
        print(f"Error saving market indices: {e}")

    # 2. Fetch All Data for Nifty 50
    try:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'nse_stocks.csv')
        stocks_df = pd.read_csv(csv_path)
        symbols = stocks_df['Symbol'].tolist()[:50]
    except Exception as e:
        print(f"Error loading stock list: {e}")
        return

    ai_predictor = predictor.StockPredictor()
    meta = meta_model.MetaModel()
    sent_analyzer = sentiment.SentimentAnalyzer()
    retry_queue = []

    for sym in symbols:
        symbol = f"{sym}.NS"
        print(f"Processing {symbol}...")
        
        try:
            # --- PHASE 1: DATA GATHERING ---
            raw_data = data_engine.get_all_raw_data(symbol)
            macro_data = macro.get_macro_data()
            macro_data['REGIME'] = market_context.get_market_regime()
            
            sector_data = sector.get_sector_analysis(symbol)
            pcr_data = fno.get_stock_pcr(symbol)
            conv_data = conviction.get_delivery_data(symbol)
            chart_data = data_engine.get_chart_data(symbol)
            bt_data = backtester.StrategyBacktester().run_backtest(symbol)
            
            # --- PHASE 2: PREDICTION (Loads pre-trained models) ---
            pred_data = ai_predictor.get_prediction(symbol)
            
            # --- PHASE 3: SENTIMENT & ML ENSEMBLE ---
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
            
            # Signal Features
            import numpy as np
            lgbm_change = pred_data.get('price_change_pct', 0) if isinstance(pred_data, dict) else 0
            lgbm_score = np.clip(lgbm_change / 5.0, -1, 1)
            xgb_prob = pred_data.get('trend_prob_up', 0.5) if isinstance(pred_data, dict) else 0.5
            xgb_score = (xgb_prob - 0.5) * 2 
            ai_score = (lgbm_score * 0.6) + (xgb_score * 0.4)
            
            cash_net = 0.0
            inst = macro_data.get('INSTITUTIONAL', {})
            if isinstance(inst, dict) and isinstance(inst.get('cash'), dict):
                cash_net = float(inst['cash'].get('fii', 0)) + float(inst['cash'].get('dii', 0))
            inst_flow_score = np.clip(cash_net / 2500.0, -1, 1)
            
            pcr_val = float(pcr_data.get('pcr', 1.0)) if pcr_data else 1.0
            pcr_score = np.clip((pcr_val - 0.9) * 2.0, -1, 1)
            
            sector_rank_score = 0.0
            if isinstance(sector_data, dict):
                peers = sector_data.get('peers', [])
                clean_sym = symbol.replace(".NS", "")
                symbol_rank = 0
                for i, p in enumerate(peers):
                    if isinstance(p, dict) and p.get('symbol') == clean_sym:
                        symbol_rank = i
                        break
                if len(peers) > 0: sector_rank_score = (1.0 - (symbol_rank / len(peers))) * 2 - 1

            regime_val = 1 if macro_data.get('REGIME', {}).get('regime') == 'BULL' else (-1 if macro_data.get('REGIME', {}).get('regime') == 'BEAR' else 0)

            features = {
                'ai_price': float(ai_score), 'sentiment': float(total_sent),
                'inst_flow': float(inst_flow_score), 'pcr': float(pcr_score),
                'sector': float(sector_rank_score), 'regime': float(regime_val)
            }

            # Get final signal from trained weights
            final_signal = meta.get_signal(features)
            verdict = "HOLD"
            if final_signal > 0.3: verdict = "BUY"
            elif final_signal < -0.3: verdict = "SELL"

            # --- PHASE 4: SAVE TO MONGODB ---
            result_doc = {
                "symbol": symbol,
                "timestamp": now_ist.isoformat(),
                "type": "nightly_dump",
                "features": features,
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
                    "sentiment_metadata": {'total_score': total_sent, 'mentions': mentions},
                    "signal": {
                        "verdict": verdict,
                        "score": round(float(final_signal), 2),
                        "method": "ML Ensemble (Cached)",
                        "breakdown": {
                            "price": round(float(ai_score), 2),
                            "sentiment": round(float(total_sent), 2),
                            "institutional": round(float(inst_flow_score), 2),
                            "sector": round(float(sector_rank_score), 2)
                        }
                    }
                }
            }
            collection.insert_one(result_doc)
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    print("\n--- [Execution] Completed Successfully ---")

if __name__ == "__main__":
    run_batch()
