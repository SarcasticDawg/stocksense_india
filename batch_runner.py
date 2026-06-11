import os
import sys
import pandas as pd
import yfinance as yf
import json # New import for JSON handling
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

# --- JSON File Paths ---
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
NIGHTLY_DUMP_PATH = os.path.join(DATA_DIR, 'nightly_dump.json')
NIGHTLY_DUMP_HISTORY_PATH = os.path.join(DATA_DIR, 'nightly_dump_history.json')
MARKET_INDICES_PATH = os.path.join(DATA_DIR, 'market_indices.json')

# --- Helper functions for JSON I/O ---
def load_json_data(file_path, default_value={}):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True) # Ensure directory exists
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return default_value # Return default if file is empty or corrupted
    return default_value

def save_json_data(file_path, data):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True) # Ensure directory exists
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def append_json_history(file_path, data_entry):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True) # Ensure directory exists
    history = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    history.append(data_entry)
    with open(file_path, 'w') as f:
        json.dump(history, f, indent=4)

# MongoDB Setup (Removed)

def run_batch():
    IST = pytz.timezone('Asia/Kolkata')
    now_ist = datetime.now(IST)
    
    print("====================================================\n")
    print(f"StockSense India - Execution Runner v4.0")
    print(f"Started at: {now_ist.strftime('%Y-%m-%d %H:%M:%S')} IST")
    print("====================================================\n")

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
        
        save_json_data(MARKET_INDICES_PATH, indices_summary)
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
    
    # Load existing nightly dump data
    current_nightly_dump = load_json_data(NIGHTLY_DUMP_PATH)

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
            corp_titles = [c['title'] for c in raw_data.get('corporate', [])]
            
            news_score = sent_analyzer.analyze_batch(news_texts, is_news=True) if news_texts else 0
            social_score = sent_analyzer.analyze_batch(social_texts) if social_texts else 0
            corp_score = sent_analyzer.analyze_batch(corp_titles, is_corporate=True) if corp_titles else 0
            
            total_sent = (news_score * 0.4) + (social_score * 0.2) + (corp_score * 0.4)
            
            mentions = {
                'corp': len(raw_data.get('corporate', [])),
                'news': len(raw_data['news']),
                'social': len(raw_data['social']),
                'total': len(raw_data.get('corporate', [])) + len(raw_data['news']) + len(raw_data['social'])
            }
            
            sentiment_metadata = {
                'total_score': total_sent,
                'news_score': news_score,
                'social_score': social_score,
                'corp_score': corp_score,
                'mentions': mentions
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

            # --- PHASE 4: SAVE TO JSON FILES ---
            # Save to NIGHTLY_DUMP_PATH (latest data)
            current_nightly_dump[symbol] = {
                "timestamp": now_ist.isoformat(),
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

            # Append to NIGHTLY_DUMP_HISTORY_PATH (for Meta-Model training)
            append_json_history(NIGHTLY_DUMP_HISTORY_PATH, {
                "symbol": symbol,
                "timestamp": now_ist.isoformat(),
                "features": features,
                "outcome": 1 if verdict == "BUY" else (0 if verdict == "SELL" else 0.5) # Simplified outcome for history
            })
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    # Save the updated nightly dump after processing all symbols
    save_json_data(NIGHTLY_DUMP_PATH, current_nightly_dump)

    print("\n====================================================\n")

if __name__ == "__main__":
    run_batch()
