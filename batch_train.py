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

# This comment is to force a new GA run. (2026-06-11)


# --- JSON File Paths ---
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
NIGHTLY_DUMP_HISTORY_PATH = os.path.join(DATA_DIR, 'nightly_dump_history.json')

# --- Helper functions for JSON I/O (copied from batch_runner.py) ---
def load_json_data(file_path, default_value={}):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True) # Ensure directory exists
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return default_value
    return default_value

# MongoDB Setup (Removed)

def run_batch_training():
    IST = pytz.timezone('Asia/Kolkata')
    now_ist = datetime.now(IST)
    print("====================================================\n")
    print(f"StockSense India - AI Training Pipeline v4.0")
    print(f"Started at: {now_ist.strftime('%Y-%m-%d %H:%M:%S')} IST")
    print("====================================================\n")
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
    print("
--[Phase 2] Retraining Meta-Model from JSON history ---")
    meta = meta_model.MetaModel()
    
    # Load historical features from JSON file
    historical_entries = load_json_data(NIGHTLY_DUMP_HISTORY_PATH, default_value=[])

    if historical_entries:
        print(f"Found {len(historical_entries)} historical records. Preparing for Meta-Model training...")
        features_list = []
        outcomes_list = []
        
        # Filter for entries from the last 30 days (more robust for retraining)
        cutoff_date_str = (now_ist - timedelta(days=30)).strftime('%Y-%m-%d')
        
        for entry in historical_entries:
            entry_date_str = entry['timestamp'].split('T')[0]
            if entry_date_str >= cutoff_date_str:
                 if 'features' in entry and 'outcome' in entry:
                    features_list.append(entry['features'])
                    outcomes_list.append(entry['outcome'])
        
        if len(features_list) > 10:
            meta.train(pd.DataFrame(features_list), pd.Series(outcomes_list))
            print("--- SUCCESS: Meta-Model weights optimized. ---")
        else:
            print("Notice: Not enough historical data from the specified period for retraining Meta-Model yet.")
    else:
        print("Notice: No historical data found in JSON for Meta-Model training.")

    print("====================================================\n")
    print(f"Training Pipeline Complete at: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')} IST")
    print("====================================================\n")

if __name__ == "__main__":
    run_batch_training()
