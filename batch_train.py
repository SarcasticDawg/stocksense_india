import os
import sys
import pandas as pd
import time
from datetime import datetime
import joblib

# Add engines to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'engines'))

import predictor
import meta_model

def run_batch_training():
    print("====================================================")
    print(f"StockSense India - Professional Batch Trainer v3.0")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("====================================================\n")

    # 1. Load stock list
    try:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'nse_stocks.csv')
        stocks_df = pd.read_csv(csv_path)
        symbols = stocks_df['Symbol'].tolist()
    except Exception as e:
        print(f"Error loading stock list: {e}")
        return

    # 2. Initialize Engines
    ai_predictor = predictor.StockPredictor()
    meta = meta_model.MetaModel()
    
    total = len(symbols)
    print(f"Found {total} stocks to process.\n")

    # 3. Process sequentially
    for i, sym in enumerate(symbols):
        symbol = f"{sym}.NS"
        print(f"[{i+1}/{total}] Training LightGBM/XGBoost for {symbol}...")
        
        start_time = time.time()
        try:
            ai_predictor.train(symbol)
            duration = round(time.time() - start_time, 2)
            print(f"--- SUCCESS: {symbol} ready in {duration}s ---")
        except Exception as e:
            print(f"--- FAILED: {symbol} Error: {e} ---")
        
        time.sleep(1) # Faster with LightGBM

    print("\n--- [Meta-Model] Finalizing Weight Blending ---")
    # If cache exists, try to refresh meta-model
    if os.path.exists(meta.cache_path):
        df = pd.read_csv(meta.cache_path)
        if 'outcome' in df.columns and df['outcome'].notnull().sum() > 10:
            features = df[['ai_price', 'sentiment', 'inst_flow', 'pcr', 'sector', 'regime']]
            outcomes = df['outcome']
            meta.train(features, outcomes)
        else:
            print("Notice: Meta-model needs more historical outcome data to train. Using manual defaults.")

    print("\n====================================================")
    print(f"Batch Training Complete at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("====================================================")

if __name__ == "__main__":
    run_batch_training()
