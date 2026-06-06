import os
import sys
import pandas as pd
import time
from datetime import datetime

# Add engines to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'engines'))

import predictor

def run_batch_training():
    print("====================================================")
    print(f"StockSense India - Batch AI Trainer v2.0")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("====================================================\n")

    # 1. Load stock list
    try:
        stocks_df = pd.read_csv('data/nse_stocks.csv')
        symbols = stocks_df['Symbol'].tolist()
    except Exception as e:
        print(f"Error loading stock list: {e}")
        return

    # 2. Initialize Predictor
    ai_predictor = predictor.StockPredictor()
    
    total = len(symbols)
    print(f"Found {total} stocks to process.\n")

    # 3. Process sequentially to save memory
    for i, sym in enumerate(symbols):
        symbol = f"{sym}.NS"
        print(f"[{i+1}/{total}] Processing {symbol}...")
        
        # Check if model already exists for today (optional, but here we force retrain)
        start_time = time.time()
        try:
            ai_predictor.train(symbol)
            duration = round(time.time() - start_time, 2)
            print(f"--- SUCCESS: {symbol} trained in {duration}s ---\n")
        except Exception as e:
            print(f"--- FAILED: {symbol} Error: {e} ---\n")
        
        # Short pause to let system RAM stabilize
        time.sleep(2)

    print("====================================================")
    print(f"Batch Training Complete at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("====================================================")

if __name__ == "__main__":
    run_batch_training()
