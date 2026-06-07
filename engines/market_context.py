import yfinance as yf
import pandas as pd
import numpy as np
import os
import sys

# Path setup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import data_engine
import technical

def get_market_regime():
    """
    Classifies the current market regime for Nifty 50.
    Regimes: BULL, BEAR, SIDEWAYS
    Criteria: 50/200 DMA cross and Volatility Percentile.
    """
    try:
        # Fetch Nifty 50 data
        df = data_engine.get_stock_data("^NSEI", period="1y")
        if df is None or df.empty: return "UNKNOWN"
        
        # Calculate Moving Averages
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        
        # Calculate Volatility (20-day rolling std of returns)
        df['Returns'] = df['Close'].pct_change()
        df['Vol'] = df['Returns'].rolling(window=20).std()
        
        latest = df.iloc[-1]
        
        # Regime Logic
        is_bull = latest['Close'] > latest['MA200'] and latest['MA50'] > latest['MA200']
        is_bear = latest['Close'] < latest['MA200'] and latest['MA50'] < latest['MA200']
        
        vol_threshold = df['Vol'].quantile(0.7) # High volatility threshold
        is_high_vol = latest['Vol'] > vol_threshold
        
        regime = "SIDEWAYS"
        if is_bull: regime = "BULL"
        elif is_bear: regime = "BEAR"
        
        return {
            "regime": regime,
            "volatility": "HIGH" if is_high_vol else "NORMAL",
            "nifty_close": round(latest['Close'], 2),
            "ma50": round(latest['MA50'], 2),
            "ma200": round(latest['MA200'], 2)
        }
    except Exception as e:
        print(f"Error in regime detection: {e}")
        return {"regime": "UNKNOWN", "volatility": "NORMAL"}

def get_market_breadth():
    """
    Calculates Advance/Decline ratio and % of stocks above MAs for Nifty 50.
    """
    try:
        stocks_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'nse_stocks.csv'))
        symbols = stocks_df['Symbol'].tolist()
        
        advances = 0
        declines = 0
        above_ma50 = 0
        above_ma200 = 0
        total = 0
        
        # We only check a subset for speed in prototype if list is long, 
        # but Nifty 50 is small enough.
        for sym in symbols[:50]: # Ensure we stay within Nifty 50
            df = data_engine.get_stock_data(f"{sym}.NS", period="1y")
            if df is not None and len(df) > 200:
                latest_close = df['Close'].iloc[-1]
                prev_close = df['Close'].iloc[-2]
                ma50 = df['Close'].rolling(window=50).mean().iloc[-1]
                ma200 = df['Close'].rolling(window=200).mean().iloc[-1]
                
                if latest_close > prev_close: advances += 1
                else: declines += 1
                
                if latest_close > ma50: above_ma50 += 1
                if latest_close > ma200: above_ma200 += 1
                total += 1
        
        if total == 0: return None
        
        return {
            "ad_ratio": round(advances / declines, 2) if declines > 0 else advances,
            "pct_above_ma50": round((above_ma50 / total) * 100, 1),
            "pct_above_ma200": round((above_ma200 / total) * 100, 1),
            "total_tracked": total
        }
    except Exception as e:
        print(f"Error in market breadth: {e}")
        return None

if __name__ == "__main__":
    print("Regime:", get_market_regime())
    print("Breadth:", get_market_breadth())
