import pandas as pd
import numpy as np

def compute_rsi(df, window=14):
    """Computes the Relative Strength Index (RSI)."""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(df, fast=12, slow=26, signal=9):
    """Computes MACD, Signal Line, and Histogram."""
    exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def compute_bollinger_bands(df, window=20, num_std=2):
    """Computes Bollinger Bands."""
    ma = df['Close'].rolling(window=window).mean()
    std = df['Close'].rolling(window=window).std()
    upper = ma + (std * num_std)
    lower = ma - (std * num_std)
    return upper, ma, lower

def add_technical_indicators(df):
    """Adds all required technical indicators to the dataframe."""
    if df is None or df.empty:
        return None
    
    # Ensure dataframe is sorted by date
    df = df.sort_index()
    
    # RSI
    df['RSI'] = compute_rsi(df)
    
    # MACD
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = compute_macd(df)
    
    # Bollinger Bands
    df['BB_Upper'], df['BB_MA'], df['BB_Lower'] = compute_bollinger_bands(df)
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Moving Averages
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Volume Change
    df['Vol_Change_Pct'] = df['Volume'].pct_change() * 100
    
    return df

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import data_engine
    
    print("Testing technical indicators on TCS...")
    tcs_df = data_engine.get_stock_data("TCS")
    if tcs_df is not None:
        tcs_df = add_technical_indicators(tcs_df)
        print(tcs_df[['Close', 'RSI', 'MACD', 'BB_Position', 'MA50', 'MA200']].tail())
