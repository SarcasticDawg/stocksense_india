import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add the current directory and parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scraper
import macro

def get_stock_data(symbol, period="1y"):
    """
    Fetches historical OHLCV data for a given symbol.
    """
    try:
        if not symbol.endswith(".NS") and not symbol.startswith("^"):
            symbol += ".NS"
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        if df.empty:
            print(f"Warning: No data found for {symbol}")
            return None
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def get_chart_data(symbol, period="3mo"):
    """
    Returns historical price points formatted for Chart.js.
    """
    df = get_stock_data(symbol, period=period)
    if df is None: return {"labels": [], "prices": []}
    
    # Format labels (Dates) and Prices (Close)
    labels = df.index.strftime('%Y-%m-%d').tolist()
    prices = df['Close'].round(2).tolist()
    
    return {
        "labels": labels,
        "prices": prices
    }

def get_market_indices():
    """
    Fetches data for Nifty 50 and Sensex.
    Returns:
        dict: Dictionary with dataframes for Nifty and Sensex
    """
    indices = {
        "Nifty 50": "^NSEI",
        "Sensex": "^BSESN"
    }
    data = {}
    for name, symbol in indices.items():
        # Fetching only 5 days for quick homepage summary
        data[name] = get_stock_data(symbol, period="5d")
    return data

def get_stock_info(symbol):
    """
    Looks up stock info from the NSE master list.
    """
    try:
        df = pd.read_csv('data/nse_stocks.csv')
        stock_info = df[df['Symbol'] == symbol.replace(".NS", "")]
        if not stock_info.empty:
            return stock_info.iloc[0].to_dict()
    except Exception as e:
        print(f"Error reading nse_stocks.csv: {e}")
    return None

def get_all_raw_data(symbol):
    """
    Aggregates all raw data for a stock as per Week 1 requirements.
    """
    print(f"--- Aggregating data for {symbol} ---")
    
    # 1. Stock Price Data
    price_data = get_stock_data(symbol)
    
    # 2. NSE Master List Info
    info = get_stock_info(symbol)
    
    # 3 & 4. News and Reddit
    scraped = scraper.get_all_scraped_data(symbol.replace(".NS", ""))
    
    # 5. Macro Data
    macros = macro.get_macro_data()
    
    return {
        'symbol': symbol,
        'info': info,
        'price_data': price_data,
        'news': scraped['news'],
        'reddit': scraped['reddit'],
        'macro': macros
    }

if __name__ == "__main__":
    # Test all-in-one aggregation
    data = get_all_raw_data("TCS")
    print("\nData Summary:")
    print(f"Symbol: {data['symbol']}")
    print(f"Sector: {data['info']['Sector'] if data['info'] else 'Unknown'}")
    print(f"Price Records: {len(data['price_data']) if data['price_data'] is not None else 0}")
    print(f"News Articles: {len(data['news'])}")
    print(f"Reddit Posts: {len(data['reddit'])}")
    print(f"Macro Signals: {list(data['macro'].keys())}")
