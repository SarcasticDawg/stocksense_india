import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import concurrent.futures

# Add the current directory and parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scraper
import macro
import corporate

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
    if df is None: return {"labels": [], "prices": [], "ohlc": []}
    
    # Format labels (Dates) and Prices (Close)
    labels = df.index.strftime('%Y-%m-%d').tolist()
    prices = df['Close'].round(2).tolist()
    
    ohlc = []
    for i, row in df.iterrows():
        ohlc.append({
            'x': i.strftime('%Y-%m-%d'),
            'o': round(float(row['Open']), 2),
            'h': round(float(row['High']), 2),
            'l': round(float(row['Low']), 2),
            'c': round(float(row['Close']), 2)
        })
        
    return {
        "labels": labels,
        "prices": prices,
        "ohlc": ohlc,
        "period": period
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
        df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'nse_stocks.csv'))
        stock_info = df[df['Symbol'] == symbol.replace(".NS", "")]
        if not stock_info.empty:
            return stock_info.iloc[0].to_dict()
    except Exception as e:
        print(f"Error reading nse_stocks.csv: {e}")
    return None

def get_all_raw_data(symbol):
    """
    Parallel fetch for all stock-related data.
    """
    clean_sym = symbol.replace(".NS", "")
    print(f"--- Parallel fetching all data for {symbol} ---")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        f_price = executor.submit(get_stock_data, symbol)
        f_scraped = executor.submit(scraper.get_all_scraped_data, clean_sym)
        f_macro = executor.submit(macro.get_macro_data)
        f_corp = executor.submit(corporate.CorporateIntelligence().get_announcements, clean_sym)
        
        price_data = f_price.result()
        scraped = f_scraped.result()
        macros = f_macro.result()
        corp_data = f_corp.result()

    return {
        'symbol': symbol,
        'info': get_stock_info(symbol),
        'price_data': price_data,
        'news': scraped['news'],
        'social': scraped['social'], # Merged Reddit + Screener
        'macro': macros,
        'corporate': corp_data
    }

if __name__ == "__main__":
    # Test parallel fetch
    data = get_all_raw_data("TITAN")
    print("\nData Summary:")
    print(f"Symbol: {data['symbol']}")
    print(f"Corporate Announcements: {len(data['corporate'])}")
    print(f"News Articles: {len(data['news'])}")
    print(f"Social Posts: {len(data['social'])}")
