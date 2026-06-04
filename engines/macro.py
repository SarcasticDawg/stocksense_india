import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup

def get_macro_data():
    """
    Fetches macro signals: USD/INR, Crude, S&P 500, NASDAQ, India VIX.
    """
    macro_symbols = {
        "USD_INR": "INR=X",
        "Crude_Oil": "BZ=F",
        "S&P_500": "^GSPC",
        "NASDAQ": "^IXIC",
        "India_VIX": "^INDIAVIX"
    }
    
    macro_data = {}
    for name, symbol in macro_symbols.items():
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="1mo")
            if not df.empty:
                # Store latest value and 1-month change
                latest = df['Close'].iloc[-1]
                prev = df['Close'].iloc[0]
                change_pct = ((latest - prev) / prev) * 100
                macro_data[name] = {
                    "latest": round(latest, 2),
                    "change_pct": round(change_pct, 2)
                }
        except Exception as e:
            print(f"Error fetching macro data for {name}: {e}")
            
    return macro_data

def get_fii_dii_data():
    """
    Scrapes FII/DII data from NSE.
    Note: High chance of blocking/DNS issues.
    """
    url = "https://www.nseindia.com/market-data/fii-dii-data"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        # In a real scenario, we might need a session to handle cookies for NSE
        # For now, we return a mock or try a simple request
        # Since NSE is restrictive, we'll return a placeholder if it fails
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            # Parse table logic here if reachable
            return {"status": "success", "data": "Table data would be here"}
    except Exception as e:
        print(f"Error fetching FII/DII data: {e}")
    
    return {"status": "unavailable", "message": "FII/DII data currently unavailable"}

if __name__ == "__main__":
    print("Fetching Macro Data...")
    macros = get_macro_data()
    for k, v in macros.items():
        print(f"{k}: {v}")
    
    print("\nFetching FII/DII Data...")
    fii_dii = get_fii_dii_data()
    print(fii_dii)
