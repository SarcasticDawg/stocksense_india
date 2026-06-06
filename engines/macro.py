import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np

def get_fii_dii_data():
    """
    Fetches FII/DII data from Sensibull API with full F&O context.
    """
    url = "https://oxide.sensibull.com/v1/compute/cache/fii_dii_daily"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            json_data = response.json()
            data_dict = json_data.get('data', {})
            if not data_dict:
                return None
            
            # Get the latest date
            latest_date = sorted(data_dict.keys())[-1]
            day_data = data_dict[latest_date]
            
            # 1. Cash Market
            cash = day_data.get('cash', {})
            fii_cash_net = cash.get('fii', {}).get('buy_sell_difference', 0)
            dii_cash_net = cash.get('dii', {}).get('buy_sell_difference', 0)
            
            # 2. Futures Segment (FII)
            future_fii = day_data.get('future', {}).get('fii', {}).get('quantity-wise', {})
            fii_future_view = future_fii.get('net_view_summary', future_fii.get('net_view', 'Neutral'))
            fii_future_strength = future_fii.get('net_view_strength_summary', future_fii.get('net_view_strength', 'Low'))
            
            # 3. Options Segment (FII)
            option_fii = day_data.get('option', {}).get('fii', {})
            fii_option_view = option_fii.get('overall_net_oi_change_view_summary', 'Neutral')
            fii_option_strength = option_fii.get('overall_net_oi_change_view_summary_strength', 'Low')
            
            # 4. Client (Retail) - Used as a contrarian indicator
            client_future = day_data.get('future', {}).get('client', {}).get('quantity-wise', {})
            client_view = client_future.get('net_view_summary', 'Neutral')
            
            # 5. Pro (Proprietary)
            pro_future = day_data.get('future', {}).get('pro', {}).get('quantity-wise', {})
            pro_view = pro_future.get('net_view', 'Neutral')
            
            return {
                "date": latest_date,
                "cash": {"fii": round(fii_cash_net, 2), "dii": round(dii_cash_net, 2)},
                "fii_future": {"view": fii_future_view, "strength": fii_future_strength},
                "fii_option": {"view": fii_option_view, "strength": fii_option_strength},
                "client_view": client_view,
                "pro_view": pro_view,
                "overall_sentiment": fii_future_view # Using Future View as baseline
            }
    except Exception as e:
        print(f"Error fetching detailed FII/DII data: {e}")
    
    return None

def get_macro_data():
    """
    Fetches macro signals: USD/INR, Crude, S&P 500, NASDAQ, India VIX.
    """
    macro_symbols = {
        "USD_INR": "INR=X",
        "Crude_Oil": "BZ=F",
        "SP500": "^GSPC",
        "NASDAQ": "^IXIC",
        "India_VIX": "^INDIAVIX"
    }
    
    macro_data = {}
    for name, symbol in macro_symbols.items():
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="1mo")
            if not df.empty:
                latest = df['Close'].iloc[-1]
                prev = df['Close'].iloc[0]
                change_pct = ((latest - prev) / prev) * 100
                macro_data[name] = {
                    "latest": round(latest, 2),
                    "change_pct": round(change_pct, 2)
                }
        except Exception as e:
            print(f"Error fetching macro data for {name}: {e}")
            
    # Add Detailed Institutional Data
    fii_dii = get_fii_dii_data()
    if fii_dii:
        macro_data['INSTITUTIONAL'] = fii_dii
            
    return macro_data

if __name__ == "__main__":
    print("Fetching Detailed Macro Data...")
    macros = get_macro_data()
    if 'INSTITUTIONAL' in macros:
        inst = macros['INSTITUTIONAL']
        print(f"\nDate: {inst['date']}")
        print(f"FII Cash: ₹{inst['cash']['fii']} Cr")
        print(f"FII Future View: {inst['fii_future']['view']} ({inst['fii_future']['strength']})")
        print(f"FII Option View: {inst['fii_option']['view']} ({inst['fii_option']['strength']})")
        print(f"Client (Retail) View: {inst['client_view']}")
