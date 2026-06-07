import requests
import re
from datetime import datetime

def get_delivery_data(symbol):
    """
    Scrapes delivery position data for a specific symbol from NiftyInvest.
    More stable than official NSE API due to lower bot protection.
    """
    clean_symbol = symbol.replace(".NS", "").upper()
    url = f"https://niftyinvest.com/delivery-position/{clean_symbol}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            # Look for the delivery percentage in the JS initialization block
            # Format usually: deliveryPercentage: [..., 45.2, 50.1]
            match = re.search(r'deliveryPercentage:\s*\[([\d\.,]+)\]', response.text)
            if match:
                values = match.group(1).split(',')
                latest_delivery = float(values[-1])
                
                # Compare against average (last 5 sessions)
                recent_values = [float(v) for v in values[-5:] if v.strip()]
                avg_delivery = sum(recent_values) / len(recent_values)
                
                # Verdict Logic
                verdict = "Normal"
                if latest_delivery > 60: verdict = "High Accumulation"
                elif latest_delivery > avg_delivery * 1.2: verdict = "Rising Conviction"
                elif latest_delivery < 20: verdict = "High Speculation"
                
                return {
                    "symbol": clean_symbol,
                    "latest_pct": round(latest_delivery, 2),
                    "avg_5d": round(avg_delivery, 2),
                    "verdict": verdict,
                    "is_high_conviction": latest_delivery > 50
                }
    except Exception as e:
        print(f"Error fetching delivery data for {symbol}: {e}")
        
    return None

if __name__ == "__main__":
    print("Testing stable delivery fetch for RELIANCE...")
    res = get_delivery_data("RELIANCE")
    print(res)
