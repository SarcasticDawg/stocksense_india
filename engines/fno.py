import requests
import re

def get_stock_pcr(symbol):
    """
    Scrapes Put/Call Ratio (PCR) for a specific stock from NiftyInvest.
    """
    # Clean symbol (NiftyInvest expects just the symbol, no .NS)
    clean_symbol = symbol.replace(".NS", "").upper()
    url = f"https://niftyinvest.com/put-call-ratio/{clean_symbol}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            # New regex: handle optional spaces, quotes, and 'null' values
            pattern = r'pcrRatio:\s*\[(.*?)\]'
            match = re.search(pattern, response.text)
            
            if match:
                raw_list = match.group(1)
                # Split and clean
                items = [x.strip().replace("'", "").replace('"', "") for x in raw_list.split(',')]
                
                # Filter out 'null' and get the last valid number
                valid_numbers = []
                for item in items:
                    if item.lower() != 'null' and item != '':
                        try:
                            valid_numbers.append(float(item))
                        except:
                            pass
                
                if valid_numbers:
                    latest_pcr = valid_numbers[-1]
                    
                    sentiment = "Neutral"
                    if latest_pcr > 1.1: sentiment = "Bullish"
                    elif latest_pcr < 0.8: sentiment = "Bearish"
                    
                    return {
                        "pcr": latest_pcr,
                        "sentiment": sentiment
                    }
    except Exception as e:
        print(f"Error fetching PCR for {symbol}: {e}")
        
    return None

if __name__ == "__main__":
    print("Testing PCR fetch for TITAN...")
    result = get_stock_pcr("TITAN")
    print(result)
