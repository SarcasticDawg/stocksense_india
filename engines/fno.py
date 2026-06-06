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
            # The data is embedded in a JS initialization block
            # pcrRatio: [0, 0, ..., 0.69]
            match = re.search(r'pcrRatio:\s*\[([\d\.,]+)\]', response.text)
            if match:
                pcr_list = match.group(1).split(',')
                latest_pcr = float(pcr_list[-1])
                
                # Interpretation:
                # High PCR (> 1.2) -> Bullish Support (Market makers bought underlying to hedge puts)
                # Low PCR (< 0.7) -> Bearish Resistance (Market makers sold underlying to hedge calls)
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
