import requests
from bs4 import BeautifulSoup
import re

class ScreenerScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def get_fundamental_data(self, symbol):
        """
        Scrapes financial ratios and 'About' info from Screener.in.
        """
        # Symbol cleaning (TCS.NS -> TCS)
        symbol_raw = symbol.replace(".NS", "").replace(".BO", "")
        url = f"https://www.screener.in/company/{symbol_raw}/consolidated/"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            if response.status_code != 200:
                # Try standalone if consolidated fails
                url = f"https://www.screener.in/company/{symbol_raw}/"
                response = requests.get(url, headers=self.headers, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 1. Extract Ratios
                ratios = {}
                ratio_list = soup.find('ul', id='top-ratios')
                if ratio_list:
                    items = ratio_list.find_all('li', class_='flex')
                    for item in items:
                        name_tag = item.find('span', class_='name')
                        value_tag = item.find('span', class_='number')
                        if name_tag and value_tag:
                            name = name_tag.get_text(strip=True)
                            value = value_tag.get_text(strip=True)
                            ratios[name] = value

                # 2. Extract About/Company Info
                about = ""
                about_section = soup.find('div', class_='company-profile') or soup.find('div', class_='about')
                if about_section:
                    about = about_section.get_text(strip=True)

                # 3. Extract Chart Data (Intercepting JSON in script tags if possible)
                # Note: Screener charts are complex. We'll extract the basic metadata.
                chart_metadata = {"status": "available"}
                
                return {
                    "ratios": ratios,
                    "about": about[:500] + "...", # Truncate for brevity
                    "source_url": url,
                    "success": True
                }
        except Exception as e:
            print(f"Screener scraping error for {symbol}: {e}")
            
        return {"success": False, "message": "Could not fetch data from Screener"}

if __name__ == "__main__":
    scraper = ScreenerScraper()
    print("Testing Screener Scraper for TITAN...")
    data = scraper.get_fundamental_data("TITAN")
    if data['success']:
        print("\nKey Ratios:")
        for k, v in data['ratios'].items():
            print(f"{k}: {v}")
        print(f"\nAbout: {data['about']}")
    else:
        print(data['message'])
