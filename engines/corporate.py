import requests
import re
from bs4 import BeautifulSoup
import os
import json

class CorporateIntelligence:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
        }
        self.id_cache_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'screener_ids.json')
        self.id_cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.id_cache_path):
            try:
                with open(self.id_cache_path, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_cache(self):
        with open(self.id_cache_path, 'w') as f:
            json.dump(self.id_cache, f)

    def get_company_id(self, symbol):
        """
        Fetches and caches the Screener.in company ID for a symbol.
        """
        clean_sym = symbol.replace(".NS", "").upper()
        if clean_sym in self.id_cache:
            return self.id_cache[clean_sym]

        try:
            url = f"https://www.screener.in/company/{clean_sym}/"
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                match = re.search(r'data-company-id="(\d+)"', response.text)
                if match:
                    company_id = match.group(1)
                    self.id_cache[clean_sym] = company_id
                    self._save_cache()
                    return company_id
        except Exception as e:
            print(f"Error fetching Screener ID for {symbol}: {e}")
        
        return None

    def get_announcements(self, symbol):
        """
        Fetches recent corporate announcements from Screener.in
        """
        company_id = self.get_company_id(symbol)
        if not company_id:
            return []

        try:
            url = f"https://www.screener.in/announcements/recent/{company_id}/"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                items = []
                # Announcements are in <li class="overflow-wrap-anywhere">
                for li in soup.find_all('li', class_='overflow-wrap-anywhere'):
                    text = li.get_text(strip=True)
                    # Often contains date and title
                    # Example: "05 Jun - Announcement under Regulation 30..."
                    parts = text.split(' - ', 1)
                    date = parts[0] if len(parts) > 1 else ""
                    title = parts[1] if len(parts) > 1 else text
                    
                    items.append({
                        'date': date,
                        'title': title,
                        'source': 'NSE/BSE Announcement'
                    })
                return items
        except Exception as e:
            print(f"Error fetching announcements for {symbol}: {e}")
            
        return []

if __name__ == "__main__":
    ci = CorporateIntelligence()
    print("Testing Corporate Intelligence for TITAN...")
    ann = ci.get_announcements("TITAN")
    for a in ann[:5]:
        print(f"[{a['date']}] {a['title']}")
