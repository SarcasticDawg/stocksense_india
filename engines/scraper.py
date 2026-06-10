import requests
from bs4 import BeautifulSoup
import feedparser
import re
import os
import sys
from datetime import datetime

# Path setup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

class MultiSourceScraper:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        self.rss_feeds = [
            ("google_news", "https://news.google.com/rss/search?q={query}+NSE+India+stock&hl=en-IN&gl=IN&ceid=IN:en"),
            ("economic_times", "https://economictimes.indiatimes.com/markets/stocks/news/rssfeeds/2146842.cms"),
            ("moneycontrol", "https://www.moneycontrol.com/rss/MCtopnews.xml"),
            ("business_standard", "https://www.business-standard.com/rss/markets-106.rss")
        ]

    def fetch_news(self, stock):
        articles = []
        query = stock.replace(".NS", "").replace(".BO", "")
        
        for source_name, url_template in self.rss_feeds:
            # Use query in URL if template allows, else static URL
            url = url_template.format(query=query) if "{query}" in url_template else url_template
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:15]:
                    title = getattr(entry, "title", "")
                    summary = getattr(entry, "summary", "") or title
                    
                    # Clean HTML from summary
                    summary_clean = BeautifulSoup(summary, "html.parser").get_text()

                    # Filtering for general feeds
                    if source_name in ("moneycontrol", "business_standard"):
                        if query.lower() not in (title + summary_clean).lower():
                            continue

                    articles.append({
                        'title': title,
                        'source': source_name.replace("_", " ").title(),
                        'link': entry.link,
                        'text': summary_clean[:400],
                        'date': datetime.now().strftime("%Y-%m-%d"), # Default to today if parse fails
                        'score': 1 # Default importance
                    })
            except Exception as e:
                print(f"RSS failed [{source_name}]: {e}")
        return articles

    def fetch_screener_discussions(self, stock):
        """
        Scrapes retail discussions from Screener.in
        """
        comments = []
        query = stock.replace(".NS", "").replace(".BO", "")
        url = f"https://www.screener.in/company/{query}/consolidated/"

        try:
            r = requests.get(url, headers=self.headers, timeout=10)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                # Selecting discussion blocks
                bodies = soup.select("div.comment-body") or soup.select("div.discussion")
                for body in bodies[:10]:
                    text = body.get_text(strip=True)
                    if len(text) < 30: continue
                    
                    # Try to find 'likes' or 'upvotes' in parent container
                    likes = 1
                    parent = body.find_parent()
                    if parent:
                        like_text = parent.get_text()
                        match = re.search(r"(\d+)\s*(likes|upvotes|up)", like_text, re.I)
                        if match: likes = int(match.group(1))

                    comments.append({
                        'title': "Screener Discussion",
                        'source': "Screener.in",
                        'link': url,
                        'text': text[:500],
                        'date': "Social",
                        'score': likes
                    })
        except Exception as e:
            print(f"Screener fetch failed: {e}")
        return comments

def get_all_scraped_data(symbol):
    scraper = MultiSourceScraper()
    news = scraper.fetch_news(symbol)
    social = scraper.fetch_screener_discussions(symbol)
    
    # Fallback if empty
    if not news:
        news = [{'title': f"Market context for {symbol}", 'source': "StockSense", 'link': "#", 'text': "Monitoring volatility.", 'date': datetime.now().strftime("%Y-%m-%d"), 'score': 1}]
        
    return {
        'news': news,
        'social': social
    }

if __name__ == "__main__":
    data = get_all_scraped_data("TCS")
    print(f"News: {len(data['news'])}, Social: {len(data['social'])}")
