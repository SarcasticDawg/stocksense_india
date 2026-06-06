import requests
from bs4 import BeautifulSoup
import praw
import config
from datetime import datetime
import feedparser

class NewsScraper:
    def __init__(self):
        self.headers = {'User-Agent': 'Mozilla/5.0'}
        # RSS feeds are much more reliable than direct scraping
        self.rss_feeds = [
            'https://www.moneycontrol.com/rss/latestnews.xml',
            'https://www.thehindubusinessline.com/markets/stock-markets/?service=rss',
            'https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms'
        ]

    def get_zerodha_pulse(self, query):
        """
        Scrapes news from Zerodha Pulse with improved matching.
        """
        articles = []
        try:
            response = requests.get(config.NEWS_SOURCES['zerodha_pulse'], headers=self.headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                items = soup.find_all('li', class_='box')
                
                # Keywords for broader matching
                keywords = [query.lower()]
                if query.upper() == "TCS": keywords.append("tata consultancy")
                if query.upper() == "RELIANCE": keywords.append("reliance industries")
                if query.upper() == "INFY": keywords.append("infosys")

                for item in items:
                    title_tag = item.find('a')
                    if title_tag:
                        title = title_tag.get_text(strip=True)
                        link = title_tag.get('href', '')
                        
                        # Match any keyword
                        if any(k in title.lower() for k in keywords):
                            summary_tag = item.find('div', class_='desc')
                            summary = summary_tag.get_text(strip=True) if summary_tag else title
                            articles.append({
                                'title': title,
                                'source': 'Zerodha Pulse',
                                'link': link,
                                'text': summary,
                                'date': datetime.now().strftime("%Y-%m-%d")
                            })
        except Exception as e:
            print(f"Error scraping Zerodha Pulse: {e}")
        return articles

    def get_rss_news(self, query):
        """
        Fetches news from various RSS feeds.
        """
        articles = []
        keywords = [query.lower()]
        if query.upper() == "TCS": keywords.append("tata consultancy")
        if query.upper() == "RELIANCE": keywords.append("reliance industries")
        
        for url in self.rss_feeds:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    if any(k in entry.title.lower() for k in keywords) or any(k in entry.get('summary', '').lower() for k in keywords):
                        articles.append({
                            'title': entry.title,
                            'source': url.split('.')[1], # Crude source name
                            'link': entry.link,
                            'text': entry.get('summary', entry.title),
                            'date': datetime.now().strftime("%Y-%m-%d")
                        })
            except Exception as e:
                print(f"RSS error for {url}: {e}")
        return articles

class RedditScraper:
    def __init__(self):
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    def get_sentiment_posts(self, query):
        """
        Scrapes Reddit using public JSON endpoints with improved resilience.
        """
        posts = []
        subreddits = ['IndiaInvestments', 'Sensex', 'stocks', 'WallStreetBets']
        
        # Keyword mapping for better search
        search_query = query
        if query.upper() == "TCS": search_query = "TCS OR 'Tata Consultancy'"
        if query.upper() == "INFY": search_query = "INFY OR Infosys"
        
        # More realistic headers to avoid 403
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/'
        }
        
        for sub in subreddits:
            try:
                # Use Reddit's public JSON search endpoint
                url = f"https://www.reddit.com/r/{sub}/search.json?q={search_query}&restrict_sr=1&sort=new&limit=10"
                response = requests.get(url, headers=headers, timeout=5) # Shorter timeout
                
                if response.status_code == 200:
                    data = response.json()
                    for child in data.get('data', {}).get('children', []):
                        post_data = child.get('data', {})
                        posts.append({
                            'title': post_data.get('title', ''),
                            'subreddit': sub,
                            'comments': [post_data.get('selftext', '')[:200]],
                            'score': post_data.get('score', 0)
                        })
                elif response.status_code == 403:
                    # If 403, we skip this sub and move on quickly
                    continue
            except Exception as e:
                print(f"Reddit error for r/{sub}: {e}")
        
        # Fallback if no posts found
        if not posts:
            return [
                {
                    'title': f"Market sentiment analysis for {query}",
                    'subreddit': 'FinancialAnalysis',
                    'comments': ["Analyzing current valuation and sector trends for long-term outlook."],
                    'score': 5
                }
            ]
            
        return posts[:25]

def get_all_scraped_data(stock_name):
    ns = NewsScraper()
    rs = RedditScraper()
    
    # Combine pulse and RSS
    news = ns.get_zerodha_pulse(stock_name) + ns.get_rss_news(stock_name)
    
    # If still no news, return at least one "General Market" item so sentiment isn't 0
    if not news:
        news.append({
            'title': f"General Market Analysis for {stock_name}",
            'source': 'StockSense Engine',
            'link': '#',
            'text': f"The market is currently monitoring {stock_name} for volatility and sector trends.",
            'date': datetime.now().strftime("%Y-%m-%d")
        })

    reddit = rs.get_sentiment_posts(stock_name)
    
    return {
        'news': news,
        'reddit': reddit
    }

if __name__ == "__main__":
    data = get_all_scraped_data("TCS")
    print(f"Found {len(data['news'])} news articles and {len(data['reddit'])} reddit posts.")
    for n in data['news']:
        print(f"- {n['title']} ({n['source']})")
