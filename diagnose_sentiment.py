import sys
import os

# Add engines to path
sys.path.append(os.path.join(os.getcwd(), 'engines'))

import data_engine
import scraper
import sentiment

def diagnose(symbol):
    print(f"--- Diagnosing {symbol} ---")
    
    # 1. Test Data Engine Aggregation
    print("\n1. Testing Scrapers...")
    raw_data = data_engine.get_all_raw_data(symbol)
    print(f"News found: {len(raw_data['news'])}")
    for n in raw_data['news'][:2]:
        print(f"  - Title: {n['title']}")
        print(f"  - Text Preview: {n['text'][:50]}...")
        
    print(f"Reddit posts found: {len(raw_data['reddit'])}")
    for r in raw_data['reddit'][:2]:
        print(f"  - Title: {r['title']}")

    # 2. Test Sentiment Analyzer
    print("\n2. Testing Sentiment Analyzer...")
    analyzer = sentiment.SentimentAnalyzer()
    print(f"Model loaded: {analyzer.is_loaded}")
    
    news_texts = [n['text'] for n in raw_data['news']]
    if news_texts:
        news_score = analyzer.analyze_batch(news_texts)
        print(f"News Sentiment Score: {news_score}")
    else:
        print("No news text to analyze.")
        
    reddit_texts = [p['title'] + " " + " ".join(p['comments']) for p in raw_data['reddit']]
    if reddit_texts:
        reddit_score = analyzer.analyze_batch(reddit_texts)
        print(f"Reddit Sentiment Score: {reddit_score}")
    else:
        print("No reddit text to analyze.")

if __name__ == "__main__":
    diagnose("TCS")
