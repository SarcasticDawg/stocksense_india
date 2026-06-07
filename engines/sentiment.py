from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import os
import sys

# Add current directory to path to import finance_lexicon
sys.path.append(os.path.dirname(__file__))
try:
    from finance_lexicon import finance_lexicon
except ImportError:
    finance_lexicon = {}

class SentimentAnalyzer:
    def __init__(self):
        print("Loading VADER Sentiment Engine with Finance Lexicon...")
        self.analyzer = SentimentIntensityAnalyzer()
        self.analyzer.lexicon.update(finance_lexicon)
        self.is_loaded = True

    def classify_news_type(self, text):
        """
        Distinguishes between 'reporting' (past facts) and 'signal' (future outlook).
        """
        reporting_keywords = [
            "rose", "fell", "gained", "lost", "closed at",
            "ended", "finished", "today", "yesterday",
            "in today's session", "on monday", "on tuesday",
            "posts revenue", "reports profit", "quarterly results"
        ]

        forward_keywords = [
            "will", "expects", "guides", "forecast", "outlook",
            "plans to", "likely to", "may", "could", "next quarter",
            "announces", "targets", "projects", "warns of",
            "raises guidance", "cuts guidance", "capex", "expansion"
        ]
        
        text_lower = text.lower()
        reporting_score = sum(1 for k in reporting_keywords if k in text_lower)
        forward_score = sum(1 for k in forward_keywords if k in text_lower)
        
        if forward_score > reporting_score:
            return "signal"
        elif reporting_score > forward_score:
            return "reporting"
        else:
            return "neutral"

    def get_sentiment(self, text):
        """
        Returns sentiment score between -1 and 1.
        """
        if not text:
            return 0.0
            
        try:
            score = self.analyzer.polarity_scores(text)
            compound = score['compound']  # -1 to +1
            return float(compound)
        except Exception as e:
            print(f"Sentiment error: {e}")
            return 0.0

    def analyze_batch(self, texts, weights=None, is_corporate=False, is_news=False):
        """
        Analyzes a batch of texts.
        is_corporate: applies a high-conviction multiplier.
        is_news: applies tense-based filtering (signal vs reporting).
        """
        if not texts:
            return 0.0
            
        scores = [self.get_sentiment(t) for t in texts]
        
        # Calculate dynamic weights if it's news
        if is_news:
            news_weights = []
            for t in texts:
                ntype = self.classify_news_type(t)
                if ntype == "signal": news_weights.append(1.0)
                elif ntype == "reporting": news_weights.append(0.2) # Heavily downweight
                else: news_weights.append(0.5)
            
            if weights:
                weights = [w * nw for w, nw in zip(weights, news_weights)]
            else:
                weights = news_weights

        # Base multiplier for corporate announcements
        conviction_mult = 2.5 if is_corporate else 1.0
        
        if weights:
            processed_weights = [np.log1p(max(0, w)) for w in weights]
            total_weight = sum(processed_weights)
            if total_weight == 0: 
                avg_score = np.mean(scores) * conviction_mult
                return float(np.clip(avg_score, -1, 1))
            
            weighted_scores = [s * w for s, w in zip(scores, processed_weights)]
            final_score = (sum(weighted_scores) / total_weight) * conviction_mult
            return float(np.clip(final_score, -1, 1))
        else:
            if not scores: return 0.0
            avg_score = np.mean(scores) * conviction_mult
            return float(np.clip(avg_score, -1, 1))

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    test_texts = [
        "The company reported a 20% increase in quarterly profit, beating analyst estimates.",
        "Stock prices plunged as the company announced a massive layoff and reduced guidance.",
        "Management will expand its footprint into new markets next quarter.",
        "The index closed at 23400 today after gaining 1%."
    ]
    for t in test_texts:
        ntype = analyzer.classify_news_type(t)
        print(f"Text: {t[:50]}... Type: {ntype}")
        print(f"Score: {analyzer.get_sentiment(t)}")
