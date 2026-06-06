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

    def analyze_batch(self, texts, weights=None):
        """
        Analyzes a batch of texts.
        If weights are provided (e.g., Reddit upvotes), it computes a weighted average.
        weights should be a list of numerical values corresponding to each text.
        """
        if not texts:
            return 0.0
            
        scores = [self.get_sentiment(t) for t in texts]
        
        if weights:
            # Apply log(upvotes + 1) weighting as suggested
            # Ensure weights are positive
            processed_weights = [np.log1p(max(0, w)) for w in weights]
            total_weight = sum(processed_weights)
            
            if total_weight == 0:
                return np.mean(scores)
                
            weighted_scores = [s * w for s, w in zip(scores, processed_weights)]
            return sum(weighted_scores) / total_weight
        else:
            return np.mean(scores) if scores else 0.0

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    test_texts = [
        "The company reported a 20% increase in quarterly profit, beating analyst estimates.",
        "Stock prices plunged as the company announced a massive layoff and reduced guidance.",
        "This stock is going to the moon! HODL diamond hands!",
        "Bagholders are dumping their shares after the crash."
    ]
    for t in test_texts:
        print(f"Text: {t[:50]}...")
        print(f"Score: {analyzer.get_sentiment(t)}")
