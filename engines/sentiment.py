import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class SentimentAnalyzer:
    def __init__(self, model_name="ProsusAI/finbert"):
        try:
            print(f"Loading {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.is_loaded = True
        except Exception as e:
            print(f"Error loading FinBERT: {e}")
            self.is_loaded = False

    def get_sentiment(self, text):
        """
        Returns sentiment score between -1 and 1.
        FinBERT returns [positive, negative, neutral].
        """
        if not self.is_loaded:
            return 0.0 # Neutral fallback
            
        try:
            # Truncate text to avoid model limits
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().numpy()[0]
            
            # Score: Pos - Neg (ignoring neutral for the magnitude)
            # FinBERT labels: 0 -> positive, 1 -> negative, 2 -> neutral
            score = probs[0] - probs[1]
            return float(score)
        except Exception as e:
            print(f"Sentiment error: {e}")
            return 0.0

    def analyze_batch(self, texts):
        if not texts: return 0.0
        scores = [self.get_sentiment(t) for n, t in enumerate(texts) if t]
        return np.mean(scores) if scores else 0.0

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    test_texts = [
        "The company reported a 20% increase in quarterly profit, beating analyst estimates.",
        "Stock prices plunged as the company announced a massive layoff and reduced guidance."
    ]
    for t in test_texts:
        print(f"Text: {t[:50]}...")
        print(f"Score: {analyzer.get_sentiment(t)}")
