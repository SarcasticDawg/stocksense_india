import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
import gc

class MetaModel:
    def __init__(self, model_dir='models'):
        # Ensure model_dir is an absolute path relative to the project root
        if not os.path.isabs(model_dir):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.model_dir = os.path.join(base_dir, model_dir)
        else:
            self.model_dir = model_dir
            
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        self.model_path = os.path.join(self.model_dir, 'meta_model.joblib')
        self.cache_path = os.path.join(self.model_dir, 'feature_cache.csv')
        self.model = self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            return joblib.load(self.model_path)
        return None

    def train(self, historical_features, historical_outcomes):
        """
        Trains the Logistic Regression Meta-model.
        historical_features: DataFrame with columns [ai_score, sentiment, inst_flow, pcr, sector, regime]
        historical_outcomes: Series of 1 (Price Up) or 0 (Price Down)
        """
        print("--- [Meta-Model] Training Machine-Learned Weights ---")
        model = LogisticRegression(class_weight='balanced')
        model.fit(historical_features, historical_outcomes)
        joblib.dump(model, self.model_path)
        self.model = model
        
        # Log coefficients as learned weights
        coeffs = dict(zip(historical_features.columns, model.coef_[0]))
        print(f"--- [Meta-Model] Learned Weights: {coeffs} ---")

    def predict(self, current_features):
        """
        Produces a blended signal score between -1 and 1.
        current_features: list or dict of feature values
        """
        if self.model is None:
            return None
            
        # Ensure input is a DataFrame with correct feature names
        if isinstance(current_features, dict):
            df = pd.DataFrame([current_features])
        else:
            df = pd.DataFrame([current_features])
            
        # Probability of 'Up' (Class 1)
        prob_up = self.model.predict_proba(df)[0][1]
        
        # Scale 0..1 to -1..1
        return (prob_up * 2) - 1

    def cache_features(self, symbol, features, outcome=None):
        """
        Stores daily features for future retraining/backtesting.
        """
        row = features.copy()
        row['symbol'] = symbol
        row['timestamp'] = pd.Timestamp.now()
        if outcome is not None:
            row['outcome'] = outcome
            
        df = pd.DataFrame([row])
        if os.path.exists(self.cache_path):
            df.to_csv(self.cache_path, mode='a', header=False, index=False)
        else:
            df.to_csv(self.cache_path, index=False)

    def get_signal(self, features, legacy_weights=None):
        """
        Ensemble fallback: Use Meta-model if available, otherwise use legacy weights.
        """
        if self.model is not None:
            try:
                return self.predict(features)
            except Exception as e:
                print(f"Meta-model prediction error: {e}")
        
        # Legacy Fallback Logic
        if legacy_weights:
            total = 0
            for feat, weight in legacy_weights.items():
                total += features.get(feat, 0) * weight
            return total
            
        return 0.0

if __name__ == "__main__":
    # Test dummy training
    meta = MetaModel()
    dummy_data = pd.DataFrame({
        'ai_price': [0.5, -0.2, 0.8, -0.5],
        'sentiment': [0.1, -0.1, 0.4, -0.3],
        'inst_flow': [0.6, -0.5, 0.2, -0.8],
        'pcr': [0.2, -0.3, 0.1, -0.5],
        'sector': [0.1, 0.0, 0.5, -0.2],
        'regime': [1, 0, 1, 0] # 1 for Bull, 0 for Bear
    })
    dummy_outcomes = [1, 0, 1, 0]
    meta.train(dummy_data, dummy_outcomes)
    print("Test Predict:", meta.predict({'ai_price': 0.7, 'sentiment': 0.3, 'inst_flow': 0.5, 'pcr': 0.1, 'sector': 0.4, 'regime': 1}))
