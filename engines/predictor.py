import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import joblib
import gc

# Path setup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import data_engine
import technical
import macro

class StockPredictor:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.scaler = MinMaxScaler()
        self.model_cache = {}

    def prepare_feature_matrix(self, symbol, period="2y"):
        """
        Aggregates technical and macro features into a single dataframe.
        """
        df = data_engine.get_stock_data(symbol, period=period)
        if df is None or df.empty: return None, None
        df = technical.add_technical_indicators(df)
        
        macros = macro.get_macro_data()
        for m_name, m_val in macros.items():
            if isinstance(m_val, dict) and 'latest' in m_val:
                df[f'macro_{m_name}'] = m_val['latest']
            else:
                df[f'macro_{m_name}'] = 0
        
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        feature_cols = [
            'Close', 'RSI', 'MACD', 'BB_Position', 'MA50', 'MA200', 'Vol_Change_Pct',
            'Lag_Close_1', 'Lag_Close_2', 'Lag_Close_3', 'Lag_Close_5', 'Lag_Close_10',
            'Roll_Mean_7', 'Roll_Mean_14', 'Roll_Std_30',
            'Momentum_5', 'Momentum_10', 'RVOL'
        ]
        macro_cols = [f'macro_{m}' for m in macros.keys() if f'macro_{m}' in df.columns]
        feature_cols += macro_cols
        
        available_cols = [c for c in feature_cols if c in df.columns]
        return df[available_cols], df['Close']

    def train(self, symbol):
        print(f"--- [AI Training] LightGBM & XGBoost for {symbol} ---")
        try:
            features, target = self.prepare_feature_matrix(symbol)
            if features is None or len(features) < 100:
                print(f"Insufficient data for {symbol}.")
                return
            
            # --- LightGBM Regressor (Next Day Price) ---
            # Using price change % as target for better stability than raw price
            y_lgb = target.shift(-1).iloc[:-1]
            X_lgb = features.iloc[:-1]
            
            lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, num_leaves=31, verbose=-1)
            lgb_model.fit(X_lgb, y_lgb)
            joblib.dump(lgb_model, os.path.join(self.model_dir, f'lgbm_{symbol}.joblib'))
            
            # --- XGBoost Classifier (5-day Direction) ---
            y_xgb = (target.shift(-5) > target).astype(int).iloc[:-5]
            X_xgb = features.iloc[:-5]
            
            xgb_model = XGBClassifier(n_estimators=50, learning_rate=0.1, max_depth=3)
            xgb_model.fit(X_xgb, y_xgb)
            xgb_model.save_model(os.path.join(self.model_dir, f'xgb_{symbol}.json'))
            
            gc.collect()
            print(f"--- [AI Training] COMPLETED for {symbol} ---")
        except Exception as e:
            print(f"AI Training Error for {symbol}: {e}")

    def get_prediction(self, symbol):
        lgbm_path = os.path.join(self.model_dir, f'lgbm_{symbol}.joblib')
        xgb_path = os.path.join(self.model_dir, f'xgb_{symbol}.json')
        
        features, target = self.prepare_feature_matrix(symbol, period="1y")
        if features is None or target is None: return None
            
        current_price = target.iloc[-1]
        
        if not os.path.exists(lgbm_path) or not os.path.exists(xgb_path):
            return {
                'current_price': float(round(current_price, 2)),
                'predicted_price': "Training...",
                'price_change_pct': 0.0,
                'trend_prob_up': 0.5,
                'confidence': 0.0
            }
            
        try:
            # LightGBM Prediction
            cache_key_lgbm = f"lgbm_{symbol}"
            if cache_key_lgbm in self.model_cache:
                model_lgbm = self.model_cache[cache_key_lgbm]
            else:
                model_lgbm = joblib.load(lgbm_path)
                self.model_cache[cache_key_lgbm] = model_lgbm
            
            last_features = features.iloc[-1:]
            pred_price = model_lgbm.predict(last_features)[0]
            
            # XGBoost Prediction
            cache_key_xgb = f"xgb_{symbol}"
            if cache_key_xgb in self.model_cache:
                model_xgb = self.model_cache[cache_key_xgb]
            else:
                model_xgb = XGBClassifier()
                model_xgb.load_model(xgb_path)
                self.model_cache[cache_key_xgb] = model_xgb
                
            prob_up = model_xgb.predict_proba(last_features.values)[0][1]
            
            # Sanity Check
            max_move = current_price * 0.05
            if pred_price > (current_price + max_move): pred_price = current_price + (current_price * 0.01)
            elif pred_price < (current_price - max_move): pred_price = current_price - (current_price * 0.01)
            
            price_diff_pct = ((pred_price - current_price) / current_price) * 100
            
            return {
                'current_price': float(round(current_price, 2)),
                'predicted_price': float(round(pred_price, 2)),
                'price_change_pct': float(round(price_diff_pct, 2)),
                'trend_prob_up': float(round(prob_up, 2)),
                'confidence': float(round(0.5 + (abs(prob_up - 0.5) * 0.8), 2))
            }
        except Exception as e:
            print(f"Prediction Error for {symbol}: {e}")
            return None

if __name__ == "__main__":
    predictor = StockPredictor()
    result = predictor.get_prediction("TCS")
    print(result)
