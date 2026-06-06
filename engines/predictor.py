import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
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
        # Cache for loaded models to save memory/time
        self.model_cache = {}

    def prepare_feature_matrix(self, symbol):
        """
        Aggregates technical and macro features into a single dataframe.
        """
        # 1. Fetch stock data (2 years for enough MA200 data)
        df = data_engine.get_stock_data(symbol, period="2y")
        if df is None or df.empty: return None, None
        df = technical.add_technical_indicators(df)
        
        # 2. Fetch macro data
        macros = macro.get_macro_data()
        
        # 3. Add macro features
        for m_name, m_val in macros.items():
            if isinstance(m_val, dict) and 'latest' in m_val:
                df[f'macro_{m_name}'] = m_val['latest']
            else:
                df[f'macro_{m_name}'] = 0 # Fallback
        
        # 4. Clean data
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        if len(df) < 60:
            return None, None
        
        # Features required
        feature_cols = [
            'Close', 'RSI', 'MACD', 'BB_Position', 'MA50', 'MA200', 'Vol_Change_Pct',
            'Lag_Close_1', 'Lag_Close_2', 'Lag_Close_3', 'Lag_Close_5', 'Lag_Close_10',
            'Roll_Mean_7', 'Roll_Mean_14', 'Roll_Std_30',
            'Momentum_5', 'Momentum_10', 'RVOL'
        ]
        # Only add valid macro columns
        macro_cols = [f'macro_{m}' for m in macros.keys() if f'macro_{m}' in df.columns]
        feature_cols += macro_cols
        
        # Ensure all columns exist
        available_cols = [c for c in feature_cols if c in df.columns]
        
        return df[available_cols], df['Close']

    def create_sequences(self, data, target, window=60):
        X, y = [], []
        if len(data) <= window:
            return np.array([]), np.array([])
        for i in range(window, len(data)):
            X.append(data[i-window:i])
            y.append(target[i])
        return np.array(X), np.array(y)

    def build_lstm(self, input_shape):
        # Reduced units for low memory
        model = Sequential([
            tf.keras.Input(shape=input_shape),
            LSTM(32, return_sequences=False),
            Dropout(0.1),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, symbol):
        print(f"--- [AI Training] Starting models for {symbol} ---")
        try:
            features, target = self.prepare_feature_matrix(symbol)
            if features is None or len(features) < 100:
                print(f"Insufficient data for training {symbol}.")
                return
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            # Scale target for LSTM
            target_scaler = MinMaxScaler()
            scaled_target = target_scaler.fit_transform(target.values.reshape(-1, 1))
            
            # --- LSTM Training ---
            X_lstm, y_lstm = self.create_sequences(scaled_features, scaled_target.flatten())
            if len(X_lstm) == 0: return
            
            # Use all data for prototype training to keep it simple
            lstm_model = self.build_lstm((X_lstm.shape[1], X_lstm.shape[2]))
            lstm_model.fit(X_lstm, y_lstm, epochs=15, batch_size=32, verbose=0)
            lstm_model.save(os.path.join(self.model_dir, f'lstm_{symbol}.keras'))
            
            # --- XGBoost Training ---
            y_xgb = (target.shift(-5) > target).astype(int).iloc[:-5]
            X_xgb = features.iloc[:-5]
            
            xgb_model = XGBClassifier(n_estimators=50, learning_rate=0.1, max_depth=3)
            xgb_model.fit(X_xgb, y_xgb)
            xgb_model.save_model(os.path.join(self.model_dir, f'xgb_{symbol}.json'))
            
            # Memory Cleanup
            tf.keras.backend.clear_session()
            gc.collect()
            print(f"--- [AI Training] COMPLETED for {symbol} ---")
        except Exception as e:
            print(f"AI Training Error for {symbol}: {e}")

    def get_prediction(self, symbol):
        """
        Loads models and returns prediction + confidence.
        """
        lstm_path = os.path.join(self.model_dir, f'lstm_{symbol}.keras')
        xgb_path = os.path.join(self.model_dir, f'xgb_{symbol}.json')
        
        # Load features once
        features, target = self.prepare_feature_matrix(symbol)
        if features is None or target is None:
            return None
            
        current_price = target.iloc[-1]
        
        # If models missing, return basic data and let app handle async training
        if not os.path.exists(lstm_path) or not os.path.exists(xgb_path):
            return {
                'current_price': float(round(current_price, 2)),
                'predicted_price': "Training...",
                'price_change_pct': 0.0,
                'trend_prob_up': 0.5,
                'confidence': 0.0
            }
            
        try:
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            # LSTM Prediction (Next Day)
            last_sequence = scaled_features[-60:].reshape(1, 60, scaled_features.shape[1])
            
            # Use cached model if available
            cache_key = f"lstm_{symbol}"
            if cache_key in self.model_cache:
                model_lstm = self.model_cache[cache_key]
            else:
                model_lstm = load_model(lstm_path)
                self.model_cache[cache_key] = model_lstm
                
            pred_scaled = model_lstm.predict(last_sequence, verbose=0)[0][0]
            
            target_scaler = MinMaxScaler()
            target_scaler.fit(target.values.reshape(-1, 1))
            pred_price = target_scaler.inverse_transform([[pred_scaled]])[0][0]
            
            # XGBoost Prediction (5-day trend)
            xgb_key = f"xgb_{symbol}"
            if xgb_key in self.model_cache:
                model_xgb = self.model_cache[xgb_key]
            else:
                model_xgb = XGBClassifier()
                model_xgb.load_model(xgb_path)
                self.model_cache[xgb_key] = model_xgb
                
            last_features = features.iloc[-1:].values
            prob_up = model_xgb.predict_proba(last_features)[0][1]
            
            # --- SANITY CHECK ---
            max_move = current_price * 0.05 # 5% limit for safety
            if pred_price > (current_price + max_move):
                pred_price = current_price + (current_price * 0.01)
            elif pred_price < (current_price - max_move):
                pred_price = current_price - (current_price * 0.01)
            
            price_diff_pct = ((pred_price - current_price) / current_price) * 100
            
            confidence = 0.5
            if (price_diff_pct > 0 and prob_up > 0.5) or (price_diff_pct < 0 and prob_up < 0.5):
                confidence = 0.5 + (abs(prob_up - 0.5) * 0.8)
                
            return {
                'current_price': float(round(current_price, 2)),
                'predicted_price': float(round(pred_price, 2)),
                'price_change_pct': float(round(price_diff_pct, 2)),
                'trend_prob_up': float(round(prob_up, 2)),
                'confidence': float(round(confidence, 2))
            }
        except Exception as e:
            print(f"Prediction Error for {symbol}: {e}")
            return None

if __name__ == "__main__":
    predictor = StockPredictor()
    result = predictor.get_prediction("TCS")
    print(result)
