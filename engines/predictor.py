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

    def prepare_feature_matrix(self, symbol):
        """
        Aggregates technical and macro features into a single dataframe.
        """
        # 1. Fetch stock data (2 years for enough MA200 data)
        df = data_engine.get_stock_data(symbol, period="2y")
        if df is None: return None
        df = technical.add_technical_indicators(df)
        
        # 2. Fetch macro data
        macros = macro.get_macro_data()
        
        # 3. Add macro features
        for m_name, m_val in macros.items():
            df[f'macro_{m_name}'] = m_val['latest']
        
        # 4. Clean data
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        if len(df) < 100:
            print(f"Warning: Only {len(df)} rows after cleaning. Trying 5y data...")
            df = data_engine.get_stock_data(symbol, period="5y")
            df = technical.add_technical_indicators(df)
            for m_name, m_val in macros.items():
                df[f'macro_{m_name}'] = m_val['latest']
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
        
        # Features required
        feature_cols = [
            'Close', 'RSI', 'MACD', 'BB_Position', 'MA50', 'MA200', 'Vol_Change_Pct',
            'Lag_Close_1', 'Lag_Close_2', 'Lag_Close_3', 'Lag_Close_5', 'Lag_Close_10',
            'Roll_Mean_7', 'Roll_Mean_14', 'Roll_Std_30',
            'Momentum_5', 'Momentum_10', 'RVOL'
        ]
        feature_cols += [f'macro_{m}' for m in macros.keys()]
        
        # Ensure all columns exist
        available_cols = [c for c in feature_cols if c in df.columns]
        
        return df[available_cols], df['Close']

    def create_sequences(self, data, target, window=60):
        X, y = [], []
        for i in range(window, len(data)):
            X.append(data[i-window:i])
            y.append(target[i])
        return np.array(X), np.array(y)

    def build_lstm(self, input_shape):
        model = Sequential([
            tf.keras.Input(shape=input_shape),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, symbol):
        print(f"Training models for {symbol}...")
        features, target = self.prepare_feature_matrix(symbol)
        if features is None or len(features) < 100:
            print("Insufficient data for training.")
            return
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Scale target for LSTM
        self.target_scaler = MinMaxScaler()
        scaled_target = self.target_scaler.fit_transform(target.values.reshape(-1, 1))
        
        # --- LSTM Training ---
        X_lstm, y_lstm = self.create_sequences(scaled_features, scaled_target.flatten())
        split = int(0.8 * len(X_lstm))
        X_train, X_test = X_lstm[:split], X_lstm[split:]
        y_train, y_test = y_lstm[:split], y_lstm[split:]
        
        lstm_model = self.build_lstm((X_train.shape[1], X_train.shape[2]))
        # Added EarlyStopping for better convergence
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
        lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, callbacks=[callback])
        lstm_model.save(os.path.join(self.model_dir, f'lstm_{symbol}.keras'))
        
        # --- XGBoost Training ---
        # Binary target: Up/Down in 5 days
        y_xgb = (target.shift(-5) > target).astype(int).iloc[:-5]
        X_xgb = features.iloc[:-5]
        
        xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1)
        xgb_model.fit(X_xgb, y_xgb)
        xgb_model.save_model(os.path.join(self.model_dir, f'xgb_{symbol}.json'))
        
        print(f"Models saved for {symbol}")

    def get_prediction(self, symbol):
        """
        Loads models and returns prediction + confidence.
        """
        lstm_path = os.path.join(self.model_dir, f'lstm_{symbol}.keras')
        xgb_path = os.path.join(self.model_dir, f'xgb_{symbol}.json')
        
        if not os.path.exists(lstm_path) or not os.path.exists(xgb_path):
            self.train(symbol)
            
        # Load and Predict
        features, target = self.prepare_feature_matrix(symbol)
        scaled_features = self.scaler.fit_transform(features)
        
        # Scaling target again for inverse transform (ideally we save the scaler)
        self.target_scaler = MinMaxScaler()
        self.target_scaler.fit(target.values.reshape(-1, 1))
        
        # LSTM Prediction (Next Day)
        last_sequence = scaled_features[-60:].reshape(1, 60, scaled_features.shape[1])
        model_lstm = load_model(lstm_path)
        pred_scaled = model_lstm.predict(last_sequence)[0][0]
        pred_price = self.target_scaler.inverse_transform([[pred_scaled]])[0][0]
        
        # --- SANITY CHECK ---
        current_price = target.iloc[-1]
        
        # If prediction is physically impossible (>15% move in a day), 
        # clip it to a reasonable range or use a conservative fallback.
        max_move = current_price * 0.10 # 10% limit
        if pred_price > (current_price + max_move):
            pred_price = current_price + (current_price * 0.015) # Cap at +1.5% if AI goes wild
        elif pred_price < (current_price - max_move):
            pred_price = current_price - (current_price * 0.015) # Cap at -1.5%
        
        # XGBoost Prediction (5-day trend)
        model_xgb = XGBClassifier()
        model_xgb.load_model(xgb_path)
        last_features = features.iloc[-1:].values
        prob_up = model_xgb.predict_proba(last_features)[0][1]
        
        # Confidence Score logic
        current_price = target.iloc[-1]
        price_diff_pct = ((pred_price - current_price) / current_price) * 100
        
        confidence = 0.5 # Baseline
        if (price_diff_pct > 0 and prob_up > 0.5) or (price_diff_pct < 0 and prob_up < 0.5):
            confidence = 0.5 + abs(prob_up - 0.5)
            
        return {
            'current_price': float(round(current_price, 2)),
            'predicted_price': float(round(pred_price, 2)),
            'price_change_pct': float(round(price_diff_pct, 2)),
            'trend_prob_up': float(round(prob_up, 2)),
            'confidence': float(round(confidence, 2))
        }

if __name__ == "__main__":
    predictor = StockPredictor()
    # Training might take a bit, so we just test prediction (which triggers training if missing)
    result = predictor.get_prediction("TCS")
    print("\nPrediction Result:")
    print(result)
