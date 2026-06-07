import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# Path setup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import data_engine
import technical
import predictor
import meta_model

class StrategyBacktester:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.ai_predictor = predictor.StockPredictor()
        self.meta = meta_model.MetaModel()

    def run_backtest(self, symbol, lookback_days=120):
        """
        Runs a historical simulation of the model's signals.
        """
        try:
            # 1. Fetch historical data (long period)
            df = data_engine.get_stock_data(symbol, period="2y")
            if df is None or len(df) < 150:
                return None
            
            # 2. Add all technical features
            df = technical.add_technical_indicators(df)
            df = df.dropna()
            
            # 3. Load Models
            lgbm_path = os.path.join(self.model_dir, f'lgbm_{symbol}.joblib')
            xgb_path = os.path.join(self.model_dir, f'xgb_{symbol}.json')
            
            if not os.path.exists(lgbm_path):
                return None
                
            model_lgbm = joblib.load(lgbm_path)
            # XGBoost loading is slightly different as it's a class
            from xgboost import XGBClassifier
            model_xgb = XGBClassifier()
            model_xgb.load_model(xgb_path)

            # 4. Simulation Loop (Last N days)
            # We skip the very last few days because we don't know the outcome yet
            results = []
            test_period = df.tail(lookback_days).copy()
            
            # Feature columns needed by predictor
            feature_cols = [
                'Close', 'RSI', 'MACD', 'BB_Position', 'MA50', 'MA200', 'Vol_Change_Pct',
                'Lag_Close_1', 'Lag_Close_2', 'Lag_Close_3', 'Lag_Close_5', 'Lag_Close_10',
                'Roll_Mean_7', 'Roll_Mean_14', 'Roll_Std_30',
                'Momentum_5', 'Momentum_10', 'RVOL'
            ]
            
            # Add placeholders for macro features to prevent crash
            # In a real backtest we'd use historical macro, here we assume current
            available_cols = [c for c in feature_cols if c in df.columns]
            
            # Iterate through days
            for i in range(len(test_period) - 5):
                current_day = test_period.iloc[i]
                future_5d_price = test_period.iloc[i+5]['Close']
                
                features = current_day[available_cols].values.reshape(1, -1)
                
                # Model outputs
                pred_price = model_lgbm.predict(features)[0]
                prob_up = model_xgb.predict_proba(features)[0][1]
                
                # Signal blending (Simplified version of app logic)
                ai_change = ((pred_price - current_day['Close']) / current_day['Close']) * 100
                ai_score = np.clip(ai_change / 5.0, -1, 1)
                
                # Blended Score
                total_score = (ai_score * 0.6) + ((prob_up - 0.5) * 2 * 0.4)
                
                verdict = "HOLD"
                if total_score > 0.3: verdict = "BUY"
                elif total_score < -0.3: verdict = "SELL"
                
                # Accuracy Check (Did it go up/down in 5 days?)
                is_correct = False
                if verdict == "BUY" and future_5d_price > current_day['Close']: is_correct = True
                if verdict == "SELL" and future_5d_price < current_day['Close']: is_correct = True
                
                if verdict != "HOLD":
                    results.append({
                        'date': test_period.index[i].strftime('%Y-%m-%d'),
                        'price': current_day['Close'],
                        'verdict': verdict,
                        'outcome_price': future_5d_price,
                        'return_pct': ((future_5d_price - current_day['Close']) / current_day['Close']) * 100 if verdict == "BUY" else ((current_day['Close'] - future_5d_price) / current_day['Close']) * 100,
                        'is_correct': is_correct
                    })

            if not results:
                return None

            # 5. Calculate Aggregated Stats
            total_trades = len(results)
            wins = len([r for r in results if r['is_correct']])
            win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
            
            total_return = sum([r['return_pct'] for r in results])
            
            return {
                'win_rate': round(win_rate, 1),
                'total_trades': total_trades,
                'avg_return_per_trade': round(total_return / total_trades, 2) if total_trades > 0 else 0,
                'cumulative_return': round(total_return, 2),
                'signals': results[-15:] # Return last 15 markers for chart
            }
        except Exception as e:
            print(f"Backtest error for {symbol}: {e}")
            return None

if __name__ == "__main__":
    bt = StrategyBacktester()
    print("Testing backtest for TCS...")
    stats = bt.run_backtest("TCS.NS")
    print(stats)
