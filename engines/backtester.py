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
import macro

class StrategyBacktester:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.ai_predictor = predictor.StockPredictor()

    def run_backtest(self, symbol, lookback_days=120):
        """
        Runs a historical simulation of the model's signals.
        """
        try:
            # Clean symbol
            symbol_ns = symbol if symbol.endswith(".NS") else symbol + ".NS"
            
            # 1. Fetch historical data (long period)
            df = data_engine.get_stock_data(symbol_ns, period="2y")
            if df is None or len(df) < 150:
                return None
            
            # 2. Add all technical features
            df = technical.add_technical_indicators(df)
            
            # 3. Add Macro Features (using current as proxy for simulation)
            macros = macro.get_macro_data()
            for m_name, m_val in macros.items():
                m_col = f'macro_{m_name}'
                if isinstance(m_val, dict) and 'latest' in m_val:
                    df[m_col] = m_val['latest']
                elif m_name == 'INSTITUTIONAL':
                    # Special case for institutional data - use a neutral proxy
                    df[m_col] = 0.0
                else:
                    df[m_col] = 0.0
            
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            
            # 4. Load Models
            lgbm_path = os.path.join(self.model_dir, f'lgbm_{symbol_ns}.joblib')
            xgb_path = os.path.join(self.model_dir, f'xgb_{symbol_ns}.json')
            
            if not os.path.exists(lgbm_path):
                return None
                
            model_lgbm = joblib.load(lgbm_path)
            from xgboost import XGBClassifier
            model_xgb = XGBClassifier()
            model_xgb.load_model(xgb_path)

            # 5. Simulation Loop
            results = []
            test_period = df.tail(lookback_days).copy()
            feature_names = model_lgbm.feature_name_

            for i in range(len(test_period) - 5):
                current_day = test_period.iloc[i]
                future_5d_price = test_period.iloc[i+5]['Close']
                
                # Extract features in correct order
                try:
                    features_row = current_day[feature_names].values.reshape(1, -1)
                except Exception as e:
                    # If columns missing, attempt to fill with zeros
                    missing = [c for c in feature_names if c not in current_day]
                    for m in missing: test_period[m] = 0.0
                    current_day = test_period.iloc[i]
                    features_row = current_day[feature_names].values.reshape(1, -1)
                
                # Model outputs
                pred_price = model_lgbm.predict(features_row)[0]
                prob_up = model_xgb.predict_proba(features_row)[0][1]
                
                # Signal blending
                ai_change = ((pred_price - current_day['Close']) / current_day['Close']) * 100
                ai_score = np.clip(ai_change / 5.0, -1, 1)
                total_score = (ai_score * 0.6) + ((prob_up - 0.5) * 2 * 0.4)
                
                verdict = "HOLD"
                if total_score > 0.3: verdict = "BUY"
                elif total_score < -0.3: verdict = "SELL"
                
                # Accuracy Check
                is_correct = False
                if verdict == "BUY" and future_5d_price > current_day['Close']: is_correct = True
                if verdict == "SELL" and future_5d_price < current_day['Close']: is_correct = True
                
                if verdict != "HOLD":
                    results.append({
                        'date': test_period.index[i].strftime('%Y-%m-%d'),
                        'price': round(float(current_day['Close']), 2),
                        'verdict': verdict,
                        'return_pct': round(((future_5d_price - current_day['Close']) / current_day['Close']) * 100 if verdict == "BUY" else ((current_day['Close'] - future_5d_price) / current_day['Close']) * 100, 2),
                        'is_correct': is_correct
                    })

            if not results:
                return {'win_rate': 0, 'total_trades': 0, 'cumulative_return': 0, 'signals': []}

            # 6. Calculate Stats
            total_trades = len(results)
            wins = len([r for r in results if r['is_correct']])
            win_rate = (wins / total_trades) * 100
            total_return = sum([r['return_pct'] for r in results])
            
            return {
                'win_rate': round(win_rate, 1),
                'total_trades': total_trades,
                'cumulative_return': round(total_return, 2),
                'signals': results[-15:]
            }
        except Exception as e:
            print(f"Backtest error for {symbol}: {e}")
            return None

if __name__ == "__main__":
    bt = StrategyBacktester()
    print("Testing backtest for TCS...")
    stats = bt.run_backtest("TCS")
    print(stats)
