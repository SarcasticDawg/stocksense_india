# StockSense India - Project Documentation

## Project Overview
StockSense India is a full-stack financial analysis dashboard built with Flask. It provides price predictions, sentiment analysis, and sector comparisons for NSE stocks using Machine Learning and NLP.

---

## 1. Technical Architecture
The project is built using a modular "Engine" architecture:

- **`app.py`**: The main entry point. Handles routing and combines scores from all engines to generate a final Buy/Hold/Sell signal.
- **`engines/data_engine.py`**: The data backbone. Fetches historical OHLCV data via `yfinance` and manages the NSE stock master list.
- **`engines/technical.py`**: Computes technical indicators (RSI, MACD, Bollinger Bands, Moving Averages) manually using `pandas`.
- **`engines/predictor.py`**: The Machine Learning core. 
    - **LSTM**: Predicts tomorrow's closing price based on the last 60 days of data.
    - **XGBoost**: Predicts the probability of a directional move (Up/Down) over the next 5 days.
- **`engines/sentiment.py`**: Uses **FinBERT** (a specialized financial NLP model) to score sentiment from news and social media.
- **`engines/scraper.py`**: Scrapes latest news from Zerodha Pulse/NDTV Profit and social sentiment from Reddit (r/IndiaInvestments).
- **`engines/macro.py`**: Tracks global signals like India VIX, Brent Crude, USD/INR, and S&P 500.
- **`engines/sector.py`**: Ranks the searched stock against its top 5 industry peers.

---

## 2. Signal Logic (Weighted Model)
The final recommendation is calculated using a weighted score:
1. **Price Prediction (30%)**: Normalized target price change.
2. **News Sentiment (20%)**: FinBERT score of recent headlines.
3. **Reddit Sentiment (15%)**: FinBERT score of retail discussions.
4. **Macro Signals (20%)**: Market volatility and global trends.
5. **Sector Position (15%)**: Performance relative to industry peers.

---

## 3. Directory Structure
```text
stocksense/
├── app.py                  # Flask Web App
├── config.py               # API Keys & Config
├── data/
│   └── nse_stocks.csv      # Stock Master List
├── engines/                # Modular Logic
│   ├── data_engine.py
│   ├── predictor.py
│   ├── sentiment.py
│   └── ...
├── models/                 # Saved ML Models (.keras & .json)
├── static/                 # CSS & JS
└── templates/              # HTML Dashboards
```

---

## 4. Key Libraries Used
- **Backend**: `Flask`
- **Data**: `pandas`, `numpy`, `yfinance`
- **ML**: `tensorflow` (Keras 3), `xgboost`, `scikit-learn`
- **NLP**: `transformers` (HuggingFace), `torch`
- **Scraping**: `beautifulsoup4`, `praw` (Reddit API)

---

## 5. Future Roadmap
- [ ] Add Chart.js for interactive historical price graphs.
- [ ] Implement a daily auto-retraining schedule for ML models.
- [ ] Add real-time price alerts via Email/Telegram.
- [ ] Implement user accounts to track favorite stocks.

---
*Generated on June 4, 2026*
