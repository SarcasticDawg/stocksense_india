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
    - **LSTM**: Predicts tomorrow's closing price based on the last 60 days of data, using advanced features like lags (1-10d), rolling statistics, and momentum.
    - **XGBoost**: Predicts the probability of a directional move (Up/Down) over the next 5 days, utilizing technical, macro, and volume-based features (RVOL).
- **`engines/sentiment.py`**: Uses **VADER** (Valence Aware Dictionary and sEntiment Reasoner) with a custom **Finance & Reddit Lexicon**.
- **`engines/scraper.py`**: Scrapes latest news from Zerodha Pulse/RSS feeds and social sentiment from Reddit (IndiaInvestments, stocks, etc.).
- **`engines/macro.py`**: Tracks global signals like India VIX, Brent Crude, USD/INR, and S&P 500.
- **`engines/sector.py`**: Ranks the searched stock against its top 5 industry peers.

---

## 2. Signal Logic (Weighted Model)
The final recommendation is calculated using a weighted score:
1. **Price Prediction (30%)**: Normalized target price change.
2. **News Sentiment (20%)**: VADER score of recent headlines.
3. **Reddit Sentiment (15%)**: VADER score of discussions, weighted by upvotes (log scale).
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
│   ├── finance_lexicon.py  # Custom sentiment rules
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
- **NLP**: `vaderSentiment` (Custom Lexicon)
- **Scraping**: `beautifulsoup4`, `praw` (Reddit API), `feedparser`

---

## 5. Future Roadmap
- [x] Integrate VADER with custom Finance Lexicon.
- [x] Implement Reddit upvote weighting for sentiment.
- [ ] Add Chart.js for interactive historical price graphs.
- [ ] Implement a daily auto-retraining schedule for ML models.
- [ ] Add real-time price alerts via Email/Telegram.

---
*Generated on June 6, 2026*
