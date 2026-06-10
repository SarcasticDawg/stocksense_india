# StockSense India 🇮🇳 📈

**StockSense India** is a professional-grade, automated financial intelligence dashboard for NSE (National Stock Exchange) stocks. It utilizes a **Market-Adaptive Ensemble** of Machine Learning models (LightGBM & XGBoost), NLP Sentiment Analysis (VADER), and Institutional Flow tracking to provide robust, real-time market insights.

This project was developed as an open-source initiative to provide the community with institutional-grade trading tools.

## 🚀 Features

*   **Automated Intelligence Pipeline:** Runs nightly via GitHub Actions to fetch data, train ML models, and save predictions to MongoDB.
*   **Dual-Mode Architecture:**
    *   **NIGHT MODE:** Lightning-fast loads using pre-computed data from MongoDB.
    *   **LIVE MODE:** Real-time data scraping and on-the-fly ML predictions during market hours.
*   **Market-Adaptive Meta-Model:** Automatically adjusts the weighting of AI, Sentiment, and Institutional signals based on recent historical accuracy.
*   **Forward-Looking Sentiment Engine:** Distinguishes between "reporting" (past facts) and "signal" (future outlooks) across news, corporate filings, and social media.
*   **Institutional Tracking:** Monitors FII/DII net flows and stock-specific Put/Call Ratios (PCR).
*   **Professional Charting:** Toggle between Area and Candlestick views with synchronized RSI panels.

## 🧠 System Architecture

1.  **ML Engine:** LightGBM (Next-day price) & XGBoost (5-day directional probability).
2.  **Meta Engine:** Logistic Regression blender that dynamically weights signals.
3.  **Data Sources:**
    *   Yahoo Finance (`yfinance`)
    *   Sensibull (FII/DII & F&O)
    *   NiftyInvest (PCR & Delivery %)
    *   Screener.in (Corporate Announcements)
    *   RSS Feeds (Economic Times, Moneycontrol, Business Standard)
4.  **Database:** MongoDB Atlas (NoSQL Document Store).
5.  **Deployment:** Render (Web Server) & GitHub Actions (Serverless Task Scheduler).

## 🛠️ Setup & Installation (For Developers)

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/stocksense_india.git
cd stocksense_india
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory and add your MongoDB connection string:
```bash
MONGODB_URI="mongodb+srv://<user>:<password>@cluster0...mongodb.net/?retryWrites=true&w=majority"
```

### 4. Run Locally
```bash
python app.py
```
The app will be available at `http://localhost:8080`.

## ⚙️ Automated Pipeline Setup
This repository contains a GitHub Actions workflow (`.github/workflows/stocksense_pipeline.yml`) that automates the data processing.

1.  Add `MONGODB_URI` to your GitHub Repository Secrets.
2.  The scheduler will automatically run:
    *   **19:00 IST:** `batch_runner.py` (Full data fetch & ML retraining)
    *   **08:30 IST:** `premarket_runner.py` (Morning sentiment refresh)

## 🤝 Contributing
Contributions are welcome! If you'd like to improve the sentiment lexicon, add new data sources, or optimize the ML models, please fork the repository and submit a Pull Request.

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
