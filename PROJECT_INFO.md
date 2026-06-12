# StockSense India - System Design Specification

## Project Overview
StockSense India is a professional-grade financial intelligence dashboard built with Flask. It provides predictive analytics, multi-source sentiment analysis, and institutional flow tracking for NSE stocks using a Market-Adaptive Ensemble architecture. 

The system operates in a **100% database-driven mode**, exclusively serving pre-calculated intelligence from JSON files to ensure maximum stability and lightning-fast load times on resource-constrained hosting environments.

---

## 1. Technical Architecture (Market-Adaptive Ensemble)
The system is designed as a **Self-Optimizing Ensemble** that transitions from a rule-based baseline to a machine-learned quantitative model.

### A. Module & Sub-Module Directory
| Primary Module | Sub-Modules / Components | Core Function |
| :--- | :--- | :--- |
| **Orchestrator** | `app.py` | Routing, UI Rendering, JSON Data Serving. |
| **ML Engine** | `predictor.py`, `batch_train.py` | LightGBM Regressor, XGBoost Classifier, Batch Retrainer. |
| **Data Generation**| `batch_runner.py`, `updates_3hr.py` | Data Fetching, Signal Blending, JSON File Generation. |
| **Meta Engine** | `meta_model.py` | Logistic Regression (Learned Weight Optimization). |
| **Institutional** | `macro.py`, `fno.py`, `conviction.py` | FII/DII Flows, F&O Positioning, PCR, Delivery %. |
| **Intelligence** | `corporate.py`, `scraper.py`, `sentiment.py` | NSE/BSE Filings, Multi-Source RSS, VADER NLP. |
| **Analytics** | `market_context.py`, `backtester.py` | Regime Detection, Breadth, Strategy Backtester. |
| **Context** | `data_engine.py`, `technical.py`, `sector.py` | yfinance API, Technical Indicators, Peer Ranking. |

### B. Data Extraction Mapping
| Metric / Feature | Data Source | Extraction Method |
| :--- | :--- | :--- |
| **Historical Price/Vol** | **Yahoo Finance** | `yfinance` Library (Direct API). |
| **FII / DII Net Cash** | **Sensibull** | `oxide.sensibull.com` JSON Endpoint. |
| **FII F&O Positioning** | **Sensibull** | Public F&O Dashboard (JSON). |
| **Stock-Specific PCR** | **NiftyInvest** | `niftyinvest.com` Scraper (Regex). |
| **NSE Delivery %** | **NiftyInvest** | `niftyinvest.com` Scraper (Regex). |
| **Corporate Filings** | **Screener.in** | AJAX Announcement endpoint (BeautifulSoup). |
| **Professional News** | **ET, Moneycontrol, BS** | Official RSS Feeds (`feedparser`). |
| **Social Sentiment** | **Reddit / Screener** | Reddit JSON API + Screener Board Scraper. |
| **Market Regime** | **Nifty 50 Index** | Calculated from 50/200 DMA + Volatility. |

### C. Modular "Engine" Core
- **`app.py`**: The web server. Operates exclusively in a read-only capacity, fetching and rendering pre-computed data from local JSON files (`nightly_dump.json`, `sentiment_update.json`, `market_indices.json`) to guarantee zero-delay loading and prevent scraping blocks.
- **`batch_train.py`**: The offline ML training script. Retrains LightGBM, XGBoost, and the Meta-Model (using historical JSON data) and persists the updated models.
- **`batch_runner.py`**: The offline data generation script. Loads trained models, fetches live data, calculates final signals, and generates the definitive JSON data files.
- **`engines/predictor.py`**: The Machine Learning core. 
    - **LightGBM (60% of module)**: Regressor for next-day price forecasting. Handles tabular features like lags (1-10d) and rolling stats.
    - **XGBoost (40% of module)**: Classifier for 5-day directional probability.
    - *Self-Optimization*: This 60/40 internal split is the quantitative baseline; the Meta-model automatically rebalances these weights if backtesting shows one model outperforming the other.
- **`engines/meta_model.py`**: The "Brain" of the ensemble. Uses **Logistic Regression** to learn optimal weights for all signals based on 30-day historical outcomes.
- **`engines/market_context.py`**: Regime detection module (**BULL/BEAR/SIDEWAYS**) and **Market Breadth** tracker (Nifty 50 stocks above 50-DMA).
- **`engines/sentiment.py`**: Advanced NLP engine (v4.0) using **VADER** with a multi-layered weighting architecture. Features a **Forward-Looking Filter** to prioritize future outlooks (Signal) 5x higher than historical facts (Reporting).
- **`engines/corporate.py`**: Corporate intelligence module that scrapes official NSE/BSE filings (Dividends, Results, Buybacks) from Screener.in with direct source verification links.
- **`engines/scraper.py`**: Multi-source aggregator fetching intelligence from 4+ professional RSS feeds (ET, Moneycontrol, BS, Google) and **Screener.in discussion boards**.
- **`engines/macro.py` & `engines/fno.py`**: Tracks **FII/DII Cash Flows**, **Index Futures/Options positioning**, and **Put/Call Ratios (PCR)**.
- **`engines/conviction.py`**: Analyzes **Delivery Percentage** to verify price move quality (Accumulation vs Speculation). Features robust fallbacks to prevent data pipeline failure.
- **`engines/sector.py`**: Ranks the searched stock against industry peers with **Dynamic Discovery** via `yfinance` and **Market Benchmark Fallbacks** for unique sectors. Filtered for complete data rows.
- **`engines/backtester.py`**: Quantitative simulation engine providing **Accuracy Proof** against 6 months of historical data.

---

## 2. Signal Weighting & Adaptive Logic
The final verdict is calculated via two distinct modes:

### I. Baseline Weighting (Manual Fallback)
| Module | Base Weight | What it Measures |
| :--- | :--- | :--- |
| **AI Predictor** | **30%** | Math-based forecast (LightGBM + XGBoost 60/40 split). |
| **Institutional Flows** | **20%** | Smart Money (FII/DII Cash + F&O positioning). |
| **Market Sentiment** | **20%** | Contextual score (Filings, News, Social). |
| **Conviction (Delivery)** | **15%** | Real accumulation vs day-trading noise. |
| **Sector Strength** | **15%** | Peer outperformance ranking. |

### II. F&O Availability Fallback
For stocks outside the F&O segment (non-derivatives), the system automatically redistributes weight from missing hedging data to **Institutional Cash Flow (+5%)** and the **AI Predictor (+10%)**.

---

## 3. Sentiment Intelligence Hierarchy
The system evaluates data quality across 6 circumstances to prevent noise from corrupting the signal:

| Circumstance | Sentiment Weight | Mode | Description |
| :--- | :--- | :--- | :--- |
| **1. Corporate Found** | **20%** | Full (Official) | Recent NSE/BSE filing (Ground Truth). |
| **2. High News Volume** | **10%** | Partial (News) | 3+ Forward-looking articles within 48h. |
| **3. Social Activity Only** | **5%** | Minimal (Social)| Screener/Social activity only. |
| **4. Reporting News Only** | **0%** | None | Data exists but is backward-looking only. |
| **5. No Data** | **0%** | None | Excluded to prevent signal corruption. |
| **6. Mixed Multi-tier** | **20%** | Full | Tier 1 Filings + Tier 2 News + Tier 3 Social. |

**Redistribution**: Floating weight flows proportionally to AI, Institutional, and Conviction modules.

---

## 4. Dashboard & Visual Identity
- **Snowy Quantitative Theme**: A modern, high-end visual design featuring a professional mountain background, **Glassmorphism UI**, and electric blue accents.
- **Interactive Charting Engine (v6.0.0)**:
    - **Dual Mode**: Instant toggle between **Area Line** and **Professional Candlestick** views.
    - **Dynamic RSI Panel**: A dedicated, enlarged RSI panel (150px) that appears in Candle mode with shaded overbought/oversold zones.
    - **Synchronized Crosshair**: Vertical dotted line that tracks across both Price and RSI charts simultaneously.
    - **Timeframe Selector**: Instant switching between 1M, 3M, 6M, 1Y, and 5Y periods.
- **Unified Information Feed**: A consolidated, clickable stream of News, Social, and Announcements with **Direct Source Hyperlinks** for verification.
- **Historical Accuracy Markers**: Visual BUY/SELL dots on the chart showing the strategy's past performance.

---

## 5. Common Operational Challenges & Solutions
- **Null-Safety**: Implemented robust `.get()` checks and redistribution logic to handle missing data in ETFs and small caps.
- **Cache Management**: Integrated cache-busting versioning for static assets and force-reload mechanisms to prevent stale data crashes.
- **Resource Optimization**: Separated heavy ML training (`batch_train.py`) and data fetching (`batch_runner.py`) from the web server (`app.py`), ensuring the site never crashes due to memory limits or scraping blocks on Free Tier hosting.
- **Namespace Integrity**: Separated logic engines from UI data dictionaries to prevent `AttributeError` namespace collisions.

---

## 6. Project Roadmap
- [x] Implement LightGBM & Meta-Model ensemble architecture.
- [x] Integrate FII/DII and stock-specific PCR data.
- [x] Overhaul Charting Engine with Candle/Line toggle and RSI panel.
- [x] Implement Market Regime and Breadth tracking.
- [x] Build Dynamic Sector Discovery & Benchmark Fallbacks.
- [x] Integrate High-Conviction Delivery Analysis.
- [x] Implement Forward-Looking News Filter.
- [x] **Decouple ML from Web Server (JSON Persistence Migration)**.
- [ ] Implement user accounts to track custom portfolios.
- [ ] Add real-time price alerts via Email/Telegram.

---
*Generated on June 12, 2026*
