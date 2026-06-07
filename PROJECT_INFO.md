# StockSense India - System Design Specification

## Project Overview
StockSense India is a professional-grade financial intelligence dashboard built with Flask. It provides predictive analytics, multi-source sentiment analysis, and institutional flow tracking for NSE stocks using a Market-Adaptive Ensemble architecture.

---

## 1. Technical Architecture (Market-Adaptive Ensemble)
The system is designed as a **Self-Optimizing Ensemble** that transitions from a rule-based baseline to a machine-learned quantitative model.

### A. Modular "Engine" Core
- **`app.py`**: The central orchestrator. Manages parallel data fetching via `ThreadPoolExecutor`, handles a 30-minute intelligence cache, and dynamically blends signals using Contextual Weighting.
- **`engines/predictor.py`**: The Machine Learning core. 
    - **LightGBM (60% of module)**: Regressor for next-day price forecasting. Handles tabular features like lags (1-10d) and rolling stats.
    - **XGBoost (40% of module)**: Classifier for 5-day directional probability.
    - *Self-Optimization*: This 60/40 internal split is the quantitative baseline; the Meta-model automatically rebalances these weights if backtesting shows one model outperforming the other in the current regime.
- **`engines/meta_model.py`**: The "Brain" of the ensemble. Uses **Logistic Regression** to learn optimal weights for all signals based on historical outcomes.
- **`engines/market_context.py`**: Regime detection module (**BULL/BEAR/SIDEWAYS**) and **Market Breadth** tracker (Nifty 50 stocks above 50-DMA).
- **`engines/sentiment.py`**: High-speed NLP using **VADER** with a custom **Finance Lexicon**. Features a **Forward-Looking Filter** to prioritize future-focused signals over historical reporting.
- **`engines/corporate.py`**: Corporate intelligence module that scrapes official NSE/BSE filings (Dividends, Results, Buybacks) from Screener.in.
- **`engines/macro.py` & `engines/fno.py`**: Tracks **FII/DII Cash Flows**, **Index Futures/Options positioning**, and **Put/Call Ratios (PCR)**.
- **`engines/conviction.py`**: Analyzes **Delivery Percentage** to verify price move quality (Accumulation vs Speculation).
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
For stocks outside the F&O segment (non-derivatives), the system automatically redistributes the signal:
- **Cash Flow Importance**: Increased by 5% within the Institutional slice.
- **AI Predictor Importance**: Increased by 10% to compensate for lack of hedging data.
- **Global Macro Importance**: Absorbs the remaining volatility signals.

---

## 3. Sentiment Intelligence Hierarchy
The system evaluates data quality across 6 circumstances to prevent noise from corrupting the signal:

| Circumstance | Sentiment Weight | Mode | Description |
| :--- | :--- | :--- | :--- |
| **1. Corporate Found** | **20%** | Full (Official) | Recent NSE/BSE filing (Ground Truth). |
| **2. High News Volume** | **10%** | Partial (News) | 3+ Forward-looking articles within 48h. |
| **3. Social Activity Only** | **5%** | Minimal (Social)| Reddit activity (5+ posts) only. |
| **4. Reporting News Only** | **0%** | None | Data exists but is backward-looking only. |
| **5. No Data** | **0%** | None | Excluded to prevent signal corruption. |
| **6. Mixed Multi-tier** | **20%** | Full | Tier 1 Filings + Tier 2 News + Tier 3 Social. |

**Source Weights**: Tier 1 Institutional (50% influence, 2.5x multiplier), Tier 2 Professional News (30%), Tier 3 Social (20%).

---

## 4. Dashboard & Intelligence Features
- **Interactive Analytics**: Chart.js integration with blue gradients, hover tooltips, and dynamic period selection (1M, 3M, 1Y, 5Y).
- **Historical Accuracy Markers**: Visual BUY/SELL dots on the chart showing where the AI *would have* acted in the past 6 months.
- **Unified Information Feed**: A consolidated "News & Announcements" stream tagged by source (Institutional, News, Social).
- **Regime Safety Valve**: In a **BEAR** regime, SELL signals are amplified by 1.2x to prioritize capital protection.
- **Batch AI Trainer**: `batch_train.py` allows for nightly pre-computation of the Nifty 50 for zero-latency loading.

---
*Generated on June 7, 2026*
