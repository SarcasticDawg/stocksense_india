from flask import Flask, render_template, request, jsonify, redirect
import sys
import os
import numpy as np

# Add engines to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'engines'))

import data_engine
import predictor
import sentiment
import sector
import macro

app = Flask(__name__)

# Lazy-loading containers for AI models
_stock_predictor = None
_sentiment_analyzer = None

def get_models():
    """Lazy-loads the heavy models only when a stock search is performed."""
    global _stock_predictor, _sentiment_analyzer
    if _stock_predictor is None:
        print("Lazy-loading ML Predictor Engine (TensorFlow/XGBoost)...")
        _stock_predictor = predictor.StockPredictor()
    if _sentiment_analyzer is None:
        print("Lazy-loading Sentiment Engine (FinBERT/Torch)...")
        _sentiment_analyzer = sentiment.SentimentAnalyzer()
    return _stock_predictor, _sentiment_analyzer

def compute_combined_signal(symbol, pred_data, news_sentiment, reddit_sentiment, macro_data, sector_data):
    """
    Computes a Buy/Hold/Sell signal based on weighted scores.
    Weights: 
    - Price prediction: 30%
    - News sentiment: 20%
    - Reddit sentiment: 15%
    - Macro signals: 20%
    - Sector position: 15%
    """
    # 1. Price Prediction Score (-1 to 1)
    price_change = pred_data['price_change_pct']
    price_score = np.clip(price_change / 5.0, -1, 1) 
    
    # 2. News and Reddit (already -1 to 1)
    news_score = news_sentiment
    reddit_score = reddit_sentiment
    
    # 3. Macro Score
    vix_change = macro_data.get('India_VIX', {}).get('change_pct', 0)
    sp_change = macro_data.get('S&P_500', {}).get('change_pct', 0)
    macro_score = np.clip((sp_change - vix_change) / 10.0, -1, 1)
    
    # 4. Sector Score
    peers = sector_data.get('peers', [])
    symbol_rank = 0
    for i, p in enumerate(peers):
        if p['symbol'] == symbol.replace(".NS", ""):
            symbol_rank = i
            break
    sector_score = 1.0 - (symbol_rank / max(len(peers), 1))
    sector_score = (sector_score * 2) - 1 
    
    total_score = (
        (price_score * 0.30) +
        (news_score * 0.20) +
        (reddit_score * 0.15) +
        (macro_score * 0.20) +
        (sector_score * 0.15)
    )
    
    verdict = "HOLD"
    if total_score > 0.3: verdict = "BUY"
    elif total_score < -0.3: verdict = "SELL"
    
    return {
        'verdict': verdict,
        'score': round(float(total_score), 2),
        'breakdown': {
            'price': round(float(price_score), 2),
            'news': round(float(news_score), 2),
            'reddit': round(float(reddit_score), 2),
            'macro': round(float(macro_score), 2),
            'sector': round(float(sector_score), 2)
        }
    }

from flask import Flask, render_template, request, jsonify, redirect
import sys
import os
import numpy as np
import concurrent.futures # For Parallel Processing
import time
import threading

# Add engines to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'engines'))

import data_engine
import predictor
import sentiment
import sector
import macro

app = Flask(__name__)

# Lazy-loading containers
_stock_predictor = None
_sentiment_analyzer = None
_models_loading = False

def load_models_task():
    """Background task to pre-warm the AI models."""
    global _stock_predictor, _sentiment_analyzer, _models_loading
    if _stock_predictor is None or _sentiment_analyzer is None:
        _models_loading = True
        print("--- [Background] Waking up AI Engines... ---")
        if _stock_predictor is None: _stock_predictor = predictor.StockPredictor()
        if _sentiment_analyzer is None: _sentiment_analyzer = sentiment.SentimentAnalyzer()
        _models_loading = False
        print("--- [Background] AI Engines Ready. ---")

def get_models():
    global _stock_predictor, _sentiment_analyzer
    if _stock_predictor is None or _sentiment_analyzer is None:
        load_models_task()
    return _stock_predictor, _sentiment_analyzer

# Simple Cache (In-memory for prototype speed)
analysis_cache = {}

@app.route('/')
def home():
    # Pre-warm AI in a separate thread when user hits the homepage
    threading.Thread(target=load_models_task).start()
    
    indices = data_engine.get_market_indices()
    index_summary = {}
    for name, df in indices.items():
        if df is not None:
            latest = df['Close'].iloc[-1]
            prev = df['Close'].iloc[0]
            change = ((latest - prev) / prev) * 100
            index_summary[name] = {"value": round(latest, 2), "change": round(change, 2)}
    
    import pandas as pd
    try:
        stocks_df = pd.read_csv('data/nse_stocks.csv')
        stocks_df = stocks_df.sort_values(by='Symbol')
        stock_list = stocks_df.to_dict('records')
    except:
        stock_list = []
    
    return render_template('home.html', indices=index_summary, stock_list=stock_list, app_name="StockIntel")

@app.route('/stock/<symbol>')
def stock_detail(symbol):
    if not symbol.endswith(".NS"): symbol += ".NS"
    
    # 0. Check Cache (30 min TTL)
    if symbol in analysis_cache:
        timestamp, data = analysis_cache[symbol]
        if time.time() - timestamp < 1800: # 30 minutes
            print(f"Loading {symbol} from Cache (INSTANT)")
            return render_template('stock.html', **data)

    # 1. Start Parallel Fetching
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Launch all heavy tasks at once
        future_models = executor.submit(get_models)
        future_raw = executor.submit(data_engine.get_all_raw_data, symbol)
        future_macro = executor.submit(macro.get_macro_data)
        future_sector = executor.submit(sector.get_sector_analysis, symbol)
        future_chart = executor.submit(data_engine.get_chart_data, symbol)
        
        # Wait for data (these run while models are warming up)
        raw_data = future_raw.result()
        macro_data = future_macro.result()
        sector_data = future_sector.result()
        chart_data = future_chart.result()
        
        if "error" in sector_data:
            return render_template('error.html', message=sector_data["error"])
        
        # Now get models (might already be warmed up)
        pred_engine, sent_engine = future_models.result()
        
    # 2. Sequential AI Processing (requires heavy computation)
    pred_data = pred_engine.get_prediction(symbol)
    
    news_texts = [n['text'] for n in raw_data['news']]
    reddit_texts = [p['title'] + " " + " ".join(p['comments']) for p in raw_data['reddit']]
    
    news_sentiment = sent_engine.analyze_batch(news_texts)
    reddit_sentiment = sent_engine.analyze_batch(reddit_texts)
    
    # 3. Final Signal
    signal = compute_combined_signal(symbol, pred_data, news_sentiment, reddit_sentiment, macro_data, sector_data)
    
    render_params = {
        'symbol': symbol,
        'prediction': pred_data,
        'sentiment': {'news': news_sentiment, 'reddit': reddit_sentiment},
        'macro': macro_data,
        'sector': sector_data,
        'signal': signal,
        'news': raw_data['news'],
        'chart_data': chart_data,
        'app_name': "StockIntel"
    }
    
    # Save to Cache
    analysis_cache[symbol] = (time.time(), render_params)
    
    return render_template('stock.html', **render_params)

@app.route('/sentiment/<symbol>')
def sentiment_page(symbol):
    if not symbol.endswith(".NS"): symbol += ".NS"
    _, sent_engine = get_models()
    raw_data = data_engine.get_all_raw_data(symbol)
    news_texts = [n['text'] for n in raw_data['news']]
    reddit_texts = [p['title'] + " " + " ".join(p['comments']) for p in raw_data['reddit']]
    
    news_sentiment = sent_engine.analyze_batch(news_texts)
    reddit_sentiment = sent_engine.analyze_batch(reddit_texts)
    
    return render_template('sentiment.html', 
                           symbol=symbol, 
                           sentiment={'news': news_sentiment, 'reddit': reddit_sentiment},
                           news=raw_data['news'],
                           app_name="StockIntel")

@app.route('/sector/<symbol>')
def sector_page(symbol):
    if not symbol.endswith(".NS"): symbol += ".NS"
    sector_data = sector.get_sector_analysis(symbol)
    return render_template('sector.html', symbol=symbol, sector=sector_data, app_name="StockIntel")

@app.route('/signal/<symbol>')
def signal_page(symbol):
    if not symbol.endswith(".NS"): symbol += ".NS"
    pred_engine, sent_engine = get_models()
    pred_data = pred_engine.get_prediction(symbol)
    raw_data = data_engine.get_all_raw_data(symbol)
    news_texts = [n['text'] for n in raw_data['news']]
    news_sentiment = sent_engine.analyze_batch(news_texts)
    macro_data = macro.get_macro_data()
    sector_data = sector.get_sector_analysis(symbol)
    
    # Simple signal re-calc for the report page
    signal = compute_combined_signal(symbol, pred_data, news_sentiment, 0.0, macro_data, sector_data)
    
    return render_template('signal.html', symbol=symbol, signal=signal, app_name="StockIntel")

@app.route('/stock_search')
def stock_search():
    symbol = request.args.get('symbol', '').upper()
    if symbol:
        return redirect(f'/stock/{symbol}')
    return redirect('/')

if __name__ == '__main__':
    # Changed to 8080 because 5060 is often blocked by browsers (UNSAFE_PORT)
    app.run(debug=True, port=8080)
