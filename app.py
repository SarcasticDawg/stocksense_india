from flask import Flask, render_template, request, jsonify, redirect
import sys
import os
import numpy as np
import concurrent.futures
import time
import threading
import pandas as pd
import gc
import traceback

# Add engines to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'engines'))

import data_engine
import predictor
import sentiment
import sector
import macro
import fno

app = Flask(__name__)

# Global state
_stock_predictor = None
_sentiment_analyzer = None
_models_loading = False
_training_lock = threading.Lock()

def load_models_task():
    global _stock_predictor, _sentiment_analyzer, _models_loading
    if _stock_predictor is None or _sentiment_analyzer is None:
        if _models_loading: return
        _models_loading = True
        print("--- [Server] Waking up AI Engines... ---")
        try:
            if _stock_predictor is None: _stock_predictor = predictor.StockPredictor()
            if _sentiment_analyzer is None: _sentiment_analyzer = sentiment.SentimentAnalyzer()
        except Exception as e:
            print(f"Error loading models: {e}")
        _models_loading = False
        print("--- [Server] AI Engines Ready. ---")

def get_models():
    global _stock_predictor, _sentiment_analyzer
    if _stock_predictor is None or _sentiment_analyzer is None:
        load_models_task()
    return _stock_predictor, _sentiment_analyzer

def safe_train(pred_engine, symbol):
    with _training_lock:
        print(f"--- [Background] Training for {symbol} ---")
        try:
            pred_engine.train(symbol)
        except Exception as e:
            print(f"Background Training Error: {e}")
        gc.collect()

def compute_combined_signal(symbol, pred_data, news_sentiment, reddit_sentiment, macro_data, sector_data):
    """
    Robust signal computation with extreme null safety.
    """
    # Initialize defaults
    price_score = 0.0
    news_score = float(news_sentiment) if news_sentiment is not None else 0.0
    reddit_score = float(reddit_sentiment) if reddit_sentiment is not None else 0.0
    macro_score = 0.0
    sector_score = 0.0
    
    # 1. Price Prediction
    if isinstance(pred_data, dict):
        change = pred_data.get('price_change_pct', 0)
        if change is not None:
            price_score = float(np.clip(float(change) / 5.0, -1, 1))

    # 2. Macro & Institutional
    if isinstance(macro_data, dict):
        vix_change = 0.0
        vix_obj = macro_data.get('India_VIX')
        if isinstance(vix_obj, dict): vix_change = float(vix_obj.get('change_pct', 0))
        
        sp_change = 0.0
        sp_obj = macro_data.get('SP500')
        if isinstance(sp_obj, dict): sp_change = float(sp_obj.get('change_pct', 0))
        
        base_macro = (sp_change - vix_change) / 10.0
        
        inst = macro_data.get('INSTITUTIONAL', {})
        if isinstance(inst, dict):
            cash_net = 0.0
            cash_obj = inst.get('cash')
            if isinstance(cash_obj, dict):
                cash_net = float(cash_obj.get('fii', 0)) + float(cash_obj.get('dii', 0))
            cash_score = float(np.clip(cash_net / 2500.0, -1, 1))
            
            view_map = {"BULLISH": 1.0, "BEARISH": -1.0, "Neutral": 0.0, "": 0.0}
            strength_map = {"Strong": 1.0, "Medium": 0.7, "Mild": 0.4, "Low": 0.2, "": 0.0}
            
            f_view = inst.get('fii_future', {})
            fii_f = 0.0
            if isinstance(f_view, dict):
                fii_f = view_map.get(f_view.get('view'), 0.0) * strength_map.get(f_view.get('strength'), 0.0)
            
            o_view = inst.get('fii_option', {})
            fii_o = 0.0
            if isinstance(o_view, dict):
                fii_o = view_map.get(o_view.get('view'), 0.0) * strength_map.get(o_view.get('strength'), 0.0)
            
            client_view = inst.get('client_view', 'Neutral')
            retail_score = -1.0 if client_view == "BULLISH" else (1.0 if client_view == "BEARISH" else 0.0)
            
            pcr_val = 1.0
            if isinstance(pred_data, dict) and isinstance(pred_data.get('pcr_data'), dict):
                pcr_val = float(pred_data['pcr_data'].get('pcr', 1.0))
            pcr_score = float(np.clip((pcr_val - 0.9) * 2.0, -1, 1))

            macro_score = float(np.clip(
                (base_macro * 0.25) + (cash_score * 0.25) + 
                ((fii_f + fii_o) / 2.0 * 0.25) + (retail_score * 0.1) + (pcr_score * 0.15), -1, 1))

    # 3. Sector
    if isinstance(sector_data, dict):
        peers = sector_data.get('peers', [])
        clean_sym = symbol.replace(".NS", "")
        symbol_rank = 0
        for i, p in enumerate(peers):
            if isinstance(p, dict) and p.get('symbol') == clean_sym:
                symbol_rank = i
                break
        if len(peers) > 0:
            sector_score = 1.0 - (symbol_rank / len(peers))
            sector_score = (sector_score * 2) - 1 
    
    total_score = (price_score * 0.3) + (news_score * 0.2) + (reddit_score * 0.15) + (macro_score * 0.2) + (sector_score * 0.15)
    
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
            'institutional': round(float(macro_score), 2),
            'sector': round(float(sector_score), 2)
        }
    }

analysis_cache = {}

@app.route('/')
def home():
    try:
        indices = data_engine.get_market_indices()
        index_summary = {}
        for name, df in indices.items():
            if df is not None and not df.empty:
                latest = df['Close'].iloc[-1]
                prev = df['Close'].iloc[0]
                change = ((latest - prev) / prev) * 100
                index_summary[name] = {"value": round(float(latest), 2), "change": round(float(change), 2)}
            else:
                index_summary[name] = {"value": "N/A", "change": 0.0}
        try:
            stocks_df = pd.read_csv('data/nse_stocks.csv')
            stock_list = stocks_df.sort_values(by='Symbol').to_dict('records')
        except:
            stock_list = []
        return render_template('home.html', indices=index_summary, stock_list=stock_list, app_name="StockIntel")
    except Exception as e:
        print(f"Home route error: {e}")
        return render_template('error.html', message="System error on home page.")

@app.route('/stock/<symbol>')
def stock_detail(symbol):
    try:
        if not symbol.endswith(".NS"): symbol += ".NS"
        if symbol in analysis_cache:
            ts, data = analysis_cache[symbol]
            if time.time() - ts < 1800: return render_template('stock.html', **data)

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            future_models = executor.submit(get_models)
            future_raw = executor.submit(data_engine.get_all_raw_data, symbol)
            future_macro = executor.submit(macro.get_macro_data)
            future_sector = executor.submit(sector.get_sector_analysis, symbol)
            future_chart = executor.submit(data_engine.get_chart_data, symbol)
            future_pcr = executor.submit(fno.get_stock_pcr, symbol)
            
            raw_data = future_raw.result()
            macro_data = future_macro.result() or {}
            sector_data = future_sector.result() or {}
            chart_data = future_chart.result() or {"labels":[], "prices":[]}
            pcr_data = future_pcr.result() or {"pcr": 1.0, "sentiment": "Neutral"}
            pred_engine, sent_engine = future_models.result()
            
        if not raw_data or raw_data.get('price_data') is None:
            return render_template('error.html', message=f"Could not fetch data for {symbol}")

        cur_price = round(float(raw_data['price_data']['Close'].iloc[-1]), 2)
        lstm_path = os.path.join('models', f'lstm_{symbol}.keras')
        
        if not os.path.exists(lstm_path):
            threading.Thread(target=safe_train, args=(pred_engine, symbol)).start()
            pred_data = {'current_price': cur_price, 'predicted_price': "Training...", 'price_change_pct': 0.0, 'trend_prob_up': 0.5}
        else:
            pred_data = pred_engine.get_prediction(symbol) or {'current_price': cur_price, 'predicted_price': "Error", 'price_change_pct': 0.0, 'trend_prob_up': 0.5}

        pred_data['pcr_data'] = pcr_data
        
        news_texts = [n['text'] for n in raw_data.get('news', [])]
        reddit_texts = [p['title'] + " " + " ".join(p['comments']) for p in raw_data.get('reddit', [])]
        reddit_scores = [p['score'] for p in raw_data.get('reddit', [])]
        
        news_sent = sent_engine.analyze_batch(news_texts) if news_texts else 0.0
        reddit_sent = sent_engine.analyze_batch(reddit_texts, weights=reddit_scores) if reddit_texts else 0.0
        
        signal = compute_combined_signal(symbol, pred_data, news_sent, reddit_sent, macro_data, sector_data)
        
        render_params = {
            'symbol': symbol, 'prediction': pred_data, 'pcr': pcr_data, 'macro': macro_data, 'sector': sector_data, 'signal': signal,
            'sentiment': {'news': news_sent, 'reddit': reddit_sent, 'mentions': {'news': len(news_texts), 'reddit': len(reddit_texts)}},
            'news': raw_data.get('news', []), 'chart_data': chart_data, 'app_name': "StockIntel"
        }
        if pred_data.get('predicted_price') != "Training...":
            analysis_cache[symbol] = (time.time(), render_params)
        return render_template('stock.html', **render_params)
    except Exception as e:
        print(f"Stock route error for {symbol}: {e}")
        traceback.print_exc()
        return render_template('error.html', message=f"Internal Error processing {symbol}")

@app.route('/api/chart/<symbol>')
def api_chart(symbol):
    period = request.args.get('period', '3mo')
    chart_data = data_engine.get_chart_data(symbol, period=period)
    return jsonify(chart_data)

@app.route('/sentiment/<symbol>')
def sentiment_page(symbol):
    if not symbol.endswith(".NS"): symbol += ".NS"
    _, sent_engine = get_models()
    raw_data = data_engine.get_all_raw_data(symbol) or {}
    news_texts = [n['text'] for n in raw_data.get('news', [])]
    reddit_texts = [p['title'] + " " + " ".join(p['comments']) for p in raw_data.get('reddit', [])]
    news_sent = sent_engine.analyze_batch(news_texts) if news_texts else 0.0
    reddit_sent = sent_engine.analyze_batch(reddit_texts) if reddit_texts else 0.0
    return render_template('sentiment.html', symbol=symbol, news=raw_data.get('news', []), app_name="StockIntel",
                           sentiment={'news': news_sent, 'reddit': reddit_sent, 'mentions': {'news': len(news_texts), 'reddit': len(reddit_texts)}})

@app.route('/sector/<symbol>')
def sector_page(symbol):
    if not symbol.endswith(".NS"): symbol += ".NS"
    return render_template('sector.html', symbol=symbol, sector=sector.get_sector_analysis(symbol) or {}, app_name="StockIntel")

@app.route('/stock_search')
def stock_search():
    symbol = request.args.get('symbol', '').upper()
    return redirect(f'/stock/{symbol}') if symbol else redirect('/')

if __name__ == '__main__':
    threading.Thread(target=load_models_task).start()
    app.run(debug=True, port=8080, use_reloader=False)
