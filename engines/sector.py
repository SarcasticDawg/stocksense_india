import pandas as pd
import os
import sys
import yfinance as yf

# Path setup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import data_engine
import technical

def get_sector_analysis(symbol):
    """
    Analyzes the stock relative to its sector peers with dynamic discovery.
    """
    try:
        # 1. Load NSE Master List
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'nse_stocks.csv')
        df_stocks = pd.read_csv(csv_path)
        
        # 2. Identify sector
        symbol_raw = symbol.replace(".NS", "")
        stock_row = df_stocks[df_stocks['Symbol'] == symbol_raw]
        
        sector_name = None
        if not stock_row.empty:
            sector_name = stock_row.iloc[0]['Sector']
        else:
            # Dynamic lookup for stocks not in our local list
            t = yf.Ticker(symbol if symbol.endswith(".NS") else symbol + ".NS")
            sector_name = t.info.get('sector', "Other")

        # 3. Find peers
        peers = df_stocks[df_stocks['Sector'] == sector_name]['Symbol'].tolist()
        
        # FALLBACK: If sparse sector, use Market Leaders to compare strength
        is_fallback = False
        if len(peers) < 2:
            peers = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']
            is_fallback = True
        
        # 4. Fetch peer data
        peer_metrics = []
        # Ensure current symbol is always in the comparison
        if symbol_raw not in peers:
            peers.append(symbol_raw)

        for p in peers:
            p_ns = p if p.endswith(".NS") else p + ".NS"
            df = data_engine.get_stock_data(p_ns, period="3mo")
            if df is not None and len(df) > 20:
                # 1-month return
                m_ret = ((df['Close'].iloc[-1] - df['Close'].iloc[-20]) / df['Close'].iloc[-20]) * 100
                # RSI
                df = technical.add_technical_indicators(df)
                rsi = df['RSI'].iloc[-1]
                
                peer_metrics.append({
                    'symbol': p,
                    'return_1m': round(float(m_ret), 2),
                    'rsi': round(float(rsi), 2)
                })
            
        # Rank by 1-month return
        peer_metrics = sorted(peer_metrics, key=lambda x: x['return_1m'], reverse=True)
        
        final_sector_label = sector_name
        if is_fallback:
            final_sector_label += " (Market Benchmark)"
            
        return {
            'sector': final_sector_label,
            'peers': peer_metrics[:6]
        }
    except Exception as e:
        print(f"Sector error: {e}")
        return {"sector": "Unknown", "peers": []}

def get_sector_heatmap():
    try:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'nse_stocks.csv')
        df_stocks = pd.read_csv(csv_path)
        sectors = df_stocks['Sector'].unique()
        heatmap = []
        for sector in sectors:
            sector_stocks = df_stocks[df_stocks['Sector'] == sector]['Symbol'].tolist()[:3]
            returns = []
            for s in sector_stocks:
                df = data_engine.get_stock_data(s + ".NS", period="1mo")
                if df is not None and len(df) > 1:
                    ret = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
                    returns.append(ret)
            if returns:
                heatmap.append({
                    'sector': sector,
                    'avg_return': round(float(sum(returns) / len(returns)), 2)
                })
        return sorted(heatmap, key=lambda x: x['avg_return'], reverse=True)
    except: return []

if __name__ == "__main__":
    print(get_sector_analysis("BHARTIARTL"))
