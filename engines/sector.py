import pandas as pd
import os
import sys

# Path setup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import data_engine
import technical

def get_sector_analysis(symbol):
    """
    Analyzes the stock relative to its sector peers.
    """
    # 1. Load NSE Master List
    df_stocks = pd.read_csv('data/nse_stocks.csv')
    
    # 2. Identify sector
    symbol_raw = symbol.replace(".NS", "")
    stock_row = df_stocks[df_stocks['Symbol'] == symbol_raw]
    if stock_row.empty:
        return {"error": "Stock not found in master list"}
    
    sector = stock_row.iloc[0]['Sector']
    
    # 3. Find peers
    peers = df_stocks[df_stocks['Sector'] == sector]['Symbol'].tolist()
    
    # 4. Fetch peer data and compute metrics
    peer_metrics = []
    for peer in peers:
        df = data_engine.get_stock_data(peer, period="3mo")
        if df is not None and len(df) > 20:
            # 1-month return (approx 20 trading days)
            monthly_return = ((df['Close'].iloc[-1] - df['Close'].iloc[-20]) / df['Close'].iloc[-20]) * 100
            # Volatility (Std of daily returns)
            volatility = df['Close'].pct_change().std() * 100
            # Current RSI
            df = technical.add_technical_indicators(df)
            current_rsi = df['RSI'].iloc[-1]
            
            peer_metrics.append({
                'symbol': peer,
                'return_1m': round(float(monthly_return), 2),
                'volatility': round(float(volatility), 2),
                'rsi': round(float(current_rsi), 2)
            })
            
    # Rank peers by 1-month return
    peer_metrics = sorted(peer_metrics, key=lambda x: x['return_1m'], reverse=True)
    
    return {
        'sector': sector,
        'peers': peer_metrics
    }

def get_sector_heatmap():
    """
    Computes average return for each major sector.
    """
    df_stocks = pd.read_csv('data/nse_stocks.csv')
    sectors = df_stocks['Sector'].unique()
    
    heatmap = []
    for sector in sectors:
        sector_stocks = df_stocks[df_stocks['Sector'] == sector]['Symbol'].tolist()[:3] # Limit to top 3 for speed
        returns = []
        for s in sector_stocks:
            df = data_engine.get_stock_data(s, period="1mo")
            if df is not None and len(df) > 1:
                ret = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
                returns.append(ret)
        
        if returns:
            heatmap.append({
                'sector': sector,
                'avg_return': round(float(sum(returns) / len(returns)), 2)
            })
            
    return sorted(heatmap, key=lambda x: x['avg_return'], reverse=True)

if __name__ == "__main__":
    print("Analyzing Sector for TCS...")
    analysis = get_sector_analysis("TCS")
    print(analysis)
    
    print("\nSector Heatmap:")
    heatmap = get_sector_heatmap()
    print(heatmap)
