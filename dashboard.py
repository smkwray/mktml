import os
import sys
from datetime import datetime

# Prevent __pycache__ generation (helps cloud sync churn).
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
sys.dont_write_bytecode = True

import duckdb
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Get the absolute path to the project root (market folder)
# Works whether dashboard.py is in /market/ or /market/src/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(_SCRIPT_DIR) == 'src':
    PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)  # Goes from src/ to market/
else:
    PROJECT_ROOT = _SCRIPT_DIR  # Already in market/

sys.path.append(os.path.join(PROJECT_ROOT, "src"))

DB_PATH = os.path.join(PROJECT_ROOT, "data", "market_data.duckdb")
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "performance_audit.html")

def get_connection():
    return duckdb.connect(DB_PATH, read_only=True)

def generate_dashboard():
    print("Generating Performance Dashboard...")
    con = get_connection()
    
    # 1. Fetch Recommendation History
    df_recs = con.execute("SELECT * FROM recommendation_history").df()
    
    # 2. Fetch Market Breadth Data from recommendation_history (has sma_200)
    print("Computing Market Breadth (this may take a moment)...")
    # Using recommendation_history since price_history doesn't store SMA values
    
    try:
        query_breadth = """
            SELECT 
                date,
                COUNT(*) as total_tickers,
                SUM(CASE WHEN price_at_rec > sma_200 THEN 1 ELSE 0 END) as above_sma200
            FROM recommendation_history
            WHERE sma_200 IS NOT NULL AND sma_200 > 0
            GROUP BY date
            ORDER BY date
        """
        df_breadth = con.execute(query_breadth).df()
        if not df_breadth.empty:
            df_breadth['breadth_pct'] = df_breadth['above_sma200'] / df_breadth['total_tickers']
        else:
            df_breadth = pd.DataFrame(columns=['date', 'breadth_pct'])
    except Exception as e:
        print(f"Warning: Could not compute market breadth: {e}")
        df_breadth = pd.DataFrame(columns=['date', 'breadth_pct'])

    con.close()
    
    if df_recs.empty:
        print("No recommendation history found. Cannot generate dashboard.")
        return

    # --- Preprocessing ---
    df_closed = df_recs[df_recs['status'] == 'CLOSED'].copy()
    df_closed['date'] = pd.to_datetime(df_closed['date'])
    df_breadth['date'] = pd.to_datetime(df_breadth['date'])
    
    # --- PLOTTING ---
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Strategy Equity Curve (Simulated)", "Confidence Discriminator", "Market Breadth (% > SMA200)"),
        vertical_spacing=0.1,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": False}]]
    )

    # 1. Equity Curve
    # Sort by date
    df_closed = df_closed.sort_values('date')
    
    # Calculate cumulative return
    # Assumption: Equal weight, reinvested. 
    # For simplicity: Sum of returns (Arithmetic) or Cumulative Product (Geometric)
    # Let's do a simple dummy cumulative sum of 1-week returns for now
    df_closed['cum_return'] = df_closed['perf_1w'].cumsum()
    
    # Aggregate by date to show daily equity curve
    daily_perf = df_closed.groupby('date')['perf_1w'].sum().cumsum()
    
    fig.add_trace(
        go.Scatter(x=daily_perf.index, y=daily_perf.values, name="Strategy (Cum Sum 1W)", mode='lines+markers'),
        row=1, col=1
    )
    
    # Add Benchmark (SPY) comparison if possible - omitted for now as we'd need SPY data specifically loaded
    # fig.add_trace(go.Scatter(x=..., y=..., name="SPY"), row=1, col=1)

    # 2. Discriminator Chart
    # Bin confidence
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ['50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    df_recs['conf_bin'] = pd.cut(df_recs['confidence'], bins=bins, labels=labels)
    
    # Calculate Win Rate per bin (using perf_1w > 0)
    # Note: We use all recs that have performance data, even if not fully closed (1m) if 1w is populated
    df_perf = df_recs[df_recs['perf_1w'].notna()].copy()
    df_perf['win'] = df_perf['perf_1w'] > 0
    
    stats = df_perf.groupby('conf_bin', observed=True).agg({
        'win': 'mean',
        'ticker': 'count'
    }).rename(columns={'win': 'win_rate', 'ticker': 'count'})
    
    fig.add_trace(
        go.Bar(x=stats.index, y=stats['win_rate'], name="Win Rate", marker_color='blue'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=stats.index, y=stats['count'], name="Signal Count", mode='markers+lines', marker=dict(color='red')),
        row=2, col=1, secondary_y=True
    )

    # 3. Market Breadth
    fig.add_trace(
        go.Scatter(x=df_breadth['date'], y=df_breadth['breadth_pct'], name="% > SMA200", line=dict(color='purple')),
        row=3, col=1
    )
    
    # Add threshold lines with high-contrast colors for dark mode
    fig.add_hline(y=0.5, line_dash="dash", line_color="#888888", row=3, col=1)
    fig.add_hline(y=0.2, line_dash="dash", line_color="#ff6b6b", annotation_text="Oversold", row=3, col=1)
    fig.add_hline(y=0.8, line_dash="dash", line_color="#51cf66", annotation_text="Overbought", row=3, col=1)

    # NATIVE DARK MODE: Use plotly_dark with explicit high-contrast colors
    fig.update_layout(
        height=1200, 
        title_text="Market Analyzer Performance Dashboard",
        template="plotly_dark",  # Native dark template
        showlegend=True,
        paper_bgcolor='#1a1a2e',  # Dark blue-black background
        plot_bgcolor='#16213e',   # Slightly lighter plot area
        font=dict(color='#e8e8e8', size=12),  # Light text
        title_font=dict(size=22, color='#ffffff')
    )
    
    # HIGH CONTRAST COLORS for dark mode visibility
    # Equity curve - bright cyan
    fig.update_traces(line=dict(color='#00d9ff', width=2), selector=dict(name="Strategy (Cum Sum 1W)"))
    
    # Win rate bars - bright blue
    fig.update_traces(marker_color='#4dabf7', selector=dict(type='bar', name="Win Rate"))
    
    # Signal count line - bright orange
    fig.update_traces(line=dict(color='#ffa94d', width=2), marker=dict(color='#ffa94d'), selector=dict(name="Signal Count"))
    
    # Market breadth - bright purple
    fig.update_traces(line=dict(color='#da77f2', width=2), selector=dict(name="% > SMA200"))
    
    # Grid lines - subtle but visible
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#2d3a52', zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#2d3a52', zeroline=False)
    
    # Axis labels
    fig.update_yaxes(title_text="Cumulative Return", title_font=dict(color='#00d9ff'), row=1, col=1)
    fig.update_yaxes(title_text="Win Rate", title_font=dict(color='#4dabf7'), row=2, col=1)
    fig.update_yaxes(title_text="Count", title_font=dict(color='#ffa94d'), row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="% > SMA200", range=[0, 1], title_font=dict(color='#da77f2'), row=3, col=1)

    print(f"Saving report to {REPORT_PATH}...")
    fig.write_html(
        REPORT_PATH, 
        include_plotlyjs=True,
        full_html=True,
        config={'displayModeBar': True, 'responsive': True}
    )
    
    print("Dashboard generated successfully.")

if __name__ == "__main__":
    generate_dashboard()
