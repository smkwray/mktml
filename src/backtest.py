import pandas as pd
import numpy as np

def run_backtest(df: pd.DataFrame, initial_capital: float = 10000.0) -> dict:
    """
    Runs a simple backtest on the provided DataFrame.
    Assumes 'signal' column exists: 1 (Long), 0 (Cash).
    """
    if 'signal' not in df.columns:
        raise ValueError("DataFrame must contain 'signal' column")
        
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    
    # Shift signal by 1 because we trade at the Open/Close of the NEXT day based on TODAY's signal
    # Simplification: We assume we can trade at the Close of the signal day or Open of next.
    # Standard: Trade at Open of next day.
    # Here: We'll implement Strategy Returns = Signal(d-1) * Returns(d)
    
    df['strategy_returns'] = df['signal'].shift(1) * df['returns']
    
    # Equity Curve
    df['equity'] = initial_capital * (1 + df['strategy_returns']).cumprod()
    
    # Metrics
    total_return = (df['equity'].iloc[-1] / initial_capital) - 1
    num_days = len(df)
    years = num_days / 252
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # Drawdown
    rolling_max = df['equity'].cummax()
    drawdown = (df['equity'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Sharpe (simplified, assuming 0 risk-free)
    daily_mean = df['strategy_returns'].mean()
    daily_std = df['strategy_returns'].std()
    sharpe = (daily_mean / daily_std) * np.sqrt(252) if daily_std > 0 else 0
    
    return {
        'total_return': total_return,
        'cagr': cagr,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'final_equity': df['equity'].iloc[-1],
        'equity_curve': df['equity']
    }
