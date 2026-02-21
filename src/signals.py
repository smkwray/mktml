import pandas as pd
import numpy as np

def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    """Calculates Simple Moving Average."""
    return series.rolling(window=window).mean()

def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculates Relative Strength Index."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculates MACD and Signal line."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def calculate_bollinger(series: pd.Series, window: int = 20, std_dev: float = 2.0):
    """
    Calculates Bollinger Bands.
    Returns: upper band, middle band (SMA), lower band, and %B position.
    %B = (price - lower) / (upper - lower)  → 0 = at lower band, 1 = at upper band
    """
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    pct_b = (series - lower) / (upper - lower)  # Position within bands
    return upper, sma, lower, pct_b

def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    """
    Calculates Stochastic Oscillator.
    %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    %D = SMA of %K
    """
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    k = 100 * (df['close'] - low_min) / (high_max - low_min)
    d = k.rolling(window=d_period).mean()
    return k, d

def calculate_52week_distance(df: pd.DataFrame):
    """
    Calculates distance from 52-week high and low.
    Returns: dist_52w_high (negative = below high), dist_52w_low (positive = above low)
    """
    rolling_high = df['high'].rolling(window=252, min_periods=50).max()
    rolling_low = df['low'].rolling(window=252, min_periods=50).min()
    
    dist_high = (df['close'] - rolling_high) / rolling_high  # Negative when below high
    dist_low = (df['close'] - rolling_low) / rolling_low    # Positive when above low
    
    return dist_high, dist_low


def generate_signals(df: pd.DataFrame, ticker: str = None, dividend_info: dict = None) -> pd.DataFrame:
    """
    Generates buy/sell signals based on momentum strategies.
    Adds 'signal' column: 1 (Buy), 0 (Hold/Exit).
    
    Dividend stocks get more lenient exit conditions.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import RSI_EXIT
    
    df = df.copy()
    
    # Auto-detect if this is a dividend stock
    is_dividend = False
    if dividend_info and dividend_info.get('is_dividend'):
        is_dividend = True
    elif ticker:
        # Try to auto-detect from yfinance (cached)
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            div_yield = stock.info.get('dividendYield', 0) or 0
            is_dividend = div_yield > 0.01  # >1% yield
        except:
            pass
    
    # Dividend stocks get more lenient exit conditions
    rsi_exit_threshold = 85 if is_dividend else RSI_EXIT  # Higher threshold for div stocks
    sell_buffer = 0.03 if is_dividend else 0  # 3% buffer below SMA200
    
    # 1. Moving Average Crossover for Trend
    df['sma_50'] = calculate_sma(df['close'], 50)
    df['sma_200'] = calculate_sma(df['close'], 200)
    
    # Golden Cross: 50 crosses above 200
    df['trend_signal'] = 0
    df.loc[df['sma_50'] > df['sma_200'], 'trend_signal'] = 1  # Bullish regime
    df.loc[df['sma_50'] <= df['sma_200'], 'trend_signal'] = -1  # Bearish regime
    
    # 2. RSI for Overbought/Oversold
    df['rsi'] = calculate_rsi(df['close'], 14)
    
    # 3. MACD for Momentum Confirmation
    df['macd'], df['macd_signal'] = calculate_macd(df['close'])
    df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(int)
    
    # 4. Bollinger Bands for Volatility
    df['bb_upper'], df['bb_middle'], df['bb_lower'], df['bb_pct_b'] = calculate_bollinger(df['close'])
    
    # 5. Stochastic Oscillator for Momentum
    df['stoch_k'], df['stoch_d'] = calculate_stochastic(df)
    df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)  # Potential buy signal
    df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)  # Potential sell signal
    
    # 6. 52-Week High/Low Distance
    df['dist_52w_high'], df['dist_52w_low'] = calculate_52week_distance(df)
    
    # Initialize raw signal
    df['raw_signal'] = 0
    
    # Long condition: Bullish trend AND RSI < 70 AND MACD bullish
    long_condition = (df['trend_signal'] == 1) & (df['rsi'] < 70) & (df['macd_bullish'] == 1)
    df.loc[long_condition, 'raw_signal'] = 1
    
    # Exit condition varies by stock type
    if is_dividend:
        # Dividend stocks: More lenient exits (hold through minor dips)
        # Only exit if: Bearish trend AND significantly below SMA200 OR extreme overbought
        price_below_buffer = df['close'] < df['sma_200'] * (1 - sell_buffer)
        exit_condition = ((df['trend_signal'] == -1) & price_below_buffer) | (df['rsi'] > rsi_exit_threshold)
    else:
        # Growth stocks: Standard exit logic
        exit_condition = (df['trend_signal'] == -1) | (df['rsi'] > rsi_exit_threshold)
    
    df.loc[exit_condition, 'raw_signal'] = 0
    
    # 4. Signal Persistence: Require 2 consecutive days
    df['signal'] = (df['raw_signal'].rolling(2).min() == 1).astype(int)
    
    # Mark as dividend stock for reporting
    df['is_dividend'] = is_dividend
    
    return df
