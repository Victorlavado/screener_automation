"""
Technical indicators calculation module.
Computes various indicators from OHLCV data.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np


def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return series.rolling(window=period, min_periods=period).mean()


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.

    Uses the standard Wilder smoothing method.
    """
    delta = series.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Use Wilder's smoothing (equivalent to EMA with alpha = 1/period)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_pct_change(series: pd.Series, periods: int) -> pd.Series:
    """Calculate percentage change over N periods."""
    return series.pct_change(periods=periods) * 100


def calculate_volume_sma(volume: pd.Series, period: int) -> pd.Series:
    """Calculate volume Simple Moving Average."""
    return calculate_sma(volume, period)


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return true_range.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def calculate_adr_pct(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Average Daily Range as percentage.
    ADR% = average of (High - Low) / Close * 100 over period.
    """
    daily_range_pct = (high - low) / close * 100
    return daily_range_pct.rolling(window=period, min_periods=period).mean()


def calculate_volatility(close: pd.Series, period: int = 21) -> pd.Series:
    """
    Calculate historical volatility as annualized standard deviation of returns.
    Returns volatility as percentage.
    """
    returns = close.pct_change()
    volatility = returns.rolling(window=period, min_periods=period).std() * np.sqrt(252) * 100
    return volatility


def compute_all_indicators(
    ohlcv: Dict[str, pd.DataFrame],
    fundamentals: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Compute all indicators for screening.

    Args:
        ohlcv: Dict mapping ticker -> OHLCV DataFrame
        fundamentals: DataFrame with fundamental data (optional)

    Returns:
        DataFrame with one row per ticker and all computed indicators
    """
    results = []

    for ticker, df in ohlcv.items():
        if df.empty or len(df) < 200:  # Need enough data for 200-day MA
            continue

        try:
            close = df['Close']
            high = df['High']
            low = df['Low']
            volume = df['Volume']

            # Get latest values
            latest_close = close.iloc[-1]
            latest_volume = volume.iloc[-1]

            # Calculate indicators
            row = {
                'ticker': ticker,
                'close': latest_close,
                'open': df['Open'].iloc[-1],
                'high': high.iloc[-1],
                'low': low.iloc[-1],
                'volume': latest_volume,

                # Simple Moving Averages
                'sma_10': calculate_sma(close, 10).iloc[-1],
                'sma_20': calculate_sma(close, 20).iloc[-1],
                'sma_50': calculate_sma(close, 50).iloc[-1],
                'sma_100': calculate_sma(close, 100).iloc[-1],
                'sma_200': calculate_sma(close, 200).iloc[-1],

                # Exponential Moving Averages
                'ema_10': calculate_ema(close, 10).iloc[-1],
                'ema_20': calculate_ema(close, 20).iloc[-1],
                'ema_50': calculate_ema(close, 50).iloc[-1],

                # RSI
                'rsi_14': calculate_rsi(close, 14).iloc[-1],

                # Volume averages
                'volume_sma_20': calculate_volume_sma(volume, 20).iloc[-1],
                'volume_sma_30': calculate_volume_sma(volume, 30).iloc[-1],
                'volume_sma_50': calculate_volume_sma(volume, 50).iloc[-1],
                'volume_sma_60': calculate_volume_sma(volume, 60).iloc[-1],

                # Percentage changes
                'pct_change_1d': calculate_pct_change(close, 1).iloc[-1],
                'pct_change_5d': calculate_pct_change(close, 5).iloc[-1],
                'pct_change_20d': calculate_pct_change(close, 20).iloc[-1],
                'pct_change_1m': calculate_pct_change(close, 21).iloc[-1],  # ~1 month trading days
                'pct_change_60d': calculate_pct_change(close, 60).iloc[-1],

                # Price relative to MAs (useful for filters)
                'price_vs_sma50_pct': ((latest_close / calculate_sma(close, 50).iloc[-1]) - 1) * 100,
                'price_vs_sma200_pct': ((latest_close / calculate_sma(close, 200).iloc[-1]) - 1) * 100,

                # ATR and ADR for volatility
                'atr_14': calculate_atr(high, low, close, 14).iloc[-1],
                'adr_pct': calculate_adr_pct(high, low, close, 20).iloc[-1],

                # Volatility (1 month)
                'volatility_1m': calculate_volatility(close, 21).iloc[-1],

                # 52-week high/low
                'high_52w': high.tail(252).max() if len(high) >= 252 else high.max(),
                'low_52w': low.tail(252).min() if len(low) >= 252 else low.min(),
            }

            # Add distance from 52w high/low
            row['pct_from_52w_high'] = ((latest_close / row['high_52w']) - 1) * 100
            row['pct_from_52w_low'] = ((latest_close / row['low_52w']) - 1) * 100

            results.append(row)

        except Exception as e:
            print(f"Error computing indicators for {ticker}: {e}")
            continue

    # Create DataFrame
    indicators_df = pd.DataFrame(results).set_index('ticker')

    # Merge with fundamentals if provided
    if fundamentals is not None and not fundamentals.empty:
        indicators_df = indicators_df.join(fundamentals, how='left')

    return indicators_df


def get_indicator_value(df: pd.DataFrame, ticker: str, field: str) -> Optional[float]:
    """Get indicator value for a specific ticker."""
    if ticker not in df.index:
        return None
    if field not in df.columns:
        return None
    return df.loc[ticker, field]


if __name__ == "__main__":
    # Test indicators
    import yfinance as yf

    print("Testing indicators on AAPL...")
    data = yf.download("AAPL", period="1y", progress=False)

    close = data['Close']
    print(f"\nLatest close: {close.iloc[-1]:.2f}")
    print(f"SMA 50: {calculate_sma(close, 50).iloc[-1]:.2f}")
    print(f"SMA 200: {calculate_sma(close, 200).iloc[-1]:.2f}")
    print(f"RSI 14: {calculate_rsi(close, 14).iloc[-1]:.2f}")
    print(f"20-day change: {calculate_pct_change(close, 20).iloc[-1]:.2f}%")
