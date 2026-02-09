"""
Technical indicators calculation module.
Computes various indicators from OHLCV data.
"""

import logging
import os
from typing import Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


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


MIN_ROWS = 20  # Minimum rows to compute any indicator at all


def _safe_last(series: pd.Series, min_len: int) -> Optional[float]:
    """Return last value of series if it has enough non-NaN data, else NaN."""
    if len(series) < min_len:
        return np.nan
    val = series.iloc[-1]
    return val if pd.notna(val) else np.nan


def _compute_single_ticker(args: tuple) -> Optional[dict]:
    """Compute all indicators for one ticker. Module-level for pickling.

    Tickers with fewer than MIN_ROWS rows are skipped entirely.
    Indicators requiring more data than available are set to NaN.
    """
    ticker, df = args

    if df.empty or len(df) < MIN_ROWS:
        return None

    try:
        n = len(df)
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        latest_close = close.iloc[-1]

        # SMAs — compute only if enough rows, otherwise NaN
        sma_10_val = _safe_last(calculate_sma(close, 10), 10)
        sma_20_val = _safe_last(calculate_sma(close, 20), 20)
        sma_50_val = _safe_last(calculate_sma(close, 50), 50)
        sma_100_val = _safe_last(calculate_sma(close, 100), 100)
        sma_200_val = _safe_last(calculate_sma(close, 200), 200)

        # EMAs
        ema_10_val = _safe_last(calculate_ema(close, 10), 10)
        ema_20_val = _safe_last(calculate_ema(close, 20), 20)
        ema_50_val = _safe_last(calculate_ema(close, 50), 50)

        # Price vs SMA — only if SMA itself is valid
        price_vs_sma50 = ((latest_close / sma_50_val) - 1) * 100 if pd.notna(sma_50_val) and sma_50_val != 0 else np.nan
        price_vs_sma200 = ((latest_close / sma_200_val) - 1) * 100 if pd.notna(sma_200_val) and sma_200_val != 0 else np.nan

        # 52-week high/low (use all available data if <252 rows)
        high_52w = high.tail(252).max() if n >= 252 else high.max()
        low_52w = low.tail(252).min() if n >= 252 else low.min()

        return {
            'ticker': ticker,
            'close': latest_close,
            'open': df['Open'].iloc[-1],
            'high': high.iloc[-1],
            'low': low.iloc[-1],
            'volume': volume.iloc[-1],

            'sma_10': sma_10_val,
            'sma_20': sma_20_val,
            'sma_50': sma_50_val,
            'sma_100': sma_100_val,
            'sma_200': sma_200_val,

            'ema_10': ema_10_val,
            'ema_20': ema_20_val,
            'ema_50': ema_50_val,

            'rsi_14': _safe_last(calculate_rsi(close, 14), 15),

            'volume_sma_20': _safe_last(calculate_volume_sma(volume, 20), 20),
            'volume_sma_30': _safe_last(calculate_volume_sma(volume, 30), 30),
            'volume_sma_50': _safe_last(calculate_volume_sma(volume, 50), 50),
            'volume_sma_60': _safe_last(calculate_volume_sma(volume, 60), 60),

            'pct_change_1d': _safe_last(calculate_pct_change(close, 1), 2),
            'pct_change_5d': _safe_last(calculate_pct_change(close, 5), 6),
            'pct_change_20d': _safe_last(calculate_pct_change(close, 20), 21),
            'pct_change_1m': _safe_last(calculate_pct_change(close, 21), 22),
            'pct_change_60d': _safe_last(calculate_pct_change(close, 60), 61),

            'price_vs_sma50_pct': price_vs_sma50,
            'price_vs_sma200_pct': price_vs_sma200,

            'atr_14': _safe_last(calculate_atr(high, low, close, 14), 15),
            'adr_pct': _safe_last(calculate_adr_pct(high, low, close, 20), 21),

            'volatility_1m': _safe_last(calculate_volatility(close, 21), 22),

            'high_52w': high_52w,
            'low_52w': low_52w,
            'pct_from_52w_high': ((latest_close / high_52w) - 1) * 100,
            'pct_from_52w_low': ((latest_close / low_52w) - 1) * 100,
        }
    except Exception:
        return None


def compute_all_indicators(
    ohlcv: Dict[str, pd.DataFrame],
    fundamentals: Optional[pd.DataFrame] = None,
    max_workers: int = None,
) -> pd.DataFrame:
    """
    Compute all indicators for screening using parallel processes.

    Args:
        ohlcv: Dict mapping ticker -> OHLCV DataFrame
        fundamentals: DataFrame with fundamental data (optional)
        max_workers: Number of parallel processes (default: CPU count)

    Returns:
        DataFrame with one row per ticker and all computed indicators
    """
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 8)

    items = list(ohlcv.items())

    # Log tickers that will be skipped (< MIN_ROWS)
    short_tickers = [t for t, df in items if not df.empty and len(df) < MIN_ROWS]
    empty_tickers = [t for t, df in items if df.empty]
    if empty_tickers:
        logger.warning(f"Skipping {len(empty_tickers)} tickers with empty data")
    if short_tickers:
        logger.warning(
            f"Skipping {len(short_tickers)} tickers with <{MIN_ROWS} rows "
            f"(e.g. {short_tickers[:5]})"
        )

    # Multiprocessing has ~1-2s spawn overhead (esp. Windows).
    # Only worth it above ~200 tickers where computation dominates.
    if len(items) < 200:
        results = [_compute_single_ticker(item) for item in items]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_compute_single_ticker, items, chunksize=64))

    failed = sum(1 for r in results if r is None) - len(short_tickers) - len(empty_tickers)
    results = [r for r in results if r is not None]

    if failed > 0:
        logger.warning(f"{failed} tickers failed during indicator computation")

    # Count tickers with partial indicators (have NaN in SMA-200)
    partial = sum(1 for r in results if pd.isna(r.get('sma_200')))
    if partial:
        logger.info(f"{partial} tickers have partial indicators (insufficient history for all)")

    indicators_df = pd.DataFrame(results).set_index('ticker')

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
