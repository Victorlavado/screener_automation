"""
Data download module.
Handles fetching OHLCV and fundamental data from yfinance.
"""

import time
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import yfinance as yf
from tqdm import tqdm

from .universe import symbols_to_tickers


# Cache directory
OHLCV_CACHE_DIR = Path(__file__).parent.parent / ".cache" / "ohlcv"


def _safe_filename(ticker: str) -> str:
    """Convert ticker to safe filename (e.g. 600519.SS -> 600519_SS)."""
    return re.sub(r'[^\w\-]', '_', ticker)


def _get_ohlcv_cache_dir(period: str, interval: str) -> Path:
    """Get cache directory for a specific period/interval combo."""
    cache_dir = OHLCV_CACHE_DIR / f"{period}_{interval}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _load_ohlcv_cache(
    tickers: List[str],
    period: str,
    interval: str,
    max_age_hours: float = 16.0,
) -> Dict[str, pd.DataFrame]:
    """Load cached OHLCV data for tickers that have a fresh cache file."""
    cache_dir = _get_ohlcv_cache_dir(period, interval)
    cached = {}
    now = time.time()
    max_age_seconds = max_age_hours * 3600

    for ticker in tickers:
        path = cache_dir / f"{_safe_filename(ticker)}.parquet"
        if path.exists() and (now - path.stat().st_mtime) < max_age_seconds:
            try:
                cached[ticker] = pd.read_parquet(path)
            except Exception:
                pass  # Corrupted cache, will re-download

    return cached


def _save_ohlcv_cache(
    data: Dict[str, pd.DataFrame],
    period: str,
    interval: str,
) -> None:
    """Persist downloaded OHLCV DataFrames to parquet cache."""
    cache_dir = _get_ohlcv_cache_dir(period, interval)

    for ticker, df in data.items():
        if not df.empty:
            path = cache_dir / f"{_safe_filename(ticker)}.parquet"
            try:
                df.to_parquet(path)
            except Exception:
                pass  # Non-critical, skip silently


def download_ohlcv_batch(
    tickers: List[str],
    period: str = "1y",
    interval: str = "1d",
    batch_size: int = 500,
    delay_between_batches: float = 0.5,
    show_progress: bool = True,
    cache_max_age_hours: float = 16.0,
) -> Dict[str, pd.DataFrame]:
    """
    Download OHLCV data for multiple tickers in batches, with persistent cache.

    Tickers with a fresh cache file (< cache_max_age_hours old) are loaded
    from disk; only the remaining tickers are downloaded from Yahoo Finance.

    Args:
        tickers: List of ticker symbols (without exchange prefix)
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
        interval: Data interval (1d, 1wk, 1mo)
        batch_size: Number of tickers per batch request (yfinance handles up to ~500)
        delay_between_batches: Seconds to wait between batches
        show_progress: Show progress bar
        cache_max_age_hours: Max age of cache in hours (0 = disable cache)

    Returns:
        Dict mapping ticker -> DataFrame with OHLCV data
    """
    results = {}

    # Load from cache
    if cache_max_age_hours > 0:
        results = _load_ohlcv_cache(tickers, period, interval, cache_max_age_hours)
        if results and show_progress:
            print(f"Loaded {len(results)} tickers from cache")

    # Determine which tickers still need downloading
    remaining = [t for t in tickers if t not in results]

    if not remaining:
        if show_progress:
            print("All tickers served from cache, no downloads needed")
        return results

    if show_progress:
        print(f"Downloading {len(remaining)} tickers ({len(results)} cached)")

    # Split into batches
    batches = [remaining[i:i + batch_size] for i in range(0, len(remaining), batch_size)]

    iterator = tqdm(batches, desc="Downloading OHLCV") if show_progress else batches

    fresh = {}
    for batch in iterator:
        try:
            batch_str = " ".join(batch)
            data = yf.download(
                batch_str,
                period=period,
                interval=interval,
                group_by='ticker',
                auto_adjust=True,
                progress=False,
                threads=True
            )

            if len(batch) == 1:
                ticker = batch[0]
                if not data.empty:
                    fresh[ticker] = data
            else:
                for ticker in batch:
                    if ticker in data.columns.get_level_values(0):
                        ticker_data = data[ticker].dropna(how='all')
                        if not ticker_data.empty:
                            fresh[ticker] = ticker_data

            if delay_between_batches > 0:
                time.sleep(delay_between_batches)

        except Exception as e:
            print(f"Error downloading batch: {e}")
            continue

    # Persist newly downloaded data
    if fresh and cache_max_age_hours > 0:
        _save_ohlcv_cache(fresh, period, interval)

    results.update(fresh)
    return results


def _fetch_single_fundamental(ticker: str) -> Optional[Dict]:
    """Fetch fundamental data for a single ticker. Returns None on failure."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'ticker': ticker,
            'market_cap': info.get('marketCap'),
            'pe_ratio': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'peg_ratio': info.get('pegRatio'),
            'price_to_book': info.get('priceToBook'),
            'dividend_yield': info.get('dividendYield'),
            'profit_margin': info.get('profitMargins'),
            'revenue_growth': info.get('revenueGrowth'),
            'earnings_growth': info.get('earningsGrowth'),
            'debt_to_equity': info.get('debtToEquity'),
            'current_ratio': info.get('currentRatio'),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
        }
    except Exception:
        return None


def download_fundamentals(
    tickers: List[str],
    show_progress: bool = True,
    max_workers: int = 8
) -> pd.DataFrame:
    """
    Download fundamental data for multiple tickers using parallel threads.

    Args:
        tickers: List of ticker symbols
        show_progress: Show progress bar
        max_workers: Number of concurrent download threads

    Returns:
        DataFrame with fundamental metrics for each ticker
    """
    fundamentals = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_single_fundamental, ticker): ticker
            for ticker in tickers
        }

        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(futures), desc="Downloading fundamentals")

        for future in iterator:
            result = future.result()
            if result is not None:
                fundamentals.append(result)

    return pd.DataFrame(fundamentals).set_index('ticker') if fundamentals else pd.DataFrame()


def download_all_data(
    symbols: List[str],
    period: str = "1y",
    include_fundamentals: bool = True,
    show_progress: bool = True
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Download all required data for screening.

    Args:
        symbols: List of symbols in EXCHANGE:SYMBOL format
        period: OHLCV data period
        include_fundamentals: Whether to download fundamental data
        show_progress: Show progress bars

    Returns:
        Tuple of (ohlcv_dict, fundamentals_df)
    """
    # Convert to tickers (remove exchange prefix)
    tickers = symbols_to_tickers(symbols)

    # Download OHLCV
    ohlcv = download_ohlcv_batch(tickers, period=period, show_progress=show_progress)

    # Download fundamentals
    fundamentals = pd.DataFrame()
    if include_fundamentals:
        fundamentals = download_fundamentals(list(ohlcv.keys()), show_progress=show_progress)

    return ohlcv, fundamentals


def get_latest_prices(ohlcv: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Extract latest prices from OHLCV data.

    Returns:
        DataFrame with latest OHLCV for each ticker
    """
    latest = []

    for ticker, df in ohlcv.items():
        if df.empty:
            continue

        last_row = df.iloc[-1]
        latest.append({
            'ticker': ticker,
            'close': last_row.get('Close'),
            'open': last_row.get('Open'),
            'high': last_row.get('High'),
            'low': last_row.get('Low'),
            'volume': last_row.get('Volume'),
            'date': df.index[-1]
        })

    return pd.DataFrame(latest).set_index('ticker') if latest else pd.DataFrame()


if __name__ == "__main__":
    # Test data download
    test_tickers = ["AAPL", "MSFT", "GOOGL"]

    print("Testing OHLCV download...")
    ohlcv = download_ohlcv_batch(test_tickers, period="1mo")
    print(f"Downloaded data for {len(ohlcv)} tickers")

    for ticker, df in ohlcv.items():
        print(f"\n{ticker}: {len(df)} rows")
        print(df.tail(3))
