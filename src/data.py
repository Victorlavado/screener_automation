"""
Data download module.
Handles fetching OHLCV and fundamental data from yfinance.
"""

import logging
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

logger = logging.getLogger(__name__)

# Cache directories
OHLCV_CACHE_DIR = Path(__file__).parent.parent / ".cache" / "ohlcv"
FUNDAMENTALS_CACHE_DIR = Path(__file__).parent.parent / ".cache" / "fundamentals"


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


def _is_rate_limit_error(error: Exception) -> bool:
    """Check if an exception indicates a Yahoo Finance rate limit."""
    msg = str(error).lower()
    return any(term in msg for term in (
        "ratelimit", "rate limit", "too many requests", "429",
        "yfrateerror", "yfpricemissing",
    ))


def download_ohlcv_batch(
    tickers: List[str],
    period: str = "1y",
    interval: str = "1d",
    batch_size: int = 200,
    delay_between_batches: float = 2.0,
    show_progress: bool = True,
    cache_max_age_hours: float = 16.0,
    max_retries: int = 3,
) -> Dict[str, pd.DataFrame]:
    """
    Download OHLCV data for multiple tickers in batches, with persistent cache.

    Tickers with a fresh cache file (< cache_max_age_hours old) are loaded
    from disk; only the remaining tickers are downloaded from Yahoo Finance.

    Args:
        tickers: List of ticker symbols (without exchange prefix)
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
        interval: Data interval (1d, 1wk, 1mo)
        batch_size: Number of tickers per batch request
        delay_between_batches: Seconds to wait between batches
        show_progress: Show progress bar
        cache_max_age_hours: Max age of cache in hours (0 = disable cache)
        max_retries: Max retry attempts per batch on failure

    Returns:
        Dict mapping ticker -> DataFrame with OHLCV data
    """
    results = {}
    cached_count = 0

    # Load from cache
    if cache_max_age_hours > 0:
        results = _load_ohlcv_cache(tickers, period, interval, cache_max_age_hours)
        cached_count = len(results)
        if results:
            logger.info("Loaded %d tickers from OHLCV cache", cached_count)

    # Determine which tickers still need downloading
    remaining = [t for t in tickers if t not in results]

    if not remaining:
        logger.info("All tickers served from cache, no downloads needed")
        return results

    logger.info("Downloading %d tickers (%d cached)", len(remaining), cached_count)

    # Split into batches
    batches = [remaining[i:i + batch_size] for i in range(0, len(remaining), batch_size)]

    iterator = tqdm(batches, desc="Downloading OHLCV") if show_progress else batches

    fresh = {}
    total_failed = []

    for batch in iterator:
        batch_success = False

        for attempt in range(max_retries + 1):
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

                batch_results = {}
                if len(batch) == 1:
                    ticker = batch[0]
                    if not data.empty:
                        batch_results[ticker] = data
                else:
                    for ticker in batch:
                        if ticker in data.columns.get_level_values(0):
                            ticker_data = data[ticker].dropna(how='all')
                            if not ticker_data.empty:
                                batch_results[ticker] = ticker_data

                # Check if we got significantly fewer results than expected
                # (> 50% missing suggests rate limiting)
                expected = len(batch)
                got = len(batch_results)
                if got < expected * 0.5 and attempt < max_retries:
                    wait = 30 * (2 ** attempt)
                    logger.warning(
                        "Batch returned %d/%d tickers — possible rate limit. "
                        "Retrying in %ds (attempt %d/%d)",
                        got, expected, wait, attempt + 1, max_retries,
                    )
                    time.sleep(wait)
                    continue

                fresh.update(batch_results)
                batch_success = True
                break

            except Exception as e:
                if _is_rate_limit_error(e) and attempt < max_retries:
                    wait = 30 * (2 ** attempt)
                    logger.warning(
                        "Rate limit error on batch: %s. "
                        "Retrying in %ds (attempt %d/%d)",
                        e, wait, attempt + 1, max_retries,
                    )
                    time.sleep(wait)
                elif attempt < max_retries:
                    wait = 5 * (2 ** attempt)
                    logger.warning(
                        "Error downloading batch: %s. "
                        "Retrying in %ds (attempt %d/%d)",
                        e, wait, attempt + 1, max_retries,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "Batch failed after %d retries: %s. "
                        "Failed tickers: %s",
                        max_retries, e,
                        ", ".join(batch[:10]) + ("..." if len(batch) > 10 else ""),
                    )
                    total_failed.extend(batch)
                    batch_success = True  # Mark as handled to avoid double-logging
                    break

        if not batch_success:
            logger.error(
                "Batch exhausted %d retries. Failed tickers: %s",
                max_retries,
                ", ".join(batch[:10]) + ("..." if len(batch) > 10 else ""),
            )
            total_failed.extend(batch)

        if delay_between_batches > 0:
            time.sleep(delay_between_batches)

    # Persist newly downloaded data
    if fresh and cache_max_age_hours > 0:
        _save_ohlcv_cache(fresh, period, interval)

    results.update(fresh)

    # Summary log
    success_count = len(results)
    failed_count = len(tickers) - success_count
    logger.info(
        "OHLCV download complete: %d/%d tickers (%d failed, %d from cache)",
        success_count, len(tickers), failed_count, cached_count,
    )

    return results


# ── Fundamentals cache ──────────────────────────────────────────────────────

def _get_fundamentals_cache_dir() -> Path:
    """Get (and create) the fundamentals cache directory."""
    FUNDAMENTALS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return FUNDAMENTALS_CACHE_DIR


def _load_fundamentals_cache(
    tickers: List[str],
    max_age_days: float = 7.0,
) -> Dict[str, Dict]:
    """Load cached fundamentals for tickers with a fresh cache file."""
    cache_dir = _get_fundamentals_cache_dir()
    cached = {}
    now = time.time()
    max_age_seconds = max_age_days * 86400

    for ticker in tickers:
        path = cache_dir / f"{_safe_filename(ticker)}.parquet"
        if path.exists() and (now - path.stat().st_mtime) < max_age_seconds:
            try:
                df = pd.read_parquet(path)
                if not df.empty:
                    cached[ticker] = df.iloc[0].to_dict()
            except Exception:
                pass  # Corrupted cache, will re-download

    return cached


def _save_fundamentals_cache(fundamentals: List[Dict]) -> None:
    """Persist fundamentals data as individual parquet files."""
    cache_dir = _get_fundamentals_cache_dir()

    for entry in fundamentals:
        ticker = entry.get('ticker')
        if not ticker:
            continue
        path = cache_dir / f"{_safe_filename(ticker)}.parquet"
        try:
            df = pd.DataFrame([entry])
            df.to_parquet(path)
        except Exception:
            pass  # Non-critical


def _fetch_single_fundamental(ticker: str, max_retries: int = 2, retry_delay: float = 5.0) -> Optional[Dict]:
    """Fetch fundamental data for a single ticker with retry logic.

    Returns None on failure after all retries exhausted.
    """
    for attempt in range(max_retries + 1):
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
        except Exception as e:
            if attempt < max_retries:
                logger.debug(
                    "Fundamentals fetch failed for %s: %s — retrying in %.0fs (%d/%d)",
                    ticker, e, retry_delay, attempt + 1, max_retries,
                )
                time.sleep(retry_delay)
            else:
                logger.warning("Fundamentals fetch failed for %s after %d retries: %s", ticker, max_retries, e)
                return None


def download_fundamentals(
    tickers: List[str],
    show_progress: bool = True,
    max_workers: int = 4,
    cache_max_age_days: float = 7.0,
) -> pd.DataFrame:
    """
    Download fundamental data for multiple tickers using parallel threads.

    Args:
        tickers: List of ticker symbols
        show_progress: Show progress bar
        max_workers: Number of concurrent download threads
        cache_max_age_days: Max age of fundamentals cache in days (0 = disable)

    Returns:
        DataFrame with fundamental metrics for each ticker
    """
    fundamentals = []
    cached_count = 0

    # Load from cache
    if cache_max_age_days > 0:
        cached = _load_fundamentals_cache(tickers, cache_max_age_days)
        cached_count = len(cached)
        if cached:
            fundamentals.extend(cached.values())
            logger.info("Loaded %d tickers from fundamentals cache", cached_count)

    # Determine which tickers still need downloading
    cached_tickers = {f.get('ticker') for f in fundamentals if f}
    remaining = [t for t in tickers if t not in cached_tickers]

    if not remaining:
        logger.info("All fundamentals served from cache, no downloads needed")
        return pd.DataFrame(fundamentals).set_index('ticker') if fundamentals else pd.DataFrame()

    logger.info("Downloading fundamentals for %d tickers (%d cached)", len(remaining), cached_count)

    fresh = []
    failed_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for ticker in remaining:
            future = executor.submit(_fetch_single_fundamental, ticker)
            futures[future] = ticker
            # Small delay between submissions to avoid burst requests
            time.sleep(0.5)

        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(futures), desc="Downloading fundamentals")

        for future in iterator:
            result = future.result()
            if result is not None:
                fresh.append(result)
            else:
                failed_count += 1

    # Cache newly downloaded fundamentals
    if fresh and cache_max_age_days > 0:
        _save_fundamentals_cache(fresh)

    fundamentals.extend(fresh)

    # Summary log
    total = len(tickers)
    success = len(fundamentals)
    logger.info(
        "Fundamentals download complete: %d/%d tickers (%d failed, %d from cache)",
        success, total, failed_count, cached_count,
    )

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
    logging.basicConfig(level=logging.INFO)
    test_tickers = ["AAPL", "MSFT", "GOOGL"]

    logger.info("Testing OHLCV download...")
    ohlcv = download_ohlcv_batch(test_tickers, period="1mo")
    logger.info("Downloaded data for %d tickers", len(ohlcv))

    for ticker, df in ohlcv.items():
        logger.info("%s: %d rows", ticker, len(df))
        print(df.tail(3))
