"""
Screener engine module.
Applies filter definitions to indicator data.
"""

from typing import Dict, List, Any, Set, Optional, Tuple
import pandas as pd
import yaml


# Fields that come from yfinance .info (fundamentals download).
# Everything else is a technical indicator computed from OHLCV.
FUNDAMENTAL_FIELDS = {
    'market_cap', 'pe_ratio', 'forward_pe', 'peg_ratio', 'price_to_book',
    'dividend_yield', 'profit_margin', 'revenue_growth', 'earnings_growth',
    'debt_to_equity', 'current_ratio', 'sector', 'industry',
}


def load_screeners_config(config_path: str = "config/screeners.yaml") -> Dict[str, Any]:
    """Load screeners configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_condition(
    df: pd.DataFrame,
    condition: Dict[str, Any]
) -> pd.Series:
    """
    Evaluate a single filter condition on the DataFrame.

    Args:
        df: DataFrame with indicator columns
        condition: Filter condition dict with field, operator, value/reference

    Returns:
        Boolean Series indicating which rows pass the condition
    """
    field = condition.get('field')
    operator = condition.get('operator')
    value = condition.get('value')
    reference = condition.get('reference')

    if field not in df.columns:
        print(f"Warning: field '{field}' not found in data")
        return pd.Series([False] * len(df), index=df.index)

    field_values = df[field]

    # Determine comparison value
    if reference is not None:
        if reference not in df.columns:
            print(f"Warning: reference field '{reference}' not found in data")
            return pd.Series([False] * len(df), index=df.index)
        compare_to = df[reference]
    else:
        compare_to = value

    # Apply operator
    if operator == '>':
        return field_values > compare_to
    elif operator == '<':
        return field_values < compare_to
    elif operator == '>=':
        return field_values >= compare_to
    elif operator == '<=':
        return field_values <= compare_to
    elif operator == '==':
        return field_values == compare_to
    elif operator == '!=':
        return field_values != compare_to
    elif operator == 'between':
        if not isinstance(value, list) or len(value) != 2:
            print(f"Warning: 'between' operator requires [min, max] value")
            return pd.Series([False] * len(df), index=df.index)
        return (field_values >= value[0]) & (field_values <= value[1])
    else:
        print(f"Warning: unknown operator '{operator}'")
        return pd.Series([False] * len(df), index=df.index)


def apply_screener(
    df: pd.DataFrame,
    screener_def: Dict[str, Any],
    allowed_tickers: Optional[Set[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply a single screener definition to indicator data.

    Args:
        df: DataFrame with all indicators
        screener_def: Screener definition from config
        allowed_tickers: If provided, restrict to only these tickers before filtering

    Returns:
        Tuple of (filtered DataFrame, boolean mask of passing symbols)
    """
    if allowed_tickers is not None:
        df = df.loc[df.index.isin(allowed_tickers)]

    requirements = screener_def.get('requirements', [])

    # Start with all True
    mask = pd.Series([True] * len(df), index=df.index)

    # Apply each requirement
    for req in requirements:
        condition_mask = evaluate_condition(df, req)
        mask = mask & condition_mask

    # Filter DataFrame
    filtered_df = df[mask].copy()

    # Apply postprocess
    postprocess = screener_def.get('postprocess', {})

    if postprocess:
        sort_by = postprocess.get('sort_by')
        sort_order = postprocess.get('sort_order', 'desc')
        top_k = postprocess.get('top_k')

        if sort_by and sort_by in filtered_df.columns:
            ascending = sort_order == 'asc'
            filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending)

        if top_k and len(filtered_df) > top_k:
            filtered_df = filtered_df.head(top_k)

    return filtered_df, mask


def _split_requirements(requirements: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
    """Split requirements into (technical, fundamental) based on field names."""
    tech, fund = [], []
    for req in requirements:
        field = req.get('field', '')
        ref = req.get('reference', '')
        if field in FUNDAMENTAL_FIELDS or ref in FUNDAMENTAL_FIELDS:
            fund.append(req)
        else:
            tech.append(req)
    return tech, fund


def needs_fundamentals(config: Dict[str, Any] = None) -> bool:
    """Return True if any screener uses fundamental fields."""
    if config is None:
        config = load_screeners_config()
    for sdef in config.get('screeners', {}).values():
        _, fund = _split_requirements(sdef.get('requirements', []))
        if fund:
            return True
    return False


def get_prescreen_candidates(
    indicators_df: pd.DataFrame,
    config: Dict[str, Any] = None,
    universe_tickers: Dict[str, Set[str]] = None,
) -> Set[str]:
    """Apply only technical conditions from all screeners to narrow candidates.

    Returns the union of tickers that pass the technical-only pre-screen for
    each screener.  These are the only tickers that need fundamentals.
    """
    if config is None:
        config = load_screeners_config()
    universe_tickers = universe_tickers or {}

    candidates: Set[str] = set()

    for sname, sdef in config.get('screeners', {}).items():
        tech_reqs, fund_reqs = _split_requirements(sdef.get('requirements', []))

        if not fund_reqs:
            # Screener has no fundamental conditions — skip pre-screen,
            # fundamentals aren't needed for this screener at all.
            continue

        # Build a technical-only screener definition
        tech_def = {**sdef, 'requirements': tech_reqs, 'postprocess': {}}
        allowed = universe_tickers.get(sname)
        filtered_df, _ = apply_screener(indicators_df, tech_def, allowed)
        candidates.update(filtered_df.index.tolist())

    return candidates


def run_all_screeners(
    indicators_df: pd.DataFrame,
    config: Dict[str, Any] = None,
    universe_tickers: Dict[str, Set[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Run all screeners and collect results with traceability.

    Args:
        indicators_df: DataFrame with computed indicators
        config: Screeners config (loaded if not provided)
        universe_tickers: Optional mapping of screener_name -> set of allowed
            tickers for that screener. When provided, each screener is restricted
            to its own universe before applying filters.

    Returns:
        Dict with results per screener and traceability info
    """
    if config is None:
        config = load_screeners_config()

    universe_tickers = universe_tickers or {}
    screeners = config.get('screeners', {})
    results = {}

    for screener_name, screener_def in screeners.items():
        allowed = universe_tickers.get(screener_name)
        universe_label = screener_def.get('universe', 'default')

        if allowed is not None:
            print(f"Running screener: {screener_name}  (universe: {universe_label}, {len(allowed)} tickers)")
        else:
            print(f"Running screener: {screener_name}  (universe: all {len(indicators_df)} tickers)")

        filtered_df, mask = apply_screener(indicators_df, screener_def, allowed)

        results[screener_name] = {
            'description': screener_def.get('description', ''),
            'symbols': filtered_df.index.tolist(),
            'count': len(filtered_df),
            'data': filtered_df,
            'mask': mask
        }

        print(f"  -> {len(filtered_df)} symbols passed")

    return results


def consolidate_results(
    screener_results: Dict[str, Dict[str, Any]],
    method: str = 'union',
    exclude_symbols: List[str] = None
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Consolidate results from multiple screeners.

    Args:
        screener_results: Results from run_all_screeners
        method: 'union' or 'intersection'
        exclude_symbols: Symbols to exclude from final results

    Returns:
        Tuple of (final symbol list, traceability dict showing which screeners each symbol passed)
    """
    exclude_symbols = exclude_symbols or []

    # Build traceability: symbol -> list of screeners that passed
    traceability: Dict[str, List[str]] = {}

    for screener_name, result in screener_results.items():
        for symbol in result['symbols']:
            if symbol not in traceability:
                traceability[symbol] = []
            traceability[symbol].append(screener_name)

    # Get all symbols
    all_symbols = set(traceability.keys())

    if method == 'union':
        final_symbols = all_symbols
    elif method == 'intersection':
        # Only symbols that passed ALL screeners
        num_screeners = len(screener_results)
        final_symbols = {s for s, screeners in traceability.items()
                        if len(screeners) == num_screeners}
    else:
        print(f"Unknown consolidation method: {method}, using union")
        final_symbols = all_symbols

    # Apply exclusions
    final_symbols = final_symbols - set(exclude_symbols)

    # Sort alphabetically
    final_list = sorted(list(final_symbols))

    # Filter traceability to only include final symbols
    final_traceability = {s: screeners for s, screeners in traceability.items()
                         if s in final_symbols}

    return final_list, final_traceability


def build_report_dataframe(
    indicators_df: pd.DataFrame,
    final_symbols: List[str],
    traceability: Dict[str, List[str]],
    symbol_to_exchange: Dict[str, str] = None
) -> pd.DataFrame:
    """
    Build a report DataFrame for auditing.

    Args:
        indicators_df: Full indicators DataFrame
        final_symbols: List of symbols that passed screening
        traceability: Dict mapping symbol -> screeners passed
        symbol_to_exchange: Optional mapping of ticker -> exchange

    Returns:
        DataFrame with audit information
    """
    symbol_to_exchange = symbol_to_exchange or {}

    report_rows = []

    for symbol in final_symbols:
        if symbol not in indicators_df.index:
            continue

        row_data = indicators_df.loc[symbol].to_dict()
        row_data['ticker'] = symbol
        row_data['exchange'] = symbol_to_exchange.get(symbol, 'UNKNOWN')
        row_data['screeners_passed'] = ','.join(traceability.get(symbol, []))
        row_data['num_screeners_passed'] = len(traceability.get(symbol, []))

        report_rows.append(row_data)

    report_df = pd.DataFrame(report_rows)

    # Reorder columns to put key info first
    key_cols = ['ticker', 'exchange', 'screeners_passed', 'num_screeners_passed',
                'close', 'volume', 'rsi_14', 'pct_change_20d']
    other_cols = [c for c in report_df.columns if c not in key_cols]
    ordered_cols = [c for c in key_cols if c in report_df.columns] + other_cols

    return report_df[ordered_cols]


if __name__ == "__main__":
    # Test with mock data
    print("Testing screener engine...")

    # Create mock indicator data
    mock_data = pd.DataFrame({
        'close': [150, 25, 100, 8, 200],
        'volume_sma_20': [1000000, 300000, 500000, 100000, 800000],
        'sma_50': [140, 30, 95, 10, 190],
        'sma_200': [130, 35, 90, 12, 180],
        'rsi_14': [55, 35, 60, 28, 70],
        'pct_change_20d': [5, -10, 8, -5, 15],
        'pe_ratio': [25, 15, 18, 10, 30],
    }, index=['AAPL', 'XYZ', 'MSFT', 'PENNY', 'GOOGL'])

    config = load_screeners_config()
    results = run_all_screeners(mock_data, config)

    for name, result in results.items():
        print(f"\n{name}: {result['symbols']}")

    final, trace = consolidate_results(results, method='union')
    print(f"\nFinal consolidated list: {final}")
    print(f"Traceability: {trace}")
