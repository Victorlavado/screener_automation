---
title: Relax post_filter EMA5 and add Relative Strength screeners
type: feat
status: completed
date: 2026-04-27
origin: docs/brainstorms/2026-04-24-relax-postfilter-and-add-rs-screeners-brainstorm.md
---

# Relax post_filter EMA5 and add Relative Strength screeners

## Enhancement Summary

**Deepened on:** 2026-04-27
**Technical review applied:** 2026-04-27 (second pass on the deepened plan)
**Sections enhanced:** Cross-cutting computation order, Implementation §2-§5, Spec Flow, Acceptance Criteria, expanded External References
**Agents used:** kieran-python-reviewer (×2), performance-oracle, code-simplicity-reviewer (×2), architecture-strategist, best-practices-researcher (quant finance), framework-docs-researcher (pandas/numpy/yfinance)

### Key Improvements

1. **Vectorized IR computation** — replaced the per-ticker `align()` loop with a wide-DataFrame broadcast subtract. Estimated runtime drops from ~8-10s to ~1-2s on the `us_eu` universe (~6,880 tickers); code is no more complex.
2. **Dedicated checkpoint stage `indicators_rs`** — RS enrichment gets its own checkpoint marker so the existing `indicators` marker keeps a stable schema contract across runs. Resume after the schema change re-runs RS enrichment instead of silently producing empty bypass results.
3. **Renamed `rs_ibd_rating` → `rs_oneil_modified`** — the canonical IBD/O'Neil RS Rating uses 40/20/20/20 weights over 3M/6M/9M/12M; this plan uses 40/30/20/10 over 1M/3M/6M/12M (front-loaded for post-2010 momentum compression). The new name flags the deviation honestly. Filter and threshold unchanged.
4. **Correctness fix in IR alignment** — align the full stock-returns series with SP500 *first*, then take the trailing 63d. Prevents EU stocks from systematically losing days near US holidays.
5. **`split_screener_config` returns a `ScreenerGroups` NamedTuple** instead of a 3-tuple. Same unpacking ergonomics, no positional-ordering hazard.
6. **Engine-startup schema check** — `run_all_screeners` validates that every YAML-referenced field exists in `indicators_df` before running. Fail-fast at load instead of silent empty results from `evaluate_condition`'s soft missing-column behavior.
7. **`^GSPC` exempted from negative cache** — `data.py` skips writing benchmark tickers to the failed-tickers cache so a transient outage doesn't 7-day-poison the SP500 download.

### New Considerations Discovered

- **IBD-style weights are not citable** as O'Neil's published formula. Renaming + documenting the deviation is the honest move; calibration belongs in Open Questions.
- **Information Ratio at 63d is a *signal*, not a manager-skill measurement** (Goodwin, SSGA). The threshold IR ≥ 1.0 is empirically reasonable for *single-stock screening* even though institutional benchmarks cluster near 0.5.
- **pandas 3.0 (Jan 2026) makes Copy-on-Write the default** and removes `SettingWithCopyWarning`. The plan's `df.copy()` pattern remains the textbook safe form.
- **yfinance occasionally returns empty DataFrames for `^GSPC`** during transient outages. The negative-cache exemption is a real correctness concern, not a hypothetical.

---

## Overview

Two complementary changes to the screening engine:

1. **Relax the existing EMA5 post-filter ranges** so the funnel admits a slightly wider pullback band, without altering the regular → post-filter architecture.
2. **Add two new Relative Strength screeners (`rs_information_ratio`, `rs_oneil_modified`)** that run on the `us_eu` universe against a single SP500 benchmark and **bypass the EMA5 post-filter** by means of a new per-screener flag `apply_post_filter: false`.

The combined effect is a final output that captures stocks the current pipeline rejects at the EMA5 bottleneck (consolidating leaders, recent breakouts, low-pullback names), while preserving the EMA5 confirmation for the screeners that need it (CANSLIM, momentum 1M/3M/6M, etc.) (see brainstorm: `docs/brainstorms/2026-04-24-relax-postfilter-and-add-rs-screeners-brainstorm.md`).

## Problem Statement / Motivation

The post_filter (EMA5 scans) currently funnels every regular-screener output through a strict pullback gate (`ema5_distance_pct ∈ [-5, 0] ∪ [0, 5]`, `pct_change_5d < 5`). This is the right gate for momentum entries but excludes two valid setups: (a) names just outside the band that would be admissible with a slightly wider window, and (b) consistent outperformers vs SP500 that don't satisfy a pullback at all. The brainstorm rejected three architectural alternatives (informative tag, dual output, eliminate the concept) in favor of:

- Surgical YAML relaxation for case (a) — no architecture touched.
- Per-screener bypass flag for case (b) — minimal architecture: one boolean.

(see brainstorm: `docs/brainstorms/2026-04-24-relax-postfilter-and-add-rs-screeners-brainstorm.md` § "Why This Approach")

## Proposed Solution

### Theme A — EMA5 post-filter relaxation (YAML-only)

Widen the pullback band ~50% on both sides; widen the 5-day change ceiling to keep the relaxation coherent. `sma_10 > sma_20` and `volatility_1m >= 3.5` stay (they encode the trend and minimum movement we still require).

| Knob | Before | After |
|---|---|---|
| `scan_below_ema5` `ema5_distance_pct` | `[0, 5]` | `[0, 7.5]` |
| `scan_above_ema5` `ema5_distance_pct` | `[-5, 0]` | `[-7.5, 0]` |
| `pct_change_5d` (both) | `< 5` | `< 7` |

### Theme B — RS screeners + bypass flag

**New flag `apply_post_filter: bool` (default `true`)** at the per-screener level in `config/screeners.yaml`. Default `true` preserves the current behavior of every existing regular screener. `false` routes a screener's output directly to the final union, skipping the EMA5 post-filter.

**Two new regular screeners** (`universe: us_eu`, `apply_post_filter: false`):

- `rs_information_ratio` — annualized excess-return-over-volatility vs SP500 over a 63d window. Filter: `IR ≥ 1.0`. Sort: `IR desc`. Formula is canonical Grinold/Kahn active-management IR (validated; see Research Insights §1).
- `rs_oneil_modified` — multi-horizon weighted return percentile-ranked across the universe (1-100 scale). Filter: `rs_rating ≥ 80 AND volatility_1m ≤ 80`. Sort: `rs_rating desc`. Note: weights `0.4·R1M + 0.3·R3M + 0.2·R6M + 0.1·R12M` deliberately deviate from the canonical O'Neil 40/20/20/20 over 3M/6M/9M/12M. The `_modified` suffix flags this; calibration is the first thing to revisit in Open Questions.

Both bypass the EMA5 post-filter so consistent outperformers reach the watchlist without requiring a pullback.

### Theme C — Three-group pipeline flow (run_weekly.py)

Replace the current two-group split (`split_screener_config`) with a three-group split returned as a `ScreenerGroups` NamedTuple:

1. **Group A** — regular screeners with `apply_post_filter: true` (default).
2. **Group B** — regular screeners with `apply_post_filter: false` (the new RS ones).
3. **Group C** — post-filter screeners (`post_filter: true`, the EMA5 scans).

Flow:
```
Group A → consolidate → Group C (EMA5 post-filter) → symbols_via_postfilter
Group B → consolidate                              → symbols_bypass
final = symbols_via_postfilter ∪ symbols_bypass
traceability merges A-tags + C-tags + B-tags per symbol
```

## Technical Considerations

### Cross-cutting computation order

The two RS screeners need indicator columns that **cannot be computed inside `_compute_single_ticker`** (the multiprocessing worker), because:

- IR needs SP500 returns aligned by date — a shared array across all workers.
- `rs_rating` is a percentile rank across the entire universe — only computable once the per-ticker pass is complete.

Solution: a new module-level function `compute_rs_indicators(indicators_df, sp500_close, ohlcv) -> pd.DataFrame` invoked **after** `compute_all_indicators` returns, as its **own pipeline stage** with its own checkpoint marker (`indicators_rs`). It enriches `indicators_df` with three new columns:

- `rs_ir` (float) — annualized Information Ratio over 63d.
- `rs_weighted_return` (float) — `0.4·R1M + 0.3·R3M + 0.2·R6M + 0.1·R12M` (percentage points).
- `rs_rating` (float) — `rank(rs_weighted_return, pct=True) * 100` over the full DataFrame.

The screener engine (`evaluate_condition`) only does column lookups, so once these are columns the YAML filters work without engine changes — no virtual operators needed.

**Performance note:** the IR is computed via a vectorized broadcast subtract on a wide DataFrame (`returns_wide.sub(sp_aligned, axis=0)`), not a per-ticker loop. This brings the enrichment to ~1-2s on the `us_eu` universe (validated in Research Insights §2). The IBD percentile rank is already vectorized via `df.rank(pct=True)`.

### SP500 (`^GSPC`) acquisition

`^GSPC` is a yfinance ticker. `download_ohlcv_batch` already handles arbitrary tickers and caches via `_safe_filename` (round-trip safe; the `_GSPC.parquet` file is read by the same function that wrote it). The download is a separate single-ticker call in stage 2 of the pipeline; it is not added to the universe, so it never reaches `compute_all_indicators` as a row. Failure of the SP500 download is tolerated: log a warning and skip the RS enrichment so the rest of the pipeline still produces its output.

**Negative-cache exemption.** `data.py` does **not** write `^GSPC` to the failed-tickers cache. Yahoo occasionally returns empty for `^GSPC` during transient outages; a 7-day cooldown on the benchmark would degrade RS computation across multiple weekly runs for no good reason. Implementation: an inline literal check in `_save_failed_tickers_cache` (one line: skip when `ticker == "^GSPC"`). Promote to a `BENCHMARK_TICKERS` set only when a second benchmark is actually added.

### Date alignment (US vs EU)

European stocks have ~252 trading days but on a different calendar than NYSE (e.g., May 1st is closed in Paris, open in NY; July 4th vice versa). Align stock daily returns and SP500 daily returns on the **intersection of dates** (`Series.align(join='inner')` — canonical pandas idiom; see Research Insights §3). Take the trailing 63d window **after** alignment, not before — otherwise EU stocks systematically lose days near US holidays inside the 63-row window. Require ≥ 50 valid days; otherwise emit `NaN` for `rs_ir` (the YAML filter `>= 1.0` rejects NaN, so it falls out naturally).

### Volatility cap unit-check

`volatility_1m` in `src/indicators.py:79-86` is computed as `std(daily_returns) * sqrt(252) * 100`, i.e., a percentage (e.g., 25 for 25% annualized). The brainstorm's `volatility_1m <= 80` therefore compares directly without rescaling. In practice this excludes clinical-stage biotechs, micro-cap miners, leveraged single-stock ETFs, and post-IPO/post-meme distressed names; mid-cap biotechs (40–60% vol) and high-momentum tech (50–70%) pass.

### Backward-compatibility via flag default

`apply_post_filter` defaults to `true`. Every existing screener config (CANSLIM, alternative_canslim, strongest_*m_*, new_rs) continues to flow through the EMA5 post-filter exactly as today. The only screeners declaring `apply_post_filter: false` are the two new RS ones.

## System-Wide Impact

- **Interaction graph**: `run_weekly.run_pipeline` orchestrates → `download_ohlcv_batch` (now also for `^GSPC`) → `compute_all_indicators` → **new** `compute_rs_indicators` (new pipeline stage, marker `indicators_rs`) → `_save_indicators_checkpoint` → `split_screener_config` (now returns `ScreenerGroups`) → `run_all_screeners` (twice, A and B) → EMA5 post-filter on A only → `consolidate_results` (final union) → `build_report_dataframe` → `export_all`.
- **Error propagation**:
  - SP500 download failure → log warning, skip `compute_rs_indicators`, set `rs_ir`/`rs_rating` columns to `NaN`. Both RS screeners produce empty results; pipeline continues.
  - A ticker with insufficient history (< 250 days for R12M) → `rs_weighted_return` is NaN → `rs_rating` is NaN → fails `>= 80` filter naturally, no special-casing needed.
  - YAML references a column that doesn't exist in `indicators_df` (e.g., resume from a stale checkpoint with no `rs_ir`) → engine-startup schema check raises with a clear message, instead of producing silent zero-symbol results from `evaluate_condition`.
- **State lifecycle risks**:
  - The `indicators_rs.parquet` checkpoint includes the RS columns. Resume semantics: presence of `indicators` marker without `indicators_rs` triggers re-enrichment, not screener execution against missing columns. Same `indicators_pure.parquet` from older runs is forward-compatible: re-enrichment runs once on resume.
  - Stale checkpoints from binaries before this change → `indicators_rs` marker is absent → re-enrichment fires automatically. No manual `--no-cache` needed.
- **API surface parity**: The screener engine itself adds one fail-fast schema check at startup; otherwise unchanged. CLI flags (`--screeners`, `--universe`, `--resume`, `--no-cache`) all continue to behave identically.
- **Integration test scenarios**: see "Detailed Implementation Plan §5" below.

## Acceptance Criteria

### Functional Requirements

- [x] `config/screeners.yaml`:
  - [x] `scan_below_ema5` admits `ema5_distance_pct` in `[0, 7.5]` and `pct_change_5d < 7`.
  - [x] `scan_above_ema5` admits `ema5_distance_pct` in `[-7.5, 0]` and `pct_change_5d < 7`.
  - [x] New screener `rs_information_ratio` is defined, `apply_post_filter: false`, `IR >= 1.0`, sort by `rs_ir desc`, universe `us_eu`.
  - [x] New screener `rs_oneil_modified` is defined, `apply_post_filter: false`, `rs_rating >= 80 AND volatility_1m <= 80`, sort by `rs_rating desc`, universe `us_eu`.
- [x] `src/data.py`:
  - [x] `_save_failed_tickers_cache` skips `^GSPC` (literal check; promotable to a set when a second benchmark appears).
- [x] `src/indicators.py`:
  - [x] New function `compute_rs_indicators(indicators_df: pd.DataFrame, sp500_close: pd.Series | None, ohlcv: dict[str, pd.DataFrame]) -> pd.DataFrame` returns the same DataFrame with three added columns: `rs_ir`, `rs_weighted_return`, `rs_rating`. Uses vectorized wide-DataFrame broadcast for IR (no per-ticker `align()` loop).
  - [x] Empty/missing-SP500 path: returns the input DataFrame unchanged with all three columns set to NaN.
- [x] `src/screener.py`:
  - [x] `split_screener_config` returns `ScreenerGroups(with_postfilter, bypass, post_filter_only)` (NamedTuple).
  - [x] `run_all_screeners` calls a new `_validate_required_columns(df, config)` helper at startup that raises `ValueError` if any YAML-referenced field/reference is missing from `df.columns`. Soft per-row missing-field behavior in `evaluate_condition` is removed.
- [x] `run_weekly.py`:
  - [x] Stage 2 fetches `^GSPC` alongside the universe; on failure, logs a warning and continues.
  - [x] New stage between current stages 3 and 4: `_mark_stage(ckpt_dir, "indicators_rs")` after `compute_rs_indicators`. Resume reads/writes `indicators_rs.parquet` with atomic `tmp + os.replace` (matches commit 503546c pattern).
  - [x] Stage 5 runs Group A → EMA5 post-filter → A-final, runs Group B → B-final, computes `final = A-final ∪ B-final`, merges traceability across all sources.
  - [x] `--resume` correctly resumes runs that already wrote the new indicators-rs parquet (reads RS columns from disk).

### Non-Functional Requirements

- [x] No measurable regression in run_weekly wall-clock time for runs without RS screeners (the new path is gated by SP500 availability).
- [x] `compute_rs_indicators` completes in < 3s on the `us_eu` universe (~6,880 tickers); IR is vectorized, percentile rank is `df.rank(pct=True)`.

### Quality Gates

- [x] All new logic covered by unit tests with synthetic data.
- [x] One integration test exercises the three-group flow on a tiny mock pipeline.
- [x] One integration test exercises resume from a pre-RS checkpoint (no `indicators_rs` marker → re-enrichment fires).
- [x] `uv run pytest` is green.
- [x] README updated: new screeners, new flag, brief explanation of the bypass path.

## Detailed Implementation Plan

### 1. `config/screeners.yaml` — YAML changes

```yaml
# (1a) Relax existing post-filters
scan_below_ema5:
  description: "Scan Below EMA5: EMA(5) below price by 0-7.5%, SMA10>SMA20, pullback"
  post_filter: true
  requirements:
    - field: ema5_distance_pct
      operator: "between"
      value: [0, 7.5]              # was [0, 5]
    - field: sma_10
      operator: ">"
      reference: sma_20
    - field: pct_change_5d
      operator: "<"
      value: 7                     # was 5
    - field: volatility_1m
      operator: ">="
      value: 3.5

scan_above_ema5:
  description: "Scan Above EMA5: EMA(5) above price by 0-7.5%, SMA10>SMA20, pullback"
  post_filter: true
  requirements:
    - field: ema5_distance_pct
      operator: "between"
      value: [-7.5, 0]             # was [-5, 0]
    - field: sma_10
      operator: ">"
      reference: sma_20
    - field: pct_change_5d
      operator: "<"
      value: 7                     # was 5
    - field: volatility_1m
      operator: ">="
      value: 3.5

# (1b) Two new RS screeners — bypass EMA5
rs_information_ratio:
  description: "RS Information Ratio: Daily outperformance vs SP500, vol-normalized (us_eu, bypasses EMA5)"
  universe: us_eu
  apply_post_filter: false
  requirements:
    - field: rs_ir
      operator: ">="
      value: 1.0
  postprocess:
    sort_by: rs_ir
    sort_order: desc

rs_oneil_modified:
  description: "RS O'Neil-modified: front-loaded weighted return ranked vs us_eu universe (bypasses EMA5)"
  universe: us_eu
  apply_post_filter: false
  requirements:
    - field: rs_rating
      operator: ">="
      value: 80
    - field: volatility_1m
      operator: "<="
      value: 80
  postprocess:
    sort_by: rs_rating
    sort_order: desc
```

### 2. `src/indicators.py` — RS enrichment (vectorized)

Add at the bottom of the file (module-level for testability):

```python
# src/indicators.py (new function — pseudocode)

RS_IR_WINDOW = 63           # ~3 months of trading days
RS_IR_MIN_OBS = 50          # require ≥50 valid aligned days inside the window
RS_IR_STD_FLOOR = 0.005     # 0.5% daily — numerical guard against index-clones (not a statistical estimator)

def compute_rs_indicators(
    indicators_df: pd.DataFrame,
    sp500_close: pd.Series | None,
    ohlcv: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Enrich indicators_df with rs_ir, rs_weighted_return, rs_rating.

    Returns a new DataFrame with three added columns. If sp500_close is None
    or empty, all three columns are set to NaN and a warning is logged.
    """
    df = indicators_df.copy()
    df['rs_ir'] = np.nan
    df['rs_weighted_return'] = np.nan
    df['rs_rating'] = np.nan

    if sp500_close is None or sp500_close.empty:
        logger.warning("SP500 data unavailable — skipping RS indicators")
        return df

    # Dedupe SP500 too — yfinance occasionally emits duplicated timestamps on
    # corporate actions / index reconstitution.
    sp = sp500_close[~sp500_close.index.duplicated(keep='last')].dropna()
    sp500_returns = sp.pct_change().dropna()

    # ── IR via wide-DataFrame broadcast (replaces per-ticker align loop) ──
    # Build a wide DataFrame of last-63d returns per ticker. Align with SP500
    # AFTER assembly so the trailing window is taken on the intersection of
    # dates (otherwise EU stocks lose days near US holidays).
    series_by_ticker = {}
    for t in df.index:
        if t not in ohlcv:
            continue
        close = ohlcv[t]['Close'].dropna()
        if not close.index.is_unique:
            close = close[~close.index.duplicated(keep='last')]
        series_by_ticker[t] = close.pct_change().dropna()

    if not series_by_ticker:
        logger.warning("No overlapping OHLCV tickers for RS — skipping IR")
        # rs_weighted_return + rs_rating still computed below from indicators_df
    else:
        # Concat into wide df indexed by date, then align with SP500 on inner-join,
        # then take the trailing window.
        returns_wide = pd.DataFrame(series_by_ticker)              # date × ticker
        aligned_idx = returns_wide.index.intersection(sp500_returns.index)
        returns_wide = returns_wide.loc[aligned_idx].tail(RS_IR_WINDOW)
        sp_aligned = sp500_returns.loc[returns_wide.index]

        excess = returns_wide.sub(sp_aligned, axis=0)             # broadcast
        counts = excess.count(axis=0)
        mean = excess.mean(axis=0)
        std = excess.std(axis=0, ddof=1).clip(lower=RS_IR_STD_FLOOR)
        ir = ((mean / std) * np.sqrt(252)).where(counts >= RS_IR_MIN_OBS)

        df.loc[ir.index, 'rs_ir'] = ir

    # ── IBD-style weighted return + percentile rank (already vectorized) ──
    weighted = (
        0.4 * df['pct_change_1m'] +
        0.3 * df['pct_change_3m'] +
        0.2 * df['pct_change_6m'] +
        0.1 * df['pct_change_1y']
    )
    df['rs_weighted_return'] = weighted
    df['rs_rating'] = (
        weighted.rank(pct=True, na_option='keep', method='average') * 100
    )

    return df
```

Key fixes vs. brainstorm sketch:
- **Vectorized** instead of per-ticker loop (perf).
- **Align then `tail`**, not the other way around (correctness for EU stocks).
- **`std(ddof=1).clip(lower=...)`** explicit, sample std (Grinold/Kahn convention).
- **`method='average'`** pinned in `rank()` to lock the contract against future pandas defaults.
- **Duplicate-date defense** before `pct_change` (yfinance occasionally returns duplicated timestamps on corporate actions).
- **`pd.Series | None`** type hint (Python 3.11+ syntax, consistent with project).

### 3. `src/screener.py` — split_screener_config update

```python
# src/screener.py (replacement)

from typing import Any, NamedTuple

class ScreenerGroups(NamedTuple):
    """Three screener-config dicts split by routing flag.

    Each field is a full screeners-config dict (preserves global settings),
    differing only in which screeners survive the filter.
    """
    with_postfilter: dict[str, Any]   # regular screeners that flow through EMA5 post-filter
    bypass: dict[str, Any]            # regular screeners with apply_post_filter: false
    post_filter_only: dict[str, Any]  # screeners with post_filter: true (the EMA5 scans)

def split_screener_config(config: dict) -> ScreenerGroups:
    """Split into the three pipeline groups.

    A screener belongs to:
      - post_filter_only:        post_filter == True
      - bypass:                  post_filter is falsy AND apply_post_filter is falsy
      - with_postfilter:         everything else (the default)
    """
    screeners = config.get('screeners', {})
    with_pf, bypass, post = {}, {}, {}
    for name, sdef in screeners.items():
        if sdef.get('post_filter', False):
            post[name] = sdef
        elif not sdef.get('apply_post_filter', True):
            bypass[name] = sdef
        else:
            with_pf[name] = sdef
    return ScreenerGroups(
        with_postfilter={**config, 'screeners': with_pf},
        bypass={**config, 'screeners': bypass},
        post_filter_only={**config, 'screeners': post},
    )

def _validate_required_columns(df: pd.DataFrame, config: dict) -> None:
    """Fail-fast schema check: every YAML-referenced field/reference must
    exist as a column in df. Raises ValueError with the missing fields.
    """
    referenced = set()
    for sdef in config.get('screeners', {}).values():
        for req in sdef.get('requirements', []):
            if 'field' in req:
                referenced.add(req['field'])
            if 'reference' in req:
                referenced.add(req['reference'])
        pp = sdef.get('postprocess') or {}
        if pp.get('sort_by'):
            referenced.add(pp['sort_by'])
    missing = referenced - set(df.columns)
    if missing:
        raise ValueError(
            f"Indicator DataFrame is missing required columns: {sorted(missing)}. "
            "If resuming from a stale checkpoint, re-run with --no-cache."
        )
```

### 4. `run_weekly.py` — stage 2, new stage 3.5, stage 5 changes

**Stage 2 addition** (after the universe OHLCV download):
```python
# run_weekly.py — stage 2 (sketch)

logger.info("Downloading SP500 (^GSPC) for RS calculations...")
sp500_data = download_ohlcv_batch(
    ["^GSPC"], period="1y",
    show_progress=False, cache_max_age_hours=cache_age,
)
sp500_df = sp500_data.get("^GSPC")
sp500_close = (
    sp500_df["Close"]
    if sp500_df is not None and not sp500_df.empty
    else None
)
if sp500_close is None:
    logger.warning("SP500 download failed — RS screeners will produce empty results")
```

**Stage 3.5 (new): RS enrichment with its own checkpoint**
```python
# run_weekly.py — stage 3.5 (new)

indicators_rs_path = ckpt_dir / "indicators_rs.parquet"

if resume and _stage_done(ckpt_dir, "indicators_rs") and indicators_rs_path.exists():
    logger.info("[Stage 3.5/5] Loading RS-enriched indicators from checkpoint...")
    indicators_df = pd.read_parquet(indicators_rs_path)
else:
    logger.info("[Stage 3.5/5] Computing RS indicators (IR + rs_rating)...")
    indicators_df = compute_rs_indicators(indicators_df, sp500_close, ohlcv_data)
    if not indicators_df.empty:
        # Atomic write — same pattern as commit 503546c (checkpoint collision fix).
        # Prevents Ctrl-C mid-write from corrupting the parquet.
        tmp = indicators_rs_path.with_suffix(".parquet.tmp")
        indicators_df.to_parquet(tmp)
        os.replace(tmp, indicators_rs_path)
    _mark_stage(ckpt_dir, "indicators_rs")
```
Note: this runs **after** the existing fundamentals merge and `indicators` marker, so resume semantics are: `indicators` present + `indicators_rs` absent → re-run RS enrichment only. Old checkpoints upgrade in place on resume.

**Stage 5 rewrite** (replace the current two-branch logic):
```python
# run_weekly.py — stage 5 (sketch)

groups = split_screener_config(screeners_config)

# Group A — regular with post-filter
results_a = run_all_screeners(indicators_df, groups.with_postfilter, screener_tickers)
sym_a, trace_a = consolidate_results(results_a, method=consolidation_method,
                                     exclude_symbols=exclude_symbols)

# Apply EMA5 post-filter on Group A only
if groups.post_filter_only.get('screeners') and sym_a:
    pf_tickers = {n: set(sym_a) for n in groups.post_filter_only['screeners']}
    results_pf = run_all_screeners(indicators_df, groups.post_filter_only, pf_tickers)
    sym_a_final, trace_pf = consolidate_results(results_pf, 'union', exclude_symbols)
else:
    sym_a_final, trace_pf = sym_a, {}

# Group B — bypass screeners
results_b = run_all_screeners(indicators_df, groups.bypass, screener_tickers)
sym_b, trace_b = consolidate_results(results_b, method=consolidation_method,
                                     exclude_symbols=exclude_symbols)

# Final union
final_symbols = sorted(set(sym_a_final) | set(sym_b))
traceability = {
    s: (trace_a.get(s, []) + trace_pf.get(s, []) + trace_b.get(s, []))
    for s in final_symbols
}

screener_results = {**results_a, **results_pf, **results_b}
```

The stage-1 universe-resolution loop (line 174 in current `run_weekly.py`) needs no change: its `if sdef.get('post_filter', False): continue` check still applies. The new RS screeners are NOT post-filters, so they receive their `us_eu` universe normally.

### 5. Tests

New file `tests/test_rs_indicators.py`:
- `test_ir_vectorized_constant_alpha`: synthetic stock series = sp500 returns + 0.001/day for 100 days → IR ≈ √252 (after the std floor doesn't bind).
- `test_ir_zero_alpha_floor_binds`: stock perfectly tracks SP500 (excess.std == 0) → std floor binds, IR ≈ 0 (numerator is 0).
- `test_ir_short_history_returns_nan`: < 50 aligned days → rs_ir is NaN.
- `test_ir_align_then_tail_eu_holiday`: stock has a date present in its index but absent in SP500 (e.g., July 4) within the 63-window → that date drops out, count is 62, IR still computed if ≥ 50.
- `test_ir_duplicate_date_in_close`: stock with a duplicated timestamp → deduplicated before pct_change, no exception.
- `test_rs_rating_three_tickers`: pct_change_* values chosen so weighted returns are 100/50/0 → rs_rating produces values consistent with `pct=True, method='average'`.
- `test_rs_rating_with_nan_history`: a ticker with NaN R12M → rs_weighted_return is NaN → rs_rating is NaN, doesn't poison rank for others.
- `test_compute_rs_indicators_no_sp500`: passing `None` for sp500_close → all three columns NaN, no exception, warning logged.

New file `tests/test_pipeline_three_groups.py`:
- `test_split_screener_config_three_groups`: build a config with one of each kind, verify `ScreenerGroups` fields.
- `test_apply_post_filter_default_true`: a screener without the flag goes into `with_postfilter`.
- `test_bypass_screener_skips_postfilter`: integration with mocked `run_all_screeners`, verify the bypass path's symbols reach the final union.
- `test_resume_pre_rs_checkpoint`: write a checkpoint dir with `indicators` marker but no `indicators_rs` → run_pipeline re-runs RS enrichment on resume, doesn't fall through to screener execution against missing columns.

New file `tests/test_engine_schema_check.py`:
- `test_run_all_screeners_raises_on_missing_column`: indicators_df missing `rs_ir` but config references it → `ValueError` with field name.

`tests/test_p7_benchmark_negative_cache.py` (numbered, follows existing fix-test convention):
- `test_gspc_not_written_to_negative_cache`: simulate a `^GSPC` failure in `download_ohlcv_batch` → `failed_tickers.json` does not contain `^GSPC` afterwards.

## Spec Flow / Edge Cases

Issues uncovered during planning that the brainstorm did not explicitly resolve:

1. **`^GSPC` and the negative cache.** Resolved: `_save_failed_tickers_cache` filters out `BENCHMARK_TICKERS` before writing. No new public API parameter, no caller-side bypass — the data-layer simply refuses to negative-cache the benchmark. (Architecture/framework alignment.)

2. **Naming distinction `post_filter` vs `apply_post_filter`.** Two flags with similar names, different roles. Documented in the YAML field-reference comment block. Validation: a screener with both `post_filter: true` and `apply_post_filter: false` is nonsensical — `split_screener_config` gives `post_filter` precedence, so the `apply_post_filter` is silently ignored. Not worth a warning yet.

3. **EU calendar misalignment.** The `align(..., join='inner')` approach takes the date intersection — that's correct and conservative (an EU stock typically has ~58-62 overlapping days inside a 63-window because most weekdays are shared). The `tail(63)` runs **after** alignment to avoid systematically losing the last few days near US holidays.

4. **Resume with old binaries' checkpoints.** The new `indicators_rs` checkpoint marker means resume-from-old-checkpoint is automatic: presence of `indicators` without `indicators_rs` triggers re-enrichment. No `--no-cache` needed.

5. **Schema check at engine startup.** `_validate_required_columns` raises early with a clear message if a YAML-referenced field is missing from `indicators_df`. Replaces the previous soft "warn and return all-False" behavior in `evaluate_condition`. This catches the silent-empty-result class of bug.

6. **`screeners_passed` order in the report.** Order: `regular_with_pf + post_filter + bypass`. Preserves the causal story per symbol (the regular tag, the post-filter qualifier that gated it, then independent bypass evidence).

7. **What happens when a stock matches both a post-filter screener and a bypass screener?** It appears in `final_symbols` once, with traceability listing both screeners. ✅ Desired union semantics.

8. **`new_rs` screener overlap with `rs_oneil_modified`.** Names are similar, but `new_rs` is a hand-rolled momentum filter (perf 1M/3M/6M/1Y thresholds), `rs_oneil_modified` is a percentile-ranked composite. They will overlap for strong leaders. Acceptable — complementary lenses, traceability shows both.

9. **Calibration uncertainty.** Thresholds (IR ≥ 1.0, RS ≥ 80, vol ≤ 80, weighted-return weights 40/30/20/10) are first-pass values. Reserve 1-2 weekly runs to verify the empirical distribution.

## Open Questions

To revisit *after* 2-3 production runs:

- **IR threshold calibration.** Is IR ≥ 1.0 producing a useful (non-empty, non-flooding) list? Industry benchmarks (Goodwin, Robeco) put 1.0 at "exceptional" for *managers*; for single-stock 63-day screens the threshold may end up tighter (1.5+) or looser (0.7) depending on regime.
- **IBD weights.** `rs_oneil_modified` uses 40/30/20/10 over 1M/3M/6M/12M. Canonical O'Neil is 40/20/20/20 over 3M/6M/9M/12M. If the modified weights produce noisy/unstable lists, fall back to canonical and add a separate short-horizon screener instead. (Short-horizon momentum at 1M risks contamination from short-term reversal — well-documented in academic literature.)
- **`rs_rating` ≥ 80 + vol ≤ 80.** Are these producing a useful list? If consistently > ~50 symbols, tighten. If consistently 0, loosen.
- **IR ↔ IBD overlap.** If union produces too many duplicates, future iteration may add a `rs_high_conviction` (IR ≥ 1.5 AND rs_rating ≥ 90) — out of scope for this plan.
- **Pipeline-stage scaling.** The current routing is two flags (`post_filter`, `apply_post_filter`). When a third routing distinction appears, replace with a single `pipeline_stage: regular | bypass | post_filter | …` enum and a dispatch dict. **Trigger:** the third bypass-shaped flag.
- **Benchmark identity.** `^GSPC` is hardcoded in stage 2. If a regional benchmark (STOXX 600 for EU) becomes interesting, promote to a `settings.rs_benchmark` config key. Not now — YAGNI.

## Sources & References

### Origin

- **Brainstorm document:** [docs/brainstorms/2026-04-24-relax-postfilter-and-add-rs-screeners-brainstorm.md](../brainstorms/2026-04-24-relax-postfilter-and-add-rs-screeners-brainstorm.md). Key decisions carried forward:
  1. Surgical YAML relaxation (no architecture change for Theme A).
  2. IR + IBD union (not a composite) for Theme B — preserves traceability.
  3. New `apply_post_filter: bool` flag, default `true` — preserves existing behavior.

### Internal References

- Pipeline orchestration: `run_weekly.py:119-418` (`run_pipeline`), specifically the regular/post-filter split at `run_weekly.py:314-376`.
- Universe-resolution skip for post-filters: `run_weekly.py:171-194` (already correct for new RS screeners since they are NOT `post_filter: true`).
- Screener engine: `src/screener.py:26-80` (`evaluate_condition` — column lookup only, justifies the precomputed-column approach), `src/screener.py:190-203` (`split_screener_config` — to be replaced with `ScreenerGroups` NamedTuple).
- Indicators pipeline: `src/indicators.py:100-195` (`_compute_single_ticker` per-ticker worker — cannot host RS), `src/indicators.py:198-254` (`compute_all_indicators` — RS enrichment hooks immediately after).
- OHLCV download (handles `^GSPC`): `src/data.py:146-331` (`download_ohlcv_batch`); negative cache at `src/data.py:111-134` (`_save_failed_tickers_cache` — needs benchmark exemption).
- Volatility unit (annualized %): `src/indicators.py:79-86`.
- Test conventions: `tests/conftest.py`, `tests/test_p1_infinity.py` (mock yf.Ticker, integration with pandas DataFrames).

### Configuration

- `config/screeners.yaml` — current state with the relaxed/added screeners.
- README screener table: `README.md:46-66` — must list the two new RS screeners under "Screeners regulares" and a brief note on the bypass flag.

### External References

**Information Ratio (Grinold/Kahn) — formula validation:**
- [Goodwin – The Information Ratio (TSG)](https://tsgperformance.com/wp-content/uploads/2020/11/Goodwin-information-ratio.pdf)
- [Wikipedia – Information ratio](https://en.wikipedia.org/wiki/Information_ratio)
- [AnalystPrep – Active Risk, Tracking Risk & IR (CFA L2)](https://analystprep.com/study-notes/cfa-level-2/explain-sources-of-active-risk-and-interpret-tracking-risk-and-the-information-ratio/)
- [SSGA – The power of Information Ratio](https://www.ssga.com/us/en/intermediary/insights/the-power-of-information-ratio-ir-in-active-management)
- [Robeco – Fundamental Law of Active Management](https://www.robeco.com/en-int/insights/2018/04/fundamental-law-of-active-management-shows-way-to-higher-information-ratio)

**IBD/O'Neil RS Rating — canonical formula and momentum-horizon literature:**
- [Skyte – IBD-style RS reference implementation](https://github.com/skyte/relative-strength) (canonical: 40/20/20/20 over 3M/6M/9M/12M)
- [Chartink – O'Neil RS formula](https://chartink.com/screener/relative-strength-of-the-stock-based-on-o-neil-formula)
- [TradingView – IBD-Style Relative Strength](https://www.tradingview.com/script/SHE1xOMC-Relative-Strength-IBD-Style/)
- [SSRN – Evaluating a 12-1 Month Momentum Strategy (2005-2024)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5367656)
- [Tandfonline – The Many Facets of Stock Momentum (CFA Institute, 2025)](https://www.tandfonline.com/doi/full/10.1080/0015198X.2025.2562790)
- [AlphaArchitect – Momentum factor investing](https://alphaarchitect.com/momentum-factor-investing/)

**Pandas / yfinance pattern docs:**
- [pandas Series.align](https://pandas.pydata.org/docs/reference/api/pandas.Series.align.html)
- [pandas Series.pct_change](https://pandas.pydata.org/docs/reference/api/pandas.Series.pct_change.html)
- [pandas DataFrame.rank](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rank.html)
- [pandas 3.0 Copy-on-Write migration](https://pandas.pydata.org/docs/dev/whatsnew/v3.0.0.html)
- [yfinance download reference](https://ranaroussi.github.io/yfinance/reference/yfinance.download.html)

### Related work

- Recent commit `44caf21 Add momentum screeners (US+EU) with post-filter pipeline` introduced the existing two-group architecture this plan extends.
- Recent commit `503546c Fix string Infinity crash and checkpoint file collision` — pattern for safe checkpoint handling (atomic `tmp + os.replace`, reused for `indicators_rs.parquet`).
