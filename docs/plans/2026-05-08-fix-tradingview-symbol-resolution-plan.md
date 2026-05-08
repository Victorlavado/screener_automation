---
title: Fix TradingView symbol resolution (ARCA/BATS prefixes, Russell 3000, dual-class shares)
type: fix
status: completed
date: 2026-05-08
origin: docs/brainstorms/2026-05-07-tradingview-symbol-resolution-brainstorm.md
---

# Fix TradingView Symbol Resolution

## Overview

Corregir los prefijos de exchange y formato de ticker que el pipeline emite a
los archivos `candidates_*.txt` para que TradingView pueda importarlos sin
errores de "símbolo inexistente". Tres fixes estáticos sobre `src/export.py` y
`src/universe.py`, sin introducir red adicional ni dependencias nuevas.

Casos guía reportados: `ARCA:JPSE`, `ARCA:HVAC` deben ser `AMEX:JPSE`,
`AMEX:HVAC`.

## Problem Statement / Motivation

En `output/candidates_2026-05-02.txt` (run reciente, universe `global_all`):

- **471 símbolos con prefijo `ARCA:`** — TradingView no acepta `ARCA:` como
  prefijo. Los ETFs listados en NYSE Arca aparecen en TradingView bajo
  `AMEX:` por legado histórico (NYSE Arca proviene de la antigua American
  Stock Exchange).
- **137 símbolos con prefijo `BATS:`** — TradingView no usa `BATS:` (ahora
  Cboe BZX). Para la mayoría de equities heredados de BATS, el prefijo
  funcional en TradingView es `AMEX:`.
- **Russell 3000 hardcodea `NYSE:`** para todos sus tickers
  (`src/universe.py:286`). Para universes que combinan Russell 3000 con
  `nasdaq_listed`/`nyse_listed`, el `set` final contiene entradas duplicadas
  (`NYSE:AAPL` y `NASDAQ:AAPL`). En `run_weekly.py:105`
  (`build_exchange_mapping`) el orden de iteración del `set` decide cuál
  "gana" — no determinístico, y aproximadamente la mitad de los tickers
  acaba con exchange equivocado.
- **Tickers con `.` (BRK.B, BF.B)** — Yahoo necesita `BRK-B`; TradingView
  necesita `BRK.B`. El exporter actual no revierte el cambio.

Total observado: ~600 de los ~6000 símbolos del archivo más grande (~10%) se
importan rotos.

(Ver brainstorm: `docs/brainstorms/2026-05-07-tradingview-symbol-resolution-brainstorm.md`)

## Proposed Solution

Fix estático en cuatro frentes, todos en `src/export.py` y `src/universe.py`,
ordenados por confianza/impacto:

1. **Mapeo `ARCA → AMEX` y `BATS → AMEX`** en `_TV_EXCHANGE_MAP`
   (`src/export.py:42`). Una entrada nueva por cada uno.
2. **Russell 3000 deja de hardcodear `NYSE:`**. `fetch_russell3000_symbols`
   reutiliza internamente `fetch_nasdaq_listed()` + `fetch_nyse_listed()`
   para construir un lookup ticker→exchange y emite cada ticker con su
   exchange real. Los tickers no encontrados en esas dos fuentes se
   descartan (Russell 3000 ⊂ NASDAQ+NYSE; los que faltan son OTC/delisted
   que tampoco resolverían en TradingView).
3. **Restaurar `.` para US dual-class shares** en `_to_tradingview_symbol`.
   Si exchange ∈ {NYSE, NASDAQ, AMEX, ARCA, BATS} y ticker matchea
   `^[A-Z]+-[A-Z]$`, revertir el guión a punto. Mantiene intacto el
   formato europeo (`OMXCOP:NOVO-B`, `OMXCOP:MAERSK-B`).

### Por qué este enfoque (ver brainstorm)

Se descartó la validación online contra `symbol-search.tradingview.com`
para esta iteración: los patrones detectados explican la mayoría del fallo
observado, son baratos, no añaden dependencias de red al pipeline ni
complican el manejo de errores en runs programados. Si tras el fix queda
residual relevante, se revisita la validación online en una iteración
posterior con datos reales (YAGNI — ver brainstorm).

## Technical Considerations

### Cambios concretos

**`src/export.py`** (≈10 líneas):

```python
# Línea 42 — añadir entradas a _TV_EXCHANGE_MAP
_TV_EXCHANGE_MAP = {
    'EPA': 'EURONEXT',
    'AMS': 'EURONEXT',
    'EBR': 'EURONEXT',
    'STO': 'OMXSTO',
    'CPH': 'OMXCOP',
    'ARCA': 'AMEX',   # NEW: NYSE Arca → AMEX en TradingView
    'BATS': 'AMEX',   # NEW: Cboe BZX → AMEX en TradingView
}

# Función nueva
import re
_US_EXCHANGES = frozenset({'NYSE', 'NASDAQ', 'AMEX', 'ARCA', 'BATS'})
_DUAL_CLASS_RE = re.compile(r'^([A-Z]+)-([A-Z])$')

def _restore_us_dual_class(ticker: str, exchange: str) -> str:
    """Restore dot notation for US dual-class shares (BRK-B → BRK.B).

    Yahoo Finance uses hyphens (BRK-B); TradingView uses dots (BRK.B).
    Only applies to US exchanges; European exchanges keep hyphens
    (e.g. OMXCOP:NOVO-B is correct).
    """
    if exchange in _US_EXCHANGES:
        m = _DUAL_CLASS_RE.match(ticker)
        if m:
            return f"{m.group(1)}.{m.group(2)}"
    return ticker

# Modificar _to_tradingview_symbol — aplicar la restauración antes de emitir
def _to_tradingview_symbol(ticker: str, exchange: str) -> str:
    tv_ticker = _strip_yahoo_suffix(ticker)
    tv_exchange = _map_tv_exchange(exchange)
    tv_ticker = _restore_us_dual_class(tv_ticker, tv_exchange)
    return f"{tv_exchange}:{tv_ticker}"
```

**`src/universe.py`** (`fetch_russell3000_symbols`, ≈15 líneas
reescritas):

```python
def fetch_russell3000_symbols() -> List[str]:
    """Fetch Russell 3000 with REAL exchanges from NASDAQ/NYSE FTP listings."""
    cache_path = get_cache_path("russell3000")
    if is_cache_valid(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)

    try:
        # ... (fetch CSV from iShares as before, parse df) ...

        # NEW: build ticker -> exchange lookup from authoritative sources
        nasdaq = fetch_nasdaq_listed()
        nyse = fetch_nyse_listed()
        ticker_to_exchange = {}
        for sym in nasdaq + nyse:
            if ':' in sym:
                exch, tick = sym.split(':', 1)
                ticker_to_exchange[tick] = exch

        symbols = []
        dropped = 0
        for ticker in df[ticker_col].dropna():
            ticker = str(ticker).strip()
            if not ticker or ticker == '-' or len(ticker) > 5:
                continue
            ticker = ticker.replace('.', '-')  # yfinance compat
            exch = ticker_to_exchange.get(ticker)
            if exch:
                symbols.append(f"{exch}:{ticker}")
            else:
                dropped += 1

        if dropped:
            logger.info("Russell 3000: dropped %d tickers not in NASDAQ/NYSE listings", dropped)

        with open(cache_path, 'w') as f:
            json.dump(symbols, f)
        return symbols
    except Exception as e:
        logger.error("Error fetching Russell 3000: %s", e)
        return []
```

### BATS → AMEX: validación empírica al implementar

Antes de fijar `BATS → AMEX` definitivamente, comprobar manualmente en
TradingView 2-3 tickers `BATS:` reales del archivo `candidates_2026-05-02.txt`
(elegir aleatorios). Si todos resuelven como `AMEX:<ticker>`, regla
confirmada. Si alguno falla, dejarlo registrado como residual para la
iteración del validador online.

### Performance

- `_to_tradingview_symbol` añade una regex match por símbolo. Con ~6000
  tickers, ms-level. Despreciable.
- `fetch_russell3000_symbols` ahora llama a `fetch_nasdaq_listed` y
  `fetch_nyse_listed`. Ambos están cacheados (24h TTL); en runs reales
  son lookup en disco. En runs en frío añade ~2 requests HTTP que el
  pipeline ya hace de todos modos para otras fuentes.

## System-Wide Impact

- **Interaction graph**: cambios en `_to_tradingview_symbol` afectan a
  `_write_tv_watchlist_file` → `export_tradingview_watchlist` y
  `export_tradingview_watchlists_per_screener` (split por screener cuando
  >1000). Ambos paths del export usan la misma función.
- **Error propagation**: ninguna ruta nueva de error. Si Russell 3000 no
  encuentra un ticker en las listings, se descarta silenciosamente con un
  log INFO. No afecta a otros sources del universe.
- **State lifecycle risks**: el caché de Russell 3000
  (`.cache/russell3000_symbols.json`) cambia de formato (los entries
  ahora pueden ser `NASDAQ:X` además de `NYSE:X`). Habrá que invalidar el
  caché existente al desplegar (eliminar el archivo) o forzar `--no-cache`
  el primer run. Documentar en el commit.
- **API surface parity**: el universe `us_extended` (sp500 + nasdaq100 +
  russell3000) sigue funcionando, ahora con exchanges correctos en
  Russell 3000. El bug análogo en `fetch_sp500_symbols` (que también
  hardcodea `NYSE:`) NO se aborda en este plan — ver Open Questions.
- **Integration test scenarios**:
  1. Universe `us_extended`: AAPL aparece como `NASDAQ:AAPL` (no
     `NYSE:AAPL`).
  2. Universe `global_all` con candidates reales: el `.txt` exportado no
     contiene `ARCA:` ni `BATS:` como prefijos.
  3. Ticker `BRK-B` en universe US se exporta como `NYSE:BRK.B`.
  4. Ticker `NOVO-B` en universe `europe` se exporta como `OMXCOP:NOVO-B`
     (hyphen preservado).

## Acceptance Criteria

- [x] `_TV_EXCHANGE_MAP` mapea `ARCA → AMEX` y `BATS → AMEX`.
- [x] `_to_tradingview_symbol("BRK-B", "NYSE")` retorna `"NYSE:BRK.B"`.
- [x] `_to_tradingview_symbol("NOVO-B", "CPH")` retorna `"OMXCOP:NOVO-B"`
  (sin alterar el guión).
- [x] `fetch_russell3000_symbols()` retorna entradas con prefijos
  `NASDAQ:` o `NYSE:` según corresponda; tickers no encontrados se
  descartan con log INFO. Misma corrección aplicada a
  `fetch_sp500_symbols` (open question del plan resuelta: incluido en este PR).
- [ ] Tras correr el pipeline (`uv run python run_weekly.py --no-cache`),
  el `output/candidates_*.txt` resultante contiene **0** prefijos
  `ARCA:`, `BATS:`, `BRK-B`, `BF-B`. *(verificación post-merge: requiere ejecución del run semanal.)*
- [ ] Validación manual: importar el `.txt` resultante en TradingView
  → tasa de errores < 2% del total (target). *(verificación post-merge.)*
- [x] Tests existentes en `tests/test_p6_tv_symbols.py` siguen pasando
  tras actualizar `test_arca_unchanged` → `test_arca_to_amex` (cambia el contrato).
- [x] Tests nuevos cubren: `test_bats_to_amex`, `test_us_dual_class_share_*`,
  `test_european_hyphen_preserved`, `test_arca_etf_to_amex`.

## Success Metrics

- **Símbolos `ARCA:` en candidates**: 471 → 0
- **Símbolos `BATS:` en candidates**: 137 → 0
- **Tickers con `-` único minúscula post-export**: 0 (todos los `BRK-B`
  reconvertidos a `BRK.B`)
- **Tasa de errores al importar a TradingView**: ~10% → <2%

## Dependencies & Risks

- **Riesgo medio**: `BATS → AMEX` puede no ser correcto para 100% de
  símbolos `BATS:`. Algunos lanzamientos recientes en Cboe BZX usan
  prefijo `CBOE:` en TradingView. Mitigación: validar 2-3 ejemplos
  durante implementación; documentar residual si lo hubiera.
- **Riesgo bajo**: `_DUAL_CLASS_RE` (`^[A-Z]+-[A-Z]$`) puede tener falsos
  positivos para tickers US legítimos con guión. Revisión rápida de
  `nasdaq_listed`/`nyse_listed` durante implementación para confirmar
  que no hay tickers US comunes que matcheen el patrón sin ser
  dual-class.
- **Riesgo bajo**: el caché existente de Russell 3000 contiene entradas
  malformadas (`NYSE:X` para todos). Mitigación: forzar `--no-cache` en
  el primer run tras el deploy; documentar en el commit.

## Out of Scope (deferred)

- Validador online contra `symbol-search.tradingview.com` (endpoint,
  caché, retry, fallback). Reabrir si el residual tras este fix > 2%.
- Bug análogo en `fetch_sp500_symbols` (`src/universe.py:95`) que también
  hardcodea `NYSE:` para todos los símbolos del S&P 500. Solo afecta a
  universes que incluyen `sp500` sin `nasdaq_listed` (e.g. `us_indices`,
  `global_indices`). El user actual usa `global_all` por defecto, donde
  no se manifiesta. Ver Open Questions.
- Detección de símbolos delisted en fuentes Wikipedia (S&P 500, FTSE 100,
  DAX, etc.).
- Símbolos europeos / Asia (los reportes apuntan solo a US).

## Open Questions

- **¿Aplicar la misma lógica de fix a `fetch_sp500_symbols`?** El bug
  estructural es el mismo (hardcodea `NYSE:` para todos los tickers). En
  el universe por defecto (`global_all`) no impacta — pero impacta a
  `us_indices` y `global_indices`. Si se aplica, el patrón es idéntico:
  reutilizar `fetch_nasdaq_listed`/`fetch_nyse_listed` como lookup. Decidir
  durante implementación si se incluye en este PR o se difiere.
- **¿BATS → AMEX cubre todos los casos?** Pendiente verificación
  empírica (ver Technical Considerations).

## Sources & References

### Origin

- **Brainstorm**: `docs/brainstorms/2026-05-07-tradingview-symbol-resolution-brainstorm.md`
  — decisiones clave carried forward: (a) fix estático antes que
  validador online, (b) Russell 3000 dedup vía nasdaq/nyse listings,
  (c) preservar guiones europeos.

### Internal References

- `src/export.py:21-86` — `_TV_EXCHANGE_MAP`, `_strip_yahoo_suffix`,
  `_to_tradingview_symbol`.
- `src/universe.py:213-214` — exchange code mapping
  (`{'N': 'NYSE', 'A': 'AMEX', 'P': 'ARCA', 'Z': 'BATS'}`) origen de los
  prefijos `ARCA:` y `BATS:`.
- `src/universe.py:241-295` — `fetch_russell3000_symbols` (objetivo del
  fix).
- `src/universe.py:74-103` — `fetch_sp500_symbols` (bug análogo, deferred).
- `run_weekly.py:105` — `build_exchange_mapping` (donde se manifiesta el
  problema de duplicados Russell 3000).
- `tests/test_p6_tv_symbols.py:127-128` — test
  `test_arca_unchanged` que debe actualizarse para reflejar el nuevo
  contrato `ARCA → AMEX`.

### Validation data

- `output/candidates_2026-05-02.txt` — archivo real con 471 `ARCA:`,
  137 `BATS:`, 75 `AMEX:` (control de regresión).
