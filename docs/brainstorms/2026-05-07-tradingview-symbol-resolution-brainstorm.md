---
date: 2026-05-07
topic: tradingview-symbol-resolution
---

# TradingView Symbol Resolution

## What We're Building

Corregir los prefijos de exchange que el pipeline emite para que TradingView pueda
importar los watchlists sin errores de "símbolo inexistente". Hoy ~600 símbolos
de cada `candidates_*.txt` (~10%) usan prefijos que TradingView no reconoce
(`ARCA:`, `BATS:`) o que no corresponden al exchange real (Russell 3000
hardcodeado como `NYSE:`).

Casos guía: `ARCA:JPSE`, `ARCA:HVAC` deben ser `AMEX:JPSE`, `AMEX:HVAC`.

## Why This Approach

Se consideraron tres caminos:

- **(A) Fix quirúrgico** de los mapeos detectados.
- **(B) Validación online** contra el endpoint no oficial
  `symbol-search.tradingview.com` para validar y autocorregir todo símbolo.
- **(C) Las dos.**

Decisión: empezar con **(A)**. Los patrones identificados explican la mayoría
del fallo observado, son baratos de aplicar y no introducen dependencias de red
nuevas en el pipeline. Si tras el fix queda un residual relevante, revisitamos
(B) con datos reales en mano (YAGNI).

## Key Decisions

- **`ARCA` → `AMEX`** en `_TV_EXCHANGE_MAP` (`src/export.py:42`). NYSE Arca usa
  el prefijo legado `AMEX:` en TradingView. Arregla ~471 símbolos del ejemplo.
- **`BATS` → `AMEX`** en el mismo mapa. Pendiente verificar con 2-3 tickers
  reales en TradingView antes de fijar la regla; si parte de los `BATS:`
  necesita `CBOE:`, se decide en implementación.
- **Russell 3000 deja de hardcodear `NYSE:`**. En `resolve_universe`
  (`src/universe.py`) se deduplica por ticker dando prioridad a las fuentes
  con exchange real (`nasdaq_listed`, `nyse_listed`). Tickers de Russell 3000
  que no aparezcan en esas fuentes se descartan (Russell 3000 ⊂ NASDAQ+NYSE,
  no se pierde cobertura).
- **Tickers con `.` (BRK.B, BF.B...)**. Yahoo necesita `BRK-B`; TradingView
  necesita `BRK.B`. Revertir `-` → `.` en la conversión a símbolo TradingView
  (función nueva o dentro de `_to_tradingview_symbol`). Solo afecta a ~6
  símbolos del S&P 500.

## Out of Scope (deferred)

- Validador online contra TradingView (endpoint, caché, retry, fallback).
- Detección de símbolos delisted en fuentes Wikipedia.
- Símbolos europeos / Asia (los reportes apuntan solo a US).

## Open Questions

- Confirmar empíricamente el destino correcto para `BATS:` (¿`AMEX:` para
  todos? ¿alguno necesita `CBOE:`?). Se resuelve con 2-3 ejemplos en TradingView
  durante la implementación.

## Next Steps

→ `/workflows:plan` para detalles de implementación (cambios concretos en
`src/export.py` y `src/universe.py`, tests/validación manual con un
`candidates_*.txt` reciente).
