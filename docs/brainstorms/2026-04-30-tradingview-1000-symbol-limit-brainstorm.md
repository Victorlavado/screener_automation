---
date: 2026-04-30
topic: tradingview-1000-symbol-limit
---

# Sortear el límite de 1000 símbolos por watchlist en TradingView

## What We're Building

Cambiar el export de la watchlist de TradingView para que, cuando el número total de candidatos supere el límite de TradingView (1000 símbolos por watchlist), el output se **divida en varios ficheros agrupados por screener** en lugar de generar un único fichero imposible de importar entero.

El comportamiento actual (un solo `candidates_YYYY-MM-DD.txt`) se mantiene cuando el total ≤ 1000. El fichero consolidado se conserva siempre como auditoría/backup.

## Why This Approach

Se descartaron alternativas más sencillas (chunking secuencial sin significado, top-1000 por ranking, división por región) en favor de la división por screener porque:

- Permite mirar en TradingView las watchlists agrupadas por la lógica que seleccionó cada símbolo (un usuario puede revisar primero los `momentum` y luego los `rs_*`).
- El código ya recibe `screener_results: Dict[str, {description, count, symbols}]` en `export_all`, así que la información está disponible sin refactor.
- División por exchange o región no escala: con la relajación reciente del post-filtro EMA5 y los nuevos screeners de Relative Strength, US por sí sola podría seguir > 1000.

## Key Decisions

- **Trigger**: solo dividir cuando `len(final_symbols) > 1000`. Con ≤ 1000 el flujo actual no cambia.
- **Criterio de división**: por screener que pasó (un símbolo puede aparecer en varios ficheros si pasó varios screeners).
- **Edge case — un screener individual con > 1000 símbolos**: trocear ese screener en `..._<screener>_part1.txt`, `..._<screener>_part2.txt`, etc. (sin ranking ni filtro adicional).
- **Fichero consolidado**: siempre generar `candidates_YYYY-MM-DD.txt` con los todos los símbolos como auditoría/backup, aunque TV no lo pueda importar entero cuando supera el límite.
- **Naming**: `candidates_YYYY-MM-DD_<screener>.txt` o `candidates_YYYY-MM-DD_<screener>_partN.txt` cuando se trocea.
- **Ubicación**: misma carpeta `output/`.

## Open Questions

- ¿Hace falta también un fichero índice (p. ej. `candidates_2026-04-30_index.txt`) que liste qué ficheros se generaron y con cuántos símbolos cada uno? Por defecto NO se incluye — el `summary_*.txt` ya cubre esa función.
- Tamaño exacto de chunk para el caso `> 1000 en un screener`: ¿1000 exacto o un poco menos para dejar margen (p. ej. 990)? Por defecto **1000 exacto** salvo confirmación.

## Next Steps

→ `/compound-engineering:workflows:plan` para detallar la implementación (función nueva en `src/export.py`, integración con `export_all`, tests).
