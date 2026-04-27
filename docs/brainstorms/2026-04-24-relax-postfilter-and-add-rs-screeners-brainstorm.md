---
date: 2026-04-24
topic: relax-postfilter-and-add-rs-screeners
---

# Relajar post_filter EMA5 y añadir screeners de Relative Strength

## What We're Building

Dos cambios complementarios al motor de screening:

1. **Relajar el post_filter EMA5** existente para ampliar ligeramente el rango de pullback aceptable, sin alterar la arquitectura de embudo.
2. **Añadir dos screeners nuevos de Relative Strength vs SP500**, ambos con normalización por volatilidad, que **bypasean el post_filter EMA5** y entran directos al output final.

Ambos cambios persiguen el mismo objetivo: que el output capture acciones que hoy se quedan fuera por el cuello de botella del EMA5, sin renunciar a la confirmación de pullback para los screeners donde sí tiene sentido.

## Why This Approach

- Para el post_filter, evaluamos cuatro alternativas (tag informativo, opt-in por screener, doble salida, eliminar concepto). Ganó la **relajación quirúrgica de rangos** por YAGNI: solo toca YAML, no toca arquitectura.
- Para RS, evaluamos tres familias (excess return ajustado por vol, Information Ratio, IBD-style RS Rating). Elegimos **(b) IR + (c) IBD en unión** porque son lentes complementarias: IR mide *consistencia diaria* vs benchmark, IBD mide *magnitud multi-horizonte* vs peers. La unión preserva la trazabilidad existente (`screeners_passed`).
- Para los horizontes, recalibramos hacia ventanas más cortas que las del libro original de O'Neil porque los ciclos de momentum se han comprimido en la última década (HFT, difusión instantánea de información, crowding institucional).

## Key Decisions

### Tema A — Post_filter EMA5

- **`scan_below_ema5`**: `ema5_distance_pct between [0, 7.5]` (antes [0, 5])
- **`scan_above_ema5`**: `ema5_distance_pct between [-7.5, 0]` (antes [-5, 0])
- **`pct_change_5d`**: `< 7` (antes `< 5`) en ambos, por coherencia con el ensanche
- **Sin cambios**: `sma_10 > sma_20` y `volatility_1m >= 3.5` se mantienen
- **Rationale**: ampliar la ventana de pullback aceptable un ~50% sin abandonar la lógica de "comprar cerca del EMA5 con tendencia corta alcista"

### Tema B — Nuevos screeners RS

**Benchmark único**: SP500 (`^GSPC`) para todo el universo `us_eu`. No FX adjustment, no benchmark regional. Asumimos "RS global vs USA"; si en el futuro se quisiera afinar para EU, sería un screener adicional, no un cambio de éste.

**`rs_information_ratio`** — consistencia de outperformance diaria
```
excess_diario = stock_returns − sp500_returns           (ventana: últimos 63d)
IR = (excess.mean() / max(excess.std(), 0.005)) * sqrt(252)
filtro: IR >= 1.0
postprocess: sort_by IR desc
```
- Suelo en `std` (0.5%/día) para evitar explosión cuando una acción se calca al SP500.
- Anualizado para que el umbral 1.0 sea interpretable.
- `apply_post_filter: false` → bypassa EMA5.

**`rs_ibd_rating`** — fuerza relativa multi-horizonte vs universo
```
weighted_return = 0.4*R1M + 0.3*R3M + 0.2*R6M + 0.1*R12M
rs_rating = percentile_rank(weighted_return, universo_us_eu) * 100   # escala 1-100
filtro: rs_rating >= 80 AND volatility_1m <= 80
postprocess: sort_by rs_rating desc
```
- Pesos recalibrados respecto al clásico O'Neil (40/20/20/20 sobre 3M-12M) para favorecer momentum reciente — refleja régimen post-2010.
- Ranking vs **universo entero** (variante α), no vs sector. No requiere fundamentals.
- Cap absoluto de volatilidad (≤80% anualizado) excluye meme/microcaps; aquí se materializa la "normalización por vol propia".
- `apply_post_filter: false` → bypassa EMA5.

### Cambio arquitectónico transversal

Añadir flag `apply_post_filter: bool` por screener en YAML.
- **Default `true`** → preserva comportamiento actual de todos los screeners existentes.
- **`false`** → screener bypassa el post_filter y su salida va directa al output final.

Flujo nuevo en `run_weekly.py`:
1. Split de regulares en dos grupos: `apply_post_filter: true` y `apply_post_filter: false`.
2. Grupo `true` → consolidación → post_filter EMA5 → `symbols_via_postfilter`.
3. Grupo `false` → consolidación → `symbols_bypass`.
4. Output final = `symbols_via_postfilter ∪ symbols_bypass`.
5. `traceability` se mantiene íntegra: cada símbolo lleva las etiquetas de los screeners que pasó.

## Calibración inicial (resumen)

| Knob | Valor |
|---|---|
| Ventana IR | 63d (3M) |
| Umbral IR | ≥ 1.0 anualizado |
| Suelo std exceso | 0.5% diario |
| Pesos IBD | 0.4·R1M + 0.3·R3M + 0.2·R6M + 0.1·R12M |
| Umbral IBD | rs_rating ≥ 80 |
| Filtro vol IBD | volatility_1m ≤ 80 |
| Histórico mínimo | 126 días (descarta IPOs muy nuevos) |
| Universo | `us_eu` |
| Benchmark | `^GSPC` (SP500) |

## Implementation Considerations (para planning)

Tres puntos técnicos no triviales que el plan debe abordar:

1. **Descarga de SP500 como ticker auxiliar**. `data.py` debe descargar `^GSPC` además del universo. Sus returns son input compartido para el IR, no un indicador per-ticker. Cachear igual que el resto.

2. **El IR no encaja en `_compute_single_ticker`** (per-ticker, paralelo). Necesita los returns del SP500 alineados por fecha. Dos opciones:
   - Pasar `sp500_returns` como parámetro adicional a la fn worker (replica el array a cada proceso).
   - Computar IR en una pasada secundaria sobre `indicators_df` después del paralelo, usando el `ohlcv` original.

3. **El `rs_rating` IBD es cross-ticker**: el `percentile_rank` requiere la columna `weighted_return` completa. Debe calcularse después de que `compute_all_indicators` haya devuelto todo el DataFrame, como columna derivada.

## Open Questions

- **Calibración real**: los umbrales (IR ≥ 1.0, RS ≥ 80, vol ≤ 80) son razonables a priori pero hay que validarlos con la distribución empírica del universo `us_eu`. Probable que tras 2-3 ejecuciones haya que ajustar.
- **Solapamiento esperado entre IR e IBD**: si la unión produce demasiados duplicados (acciones que pasan ambos), tal vez merezca exponer un screener compuesto adicional (`rs_high_conviction = IR ≥ 1.5 AND rs_rating ≥ 90`) en una iteración futura. No para ahora.
- **Frecuencia de rotación**: las ventanas cortas implican que la lista cambiará más semana a semana. Si resulta excesivo, pasar a Opción B (recalibración agresiva) sería *contraproducente*; convendría más bien volver hacia ventanas largas.

## Next Steps

→ `/workflows:plan` para detallar:
- Cambios concretos en `config/screeners.yaml` (nuevos screeners + flag `apply_post_filter` + relajación rangos EMA5).
- Cambios en `src/data.py` (descarga `^GSPC`).
- Nueva función `compute_rs_indicators(indicators_df, sp500_returns, ohlcv)` o equivalente, y dónde encaja en el pipeline.
- Cambios en `src/screener.py` (`split_screener_config` ampliada, evaluador de IR e IBD si fueran "operadores virtuales", o pre-cálculo como columnas).
- Cambios en `run_weekly.py` (flujo de tres grupos: post_filter / bypass / post_filter screeners).
- Tests para fórmulas IR e IBD con datos sintéticos.
