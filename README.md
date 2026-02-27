# Screener Automation

Automatizacion de screeners de acciones para generar watchlists importables a TradingView.

## Instalacion

Requiere [uv](https://docs.astral.sh/uv/) y Python >= 3.11.

```bash
# Instalar dependencias
uv sync
```

## Uso

### Ejecucion manual

```bash
# Ejecucion completa
uv run python run_weekly.py

# Con opciones
uv run python run_weekly.py --verbose              # Progreso detallado
uv run python run_weekly.py --dry-run              # Sin guardar archivos
uv run python run_weekly.py --screeners canslim    # Solo screener especifico
uv run python run_weekly.py --universe europe      # Usar universo especifico
uv run python run_weekly.py --resume               # Reanudar desde checkpoint
uv run python run_weekly.py --no-cache             # Ignorar caches, descargar todo
```

### Pipeline

El pipeline ejecuta 5 etapas con checkpoint/resume automatico:

1. **Resolucion de universos** - Obtiene la lista de simbolos por screener (paralelo)
2. **Descarga OHLCV** - Datos historicos de precio/volumen con cache en parquet
3. **Indicadores tecnicos** - Calculo paralelo (ProcessPoolExecutor)
4. **Fundamentales** - Pre-screen tecnico para reducir descargas, luego descarga paralela (ThreadPoolExecutor)
5. **Screeners + Exportacion** - Pipeline de 2 etapas:
   - Ejecuta screeners regulares y consolida resultados (union)
   - Aplica post-filtros (EMA5 scans) sobre los resultados consolidados
   - El resultado final es la union de los post-filtros

Si el pipeline se interrumpe, se puede reanudar con `--resume` sin perder progreso.

### Screeners incluidos

#### Screeners regulares (US + EU)

| Screener | Descripcion | Filtros clave |
|----------|-------------|---------------|
| `canslim` | CANSLIM clasico | Revenue growth >25%, EMA10>SMA20, Price<SMA50, Market cap 150M-1500T |
| `alternative_canslim` | CANSLIM alternativo | EPS growth >25% (mismos filtros tecnicos que canslim) |
| `strongest_1m` | Momentum extremo 1M | Perf 1M >50%, Vol1M >5%, Market cap 300M-3000T |
| `strongest_1m_30` | Momentum fuerte 1M | Perf 1M >30%, Vol1M >5%, Market cap 300M-3000T |
| `strongest_3m_50` | Momentum fuerte 3M | Perf 3M >50%, Vol1M >5%, Market cap 300M-3000T |
| `strongest_6m_100` | Momentum fuerte 6M | Perf 6M >100%, Vol1M >5%, Market cap 300M-3000T |
| `new_rs` | Nueva fuerza relativa | Perf 6M entre -25%/25%, Perf 3M >15%, Perf 1M >10%, Perf 1Y >0% |

#### Post-filtros (aplicados sobre resultados de screeners regulares)

| Post-filtro | Descripcion | Filtros |
|-------------|-------------|---------|
| `scan_below_ema5` | Precio cerca por encima de EMA5 | EMA5 debajo del precio 0-5%, SMA10>SMA20, Perf 1W <5%, Vol1M >3.5% |
| `scan_above_ema5` | Precio cerca por debajo de EMA5 | EMA5 encima del precio 0-5%, SMA10>SMA20, Perf 1W <5%, Vol1M >3.5% |

Los post-filtros no se aplican sobre todo el universo. Se ejecutan sobre las acciones que ya pasaron los screeners regulares, filtrando por proximidad a la EMA(5) y condiciones de pullback.

### Salidas

Los archivos se generan en `output/`:

- `candidates_YYYY-MM-DD.txt` - Lista para importar en TradingView
- `report_YYYY-MM-DD.csv` - Reporte detallado con indicadores
- `summary_YYYY-MM-DD.txt` - Resumen legible

### Importar a TradingView

1. En TradingView, ir a cualquier watchlist
2. Click en "..." > "Import list..."
3. Seleccionar el archivo `candidates_*.txt`

## Configuracion

### Universos (`config/universes.yaml`)

Define las fuentes de simbolos. Se resuelven en paralelo via HTTP.

| Universo | Descripcion | Simbolos aprox. |
|----------|-------------|-----------------|
| `us_indices` | S&P 500 + NASDAQ 100 | ~600 |
| `us_complete` | Todos los tickers NASDAQ + NYSE | ~6,500 |
| `us_extended` | S&P 500 + NASDAQ 100 + Russell 3000 | ~3,600 |
| `europe` | FTSE 100, DAX, CAC 40, IBEX 35, FTSE MIB, BEL 20, AEX 25, OMX C25, OMX S30, OBX | ~380 |
| `us_eu` | US completo + todos los indices europeos | ~6,880 |
| `asia_pacific` | Nikkei 225, Hang Seng, KOSPI 200, FTSE China A50, ASX 200 | ~760 |
| `global_indices` | Todos los indices (US + Europa + Asia) | ~1,740 |
| `global_all` | US completo + todos los indices internacionales | ~12,950 |
| `my_watchlists` | Archivos locales en `watchlists/` | Variable |
| `curated` | Lista manual de simbolos | Variable |

### Screeners (`config/screeners.yaml`)

Define los filtros a aplicar. Cada screener tiene:

- `universe`: Que universo evaluar (cada screener puede usar uno diferente)
- `requirements`: Lista de condiciones (tecnicas y/o fundamentales)
- `postprocess`: Ordenamiento y limites
- `post_filter`: `true` para screeners que se aplican sobre resultados de otros screeners

Los screeners que usan campos fundamentales se benefician de un **pre-screen tecnico**: primero se aplican los filtros tecnicos para reducir la lista, y solo se descargan fundamentales para los candidatos resultantes.

#### Operadores soportados

- `>`, `<`, `>=`, `<=`, `==`, `!=`
- `between`: Rango [min, max]
- `reference`: Comparar contra otro campo (ej: `ema_10 > sma_20`)

#### Campos disponibles

**Precio/Volumen:**
- `close`, `open`, `high`, `low`, `volume`

**Medias moviles:**
- `sma_10`, `sma_20`, `sma_50`, `sma_100`, `sma_200`
- `ema_5`, `ema_10`, `ema_20`, `ema_50`

**Indicadores tecnicos:**
- `rsi_14`
- `volume_sma_20`, `volume_sma_30`, `volume_sma_50`, `volume_sma_60`
- `pct_change_1d`, `pct_change_5d`, `pct_change_20d`, `pct_change_1m`, `pct_change_60d`
- `pct_change_3m`, `pct_change_6m`, `pct_change_1y`
- `price_vs_sma50_pct`, `price_vs_sma200_pct`
- `atr_14`, `adr_pct`
- `volatility_1m`
- `ema5_distance_pct` (distancia % del precio respecto a EMA5: positivo = precio encima)
- `high_52w`, `low_52w`, `pct_from_52w_high`, `pct_from_52w_low`

**Fundamentales (descargados bajo demanda):**
- `market_cap`, `pe_ratio`, `forward_pe`, `peg_ratio`, `price_to_book`
- `dividend_yield`, `profit_margin`, `revenue_growth`, `earnings_growth`
- `debt_to_equity`, `current_ratio`
- `sector`, `industry`

> Los indicadores se computan con degradacion gradual: tickers con poco historico (IPOs, mercados recientes) obtienen indicadores parciales en vez de ser descartados. Minimo 20 filas requeridas.

## Cache y rendimiento

El sistema implementa varias capas de cache y paralelismo:

- **OHLCV**: Cache persistente en parquet por ticker (`.cache/ohlcv/`). Validez configurable (16h por defecto).
- **Pipeline**: Checkpoint por etapa (`.cache/pipeline/YYYY-MM-DD/`). `--resume` salta etapas completadas.
- **Universos**: Resolucion paralela de todas las fuentes HTTP con ThreadPoolExecutor.
- **Indicadores**: Calculo paralelo con ProcessPoolExecutor (>200 tickers) con degradacion gradual.
- **Fundamentales**: Pre-screen tecnico reduce ~90-95% de las descargas. Descarga paralela con 8 hilos.
- **OHLCV batches**: Lotes de 500 tickers por peticion a Yahoo Finance.

## Programar ejecucion semanal (Windows Task Scheduler)

1. Abrir "Programador de tareas" (Task Scheduler)
2. Click "Crear tarea basica..."
3. Configurar:
   - **Nombre:** Screener Semanal
   - **Desencadenador:** Semanal, dia y hora preferidos (ej: Domingo 18:00)
   - **Accion:** Iniciar programa
   - **Programa:** `run_weekly.bat`
   - **Iniciar en:** `C:\Users\victo\Documentos\Develop\screener_automation`
4. En propiedades avanzadas:
   - Marcar "Ejecutar tanto si el usuario inicio sesion como si no"
   - Marcar "Ejecutar con los privilegios mas altos"

## Validacion contra TradingView

Para verificar que los resultados coinciden con TradingView:

1. En TradingView, ejecutar tu screener manualmente
2. Click "Export screen results" para descargar CSV
3. Ejecutar validacion:

```bash
uv run python -m src.validate path/to/tradingview_export.csv output/candidates_YYYY-MM-DD.txt
```

## Estructura del proyecto

```
screener_automation/
├── config/
│   ├── screeners.yaml      # Definicion de screeners y post-filtros
│   └── universes.yaml      # Definicion de universos
├── src/
│   ├── universe.py         # Resolucion de universos (19 fuentes globales)
│   ├── data.py             # Descarga OHLCV + fundamentales con cache
│   ├── indicators.py       # Calculo paralelo de indicadores
│   ├── screener.py         # Motor de filtrado con pre-screen y post-filtros
│   ├── export.py           # Exportacion de resultados
│   └── validate.py         # Validacion vs TradingView
├── watchlists/             # Watchlists locales (.txt)
├── output/                 # Archivos generados (ignorado por git)
├── logs/                   # Logs de ejecucion
├── .cache/                 # Cache de datos (ignorado por git)
├── run_weekly.py           # Script principal con checkpoint/resume
├── run_weekly.bat          # Wrapper para Task Scheduler
├── pyproject.toml          # Dependencias y metadata del proyecto
└── uv.lock                 # Lockfile de dependencias
```
