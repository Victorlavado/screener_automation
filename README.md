# Screener Automation

Automatización de screeners de acciones para generar watchlists importables a TradingView.

## Instalación

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno (Windows)
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

### Ejecución manual

```bash
# Ejecución completa
python run_weekly.py

# Con opciones
python run_weekly.py --verbose              # Mostrar progreso detallado
python run_weekly.py --dry-run              # Sin guardar archivos
python run_weekly.py --screeners momentum   # Solo screener específico
python run_weekly.py --universe curated     # Usar universo específico
```

### Salidas

Los archivos se generan en `output/`:

- `candidates_YYYY-MM-DD.txt` - Lista para importar en TradingView
- `report_YYYY-MM-DD.csv` - Reporte detallado con indicadores
- `summary_YYYY-MM-DD.txt` - Resumen legible

### Importar a TradingView

1. En TradingView, ir a cualquier watchlist
2. Click en "..." → "Import list..."
3. Seleccionar el archivo `candidates_*.txt`

## Configuración

### Universos (`config/universes.yaml`)

Define las fuentes de símbolos:

- `us_market`: S&P 500 + NASDAQ 100 (descarga automática)
- `my_watchlists`: Archivos locales en `watchlists/`
- `curated`: Lista manual de símbolos

### Screeners (`config/screeners.yaml`)

Define los filtros a aplicar. Cada screener tiene:

- `universe`: Qué lista evaluar
- `requirements`: Lista de condiciones
- `postprocess`: Ordenamiento y límites

#### Operadores soportados

- `>`, `<`, `>=`, `<=`, `==`, `!=`
- `between`: Rango [min, max]

#### Campos disponibles

**Precio/Volumen:**
- `close`, `open`, `high`, `low`, `volume`

**Medias móviles:**
- `sma_10`, `sma_20`, `sma_50`, `sma_100`, `sma_200`
- `ema_10`, `ema_20`, `ema_50`

**Indicadores:**
- `rsi_14`
- `volume_sma_20`, `volume_sma_50`
- `pct_change_1d`, `pct_change_5d`, `pct_change_20d`, `pct_change_60d`
- `atr_14`

**Fundamentales (si disponibles):**
- `pe_ratio`, `market_cap`, `dividend_yield`, etc.

## Programar ejecución semanal (Windows Task Scheduler)

1. Abrir "Programador de tareas" (Task Scheduler)
2. Click "Crear tarea básica..."
3. Configurar:
   - **Nombre:** Screener Semanal
   - **Desencadenador:** Semanal, día y hora preferidos (ej: Domingo 18:00)
   - **Acción:** Iniciar programa
   - **Programa:** `C:\Users\victo\Documentos\Develop\screener_automation\run_weekly.bat`
   - **Iniciar en:** `C:\Users\victo\Documentos\Develop\screener_automation`
4. En propiedades avanzadas:
   - Marcar "Ejecutar tanto si el usuario inició sesión como si no"
   - Marcar "Ejecutar con los privilegios más altos"

## Validación contra TradingView

Para verificar que los resultados coinciden con TradingView:

1. En TradingView, ejecutar tu screener manualmente
2. Click "Export screen results" para descargar CSV
3. Ejecutar validación:

```bash
python -m src.validate path/to/tradingview_export.csv output/candidates_YYYY-MM-DD.txt
```

El reporte mostrará diferencias y posibles causas.

## Estructura del proyecto

```
screener_automation/
├── config/
│   ├── screeners.yaml      # Definición de screeners
│   └── universes.yaml      # Definición de universos
├── src/
│   ├── universe.py         # Resolución de universos
│   ├── data.py             # Descarga de datos
│   ├── indicators.py       # Cálculo de indicadores
│   ├── screener.py         # Motor de filtrado
│   ├── export.py           # Exportación de resultados
│   └── validate.py         # Validación vs TradingView
├── watchlists/             # Watchlists locales (.txt)
├── output/                 # Archivos generados (ignorado por git)
├── logs/                   # Logs de ejecución
├── run_weekly.py           # Script principal
├── run_weekly.bat          # Wrapper para Task Scheduler
└── requirements.txt        # Dependencias Python
```
