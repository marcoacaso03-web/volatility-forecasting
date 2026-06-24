"""
Configurazione globale del progetto Volatility Forecasting.

Contiene costanti e parametri utilizzati in tutti i moduli:
- ticker di default, orizzonti di previsione, modelli disponibili,
  fattore di annualizzazione, livelli di confidenza.
"""

from datetime import timedelta

# ---------------------------------------------------------------------------
# Ticker di default mostrati nella sidebar
# ---------------------------------------------------------------------------
DEFAULT_TICKERS: list[str] = ["SPY", "NVDA", "AAPL", "MSFT"]

# ---------------------------------------------------------------------------
# Orizzonti di previsione (trading days)
# 1 giorno  = 1 TD
# 1 mese    ≈ 21 TD
# 3 mesi    ≈ 63 TD
# 6 mesi    ≈ 126 TD
# ---------------------------------------------------------------------------
HORIZONS: dict[str, int] = {
    "1 giorno": 1,
    "1 mese": 21,
    "3 mesi": 63,
    "6 mesi": 126,
}

# ---------------------------------------------------------------------------
# Modelli GARCH disponibili
# ---------------------------------------------------------------------------
MODELS_AVAILABLE: list[str] = ["GARCH", "EGARCH", "GJR-GARCH"]

# ---------------------------------------------------------------------------
# Parametri generali
# ---------------------------------------------------------------------------
DEFAULT_LOOKBACK_YEARS: int = 5      # anni di storia scaricati di default
ANNUALIZATION_FACTOR: int = 252      # trading days per anno
CONFIDENCE_LEVELS: list[float] = [0.95, 0.99]  # intervalli di confidenza

# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------
DEFAULT_START_DELTA = timedelta(days=DEFAULT_LOOKBACK_YEARS * 365)
