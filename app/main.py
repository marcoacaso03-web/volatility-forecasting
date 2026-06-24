"""
Entry point dell'applicazione Volatility Forecasting.

Avvia la web app Streamlit che permette di:
1. Selezionare un ticker finanziario
2. Scegliere i modelli GARCH da confrontare
3. Configurare gli orizzonti di previsione
4. Visualizzare forecast, metriche e diagnostica del modello

Uso:
    streamlit run app/main.py
"""

import os
import sys

# Ensure the app directory is in sys.path for Streamlit Cloud
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Dict, List

import streamlit as st

from components.diagnostics import render_diagnostics
from components.results_table import render_results
from components.sidebar import render_sidebar
from core.data_loader import load_data
from core.forecaster import forecast_volatility
from core.models import EGARCHModel, GARCHModel, GJRGARCHModel
from core.returns import (
    compute_log_returns,
    compute_realized_volatility,
)
from config.settings import HORIZONS

# Model registry: maps display name → class
MODEL_REGISTRY: Dict[str, type] = {
    "GARCH": GARCHModel,
    "EGARCH": EGARCHModel,
    "GJR-GARCH": GJRGARCHModel,
}


def main() -> None:
    """Flusso principale dell'applicazione."""
    st.set_page_config(
        page_title="Volatility Forecasting",
        page_icon="📈",
        layout="wide",
    )
    st.title("📈 Volatility Forecasting — GARCH Family")
    st.caption(
        "Stima della volatilità futura con modelli GARCH, EGARCH e GJR-GARCH"
    )

    # ------------------------------------------------------------------
    # 1. Render sidebar e raccolta input
    # ------------------------------------------------------------------
    inputs = render_sidebar()

    if not inputs["run"]:
        st.info("👈 Configura i parametri nella sidebar e premi **Calcola Forecast**.")
        return

    # Validate inputs
    if not inputs["models"]:
        st.error("Seleziona almeno un modello.")
        st.stop()

    if not inputs["horizons"]:
        st.error("Seleziona almeno un orizzonte di previsione.")
        st.stop()

    # ------------------------------------------------------------------
    # 2. Download dati
    # ------------------------------------------------------------------
    try:
        with st.spinner(f"📡 Download dati per {inputs['ticker']}..."):
            prices = load_data(inputs["ticker"], inputs["years"])
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Errore imprevisto durante il download: {exc}")
        st.stop()

    # ------------------------------------------------------------------
    # 3. Calcolo log-return
    # ------------------------------------------------------------------
    returns = compute_log_returns(prices["Close"])

    # Statistiche descrittive
    st.subheader("Statistiche dei log-return")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Return medio", f"{returns.mean() * 100:.4f}%")
    col2.metric(
        "Vol. storica ann.",
        f"{compute_realized_volatility(returns) * 100:.2f}%",
    )
    col3.metric("Skewness", f"{returns.skew():.3f}")
    col4.metric("Kurtosi", f"{returns.kurt():.3f}")

    # ------------------------------------------------------------------
    # 4. Fitting modelli selezionati
    # ------------------------------------------------------------------
    results_forecast: Dict[str, object] = {}
    models_fitted: Dict[str, object] = {}
    selected_horizons = {k: HORIZONS[k] for k in inputs["horizons"]}

    for model_name in inputs["models"]:
        with st.spinner(f"🔧 Fitting {model_name}..."):
            try:
                model_cls = MODEL_REGISTRY[model_name]
                m = model_cls()
                m.fit(returns)
                models_fitted[model_name] = m
                results_forecast[model_name] = forecast_volatility(
                    m, selected_horizons
                )
            except RuntimeError as exc:
                st.error(f"Errore nel fitting di {model_name}: {exc}")
                continue
            except Exception as exc:
                st.error(f"Errore imprevisto con {model_name}: {exc}")
                continue

    if not results_forecast:
        st.error("Nessun modello ha prodotto risultati. Controlla i dati e riprova.")
        st.stop()

    # ------------------------------------------------------------------
    # 5. Risultati e diagnostica
    # ------------------------------------------------------------------
    st.subheader("Forecast della Volatilità")
    render_results(results_forecast, models_fitted)

    st.subheader("Diagnostica del Modello")
    render_diagnostics(models_fitted)


if __name__ == "__main__":
    main()
