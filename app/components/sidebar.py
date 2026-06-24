"""
Componente sidebar dell'applicazione Streamlit.

Raccoglie le scelte dell'utente (ticker, orizzonte temporale, modelli,
orizzonti di previsione) e le restituisce come dizionario.
"""

from typing import Dict, List

import streamlit as st

from config.settings import (
    DEFAULT_LOOKBACK_YEARS,
    DEFAULT_TICKERS,
    HORIZONS,
    MODELS_AVAILABLE,
)


def render_sidebar() -> Dict[str, object]:
    """Renderizza la sidebar e raccoglie gli input dell'utente.

    Returns
    -------
    dict
        ``{"ticker": str, "years": int, "models": list[str],
          "horizons": list[str], "run": bool}``
    """
    st.sidebar.header("⚙️ Configurazione")

    # ------------------------------------------------------------------
    # 1. Ticker selection
    # ------------------------------------------------------------------
    st.sidebar.markdown("**Ticker**")
    ticker_choice = st.sidebar.selectbox(
        "Seleziona un ticker",
        options=DEFAULT_TICKERS,
        index=0,
        label_visibility="collapsed",
    )
    custom_ticker = st.sidebar.text_input(
        "Oppure inserisci un ticker personalizzato",
        value="",
        placeholder="Es. TSLA, BTC-USD, ^GSPC",
        label_visibility="collapsed",
    )
    ticker = custom_ticker.strip().upper() if custom_ticker.strip() else ticker_choice

    st.sidebar.divider()

    # ------------------------------------------------------------------
    # 2. Anni di storia
    # ------------------------------------------------------------------
    years: int = st.sidebar.slider(
        "Anni di storia",
        min_value=2,
        max_value=10,
        value=DEFAULT_LOOKBACK_YEARS,
        step=1,
    )

    st.sidebar.divider()

    # ------------------------------------------------------------------
    # 3. Modelli da confrontare
    # ------------------------------------------------------------------
    models: List[str] = st.sidebar.multiselect(
        "Modelli da confrontare",
        options=MODELS_AVAILABLE,
        default=MODELS_AVAILABLE,
    )

    st.sidebar.divider()

    # ------------------------------------------------------------------
    # 4. Orizzonti di previsione
    # ------------------------------------------------------------------
    all_horizon_labels = list(HORIZONS.keys())
    horizons: List[str] = st.sidebar.multiselect(
        "Orizzonti di previsione",
        options=all_horizon_labels,
        default=all_horizon_labels,
    )

    st.sidebar.divider()

    # ------------------------------------------------------------------
    # 5. Pulsante di esecuzione
    # ------------------------------------------------------------------
    run: bool = st.sidebar.button(
        "🚀 Calcola Forecast",
        type="primary",
        use_container_width=True,
    )

    return {
        "ticker": ticker,
        "years": years,
        "models": models,
        "horizons": horizons,
        "run": run,
    }
