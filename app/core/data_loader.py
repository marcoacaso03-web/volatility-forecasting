"""
Modulo per il download e preprocessing dei dati di mercato da Yahoo Finance.

Espone la funzione `load_data` che scarica i prezzi Adjusted Close per un
dato ticker e un orizzonte temporale, restituendo un DataFrame pronto per
il calcolo dei rendimenti.
"""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import streamlit as st
import yfinance as yf

from config.settings import DEFAULT_LOOKBACK_YEARS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_data(ticker: str, years: int = DEFAULT_LOOKBACK_YEARS) -> pd.DataFrame:
    """Scarica i prezzi Adjusted Close da Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Simbolo del titolo (es. ``"SPY"``).
    years : int, default ``DEFAULT_LOOKBACK_YEARS``
        Numero anni di storia da scaricare.

    Returns
    -------
    pd.DataFrame
        DataFrame con indice ``DatetimeIndex`` e colonna ``Close``.
        Se ``Adj Close`` è disponibile viene usata, altrimenti ``Close``.

    Raises
    ------
    ValueError
        Se il ticker non è trovato o i dati restituiti sono vuoti.

    Examples
    --------
    >>> df = load_data("SPY", years=5)
    >>> df.columns
    Index(['Close'], dtype='object')
    """
    # Input validation
    if not ticker or not isinstance(ticker, str):
        raise ValueError(f"Ticker non valido: {ticker!r}")

    ticker = ticker.strip().upper()
    end_dt: datetime = datetime.today()
    start_dt: datetime = end_dt - timedelta(days=years * 365)

    try:
        raw = yf.download(
            ticker,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        raise ValueError(
            f"Errore di connessione durante il download di {ticket}: {exc}"
        ) from exc

    if raw is None or raw.empty:
        raise ValueError(
            f"Nessun dato trovato per il ticker '{ticker}'. "
            "Controlla il simbolo e riprova."
        )

    # yfinance con auto_adjust=True usa 'Close' per i prezzi adjusted
    df = raw[["Close"]].copy()
    df.dropna(inplace=True)

    if df.empty:
        raise ValueError(
            f"I dati per '{ticker}' sono vuoti dopo la rimozione dei NaN."
        )

    return df
