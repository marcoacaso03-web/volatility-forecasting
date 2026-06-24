"""
Modulo per il calcolo dei log-return e della volatilità storica/realizzata.

Contiene tre funzioni principali:
- ``compute_log_return``  : log-return giornalieri
- ``compute_historical_volatility`` : rolling volatilità annualizzata
- ``compute_realized_volatility``   : volatilità realizzata sul campione
"""

import numpy as np
import pandas as pd

from config.settings import ANNUALIZATION_FACTOR


# ---------------------------------------------------------------------------
# Log-returns
# ---------------------------------------------------------------------------

def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Calcola i log-return giornalieri da una serie di prezzi.

    .. math::

        r_t = \ln\\left(\\frac{P_t}{P_{t-1}}\\right)

    Parameters
    ----------
    prices : pd.Series
        Serie dei prezzi con indice datetime.

    Returns
    -------
    pd.Series
        Log-return giornaliani. Il primo valore (NaN) viene rimosso.
    """
    log_ret: pd.Series = np.log(prices / prices.shift(1))
    log_ret = log_ret.dropna()
    log_ret.name = "log_return"
    return log_ret


# ---------------------------------------------------------------------------
# Volatilità storica (rolling)
# ---------------------------------------------------------------------------

def compute_historical_volatility(
    returns: pd.Series,
    window: int = 21,
) -> pd.Series:
    """Calcola la volatilità annualizzata su finestra mobile.

    Parameters
    ----------
    returns : pd.Series
        Log-return giornalieri.
    window : int, default 21
        Ampiezza della finestra mobile (≈ 1 mese di trading days).

    Returns
    -------
    pd.Series
        Volatilità annualizzata (valori in unità, NON in %).
    """
    rolling_std: pd.Series = returns.rolling(window=window).std()
    hist_vol: pd.Series = rolling_std * np.sqrt(ANNUALIZATION_FACTOR)
    hist_vol.name = "historical_vol"
    return hist_vol


# ---------------------------------------------------------------------------
# Volatilità realizzata (campione completo)
# ---------------------------------------------------------------------------

def compute_realized_volatility(returns: pd.Series) -> float:
    """Calcola la volatilità realizzata sull'intero campione.

    .. math::

        \\sigma_{real} = \\text{std}(r) \\times \\sqrt{252}

    Parameters
    ----------
    returns : pd.Series
        Log-return giornalieri.

    Returns
    -------
    float
        Volatilità annualizzata (unità, NON in %).
    """
    return float(returns.std() * np.sqrt(ANNUALIZATION_FACTOR))
