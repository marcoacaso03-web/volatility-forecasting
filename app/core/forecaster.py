"""
Modello per il forecast multi-horizon della volatilità condizionale.

La funzione principale ``forecast_volatility`` prende un modello già fittato
e produce un DataFrame con la volatilità annualizzata prevista per ciascun
orizzonte richiesto.

Nota teorica (implementata nei commenti):
    Il forecast multi-step con GARCH usa la proprietà di mean-reversion
    della varianza: la varianza h-step ahead converge a ω/(1-α-β) per h→∞.
    La volatilità annualizzata per orizzonte h è:
        σ_h = sqrt( Σ_{i=1}^{h} σ²_{t+i} / h  × 252 )
"""

from typing import Dict

import numpy as np
import pandas as pd

from config.settings import ANNUALIZATION_FACTOR
from app.core.models import BaseGARCHModel


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def forecast_volatility(
    model: BaseGARCHModel,
    horizons: Dict[str, int],
) -> pd.DataFrame:
    """Produce il forecast della volatilità annualizzata per più orizzonti.

    Parameters
    ----------
    model : BaseGARCHModel
        Istanza di un modello GARCH già fittato (``.fit()`` già chiamato).
    horizons : dict[str, int]
        Dizionario ``{"nome_orizzonte": trading_days}``.
        Es. ``{"1 giorno": 1, "1 mese": 21, "3 mesi": 63, "6 mesi": 126}``.

    Returns
    -------
    pd.DataFrame
        Colonne: ``["Orizzonte", "Trading Days", "Vol Forecast (%)"]``.
        La volatilità è espressa in **percentuale annualizzata**.

    Notes
    -----
    Il forecast multi-step con GARCH usa la proprietà di mean-reversion
    della varianza: la varianza h-step ahead converge a ω/(1-α-β) per h→∞.

    La volatilità annualizzata per orizzonte h è:

    .. math::

        \\sigma_h = \\sqrt{ \\frac{\\sum_{i=1}^{h} \\sigma^2_{t+i}}{h} \\times 252 }

    Poiché i return sono stati scalati ×100 prima del fitting, i risultati
    vanno divisi per 100 per tornare alle unità originali.
    """
    if model.result is None:
        raise RuntimeError("Modello non fittato. Chiama .fit(returns) prima.")

    if not horizons:
        raise ValueError("Il dizionario 'horizons' non può essere vuoto.")

    max_horizon: int = max(horizons.values())

    # ------------------------------------------------------------------
    # 1. Ottieni il forecast della varianza condizionale
    #    arch restituisce la varianza per ciascuno step 1..max_horizon
    #    Il risultato è in unità "return scalati ×100", quindi varianza
    #    va divisa per 100² = 10000 per tornare alle unità originali.
    # ------------------------------------------------------------------
    forecast_result = model.result.forecast(
        horizon=max_horizon,
        reindex=False,
    )

    # forecast_result.variance è un DataFrame con shape (1, max_horizon)
    # L'ultima riga contiene le varianze previste per gli step 1..H
    variance_forecast = forecast_result.variance.values[-1, :]  # shape (H,)

    # ------------------------------------------------------------------
    # 2. Per ogni orizzonte h, calcola la volatilità annualizzata
    # ------------------------------------------------------------------
    rows = []
    for label, h in horizons.items():
        if h < 1:
            continue

        # Varianza cumulata fino al giorno h (unità scalate ×100)
        cum_variance_scaled = float(np.sum(variance_forecast[:h]))

        # Converti in volatilità annualizzata (unità originali):
        #   1. Media per giorno: cum_var / h
        #   2. Annualizza: × 252
        #   3. Radice quadrata: sqrt
        #   4. Rescala: / 100 (perché i return erano ×100)
        annual_vol = np.sqrt(cum_variance_scaled / h * ANNUALIZATION_FACTOR) / 100.0

        rows.append({
            "Orizzonte": label,
            "Trading Days": h,
            "Vol Forecast (%)": round(annual_vol * 100, 2),
        })

    return pd.DataFrame(rows)
