"""
Modelli della famiglia GARCH.

Implementa tre modelli con un'interfaccia comune (``BaseGARCHModel``):

- ``GARCHModel``    — GARCH(1,1) simmetrico
- ``EGARCHModel``   — EGARCH(1,1) con effetto asimmetrico nel log-varianza
- ``GJRGARCHModel`` — GJR-GARCH(1,1) con termine di leverage di Glosten-Jagannathan-Runkle

Tutti i modelli scalano i return per 100 prima del fitting (consigliato da
``arch`` per stabilità numerica) e restituiscono parametri ri-scalati.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import pandas as pd
from arch import arch_model

from config.settings import ANNUALIZATION_FACTOR


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseGARCHModel(ABC):
    """Interfaccia comune per tutti i modelli GARCH."""

    def __init__(self) -> None:
        self.result: Optional[Any] = None
        self._model_name: str = self.__class__.__name__

    @abstractmethod
    def fit(self, returns: pd.Series) -> None:
        """Fitta il modello sui log-return.

        Parameters
        ----------
        returns : pd.Series
            Log-return giornaliani (NON scalati).
        """
        ...

    def get_params(self) -> dict[str, float]:
        """Restituisce i parametri stimati come dict.

        Returns
        -------
        dict[str, float]
            Parametri con nome → valore.
        """
        if self.result is None:
            raise RuntimeError("Modello non ancora fittato. Chiama fit() prima.")
        return dict(self.result.params)

    def get_fit_summary(self) -> dict[str, float]:
        """Restituisce le statistiche di fit.

        Returns
        -------
        dict[str, float]
            ``{"log_likelihood": ..., "AIC": ..., "BIC": ...}``
        """
        if self.result is None:
            raise RuntimeError("Modello non ancora fittato. Chiama fit() prima.")
        return {
            "log_likelihood": float(self.result.loglikelihood),
            "AIC": float(self.result.aic),
            "BIC": float(self.result.bic),
        }

    def get_persistence(self) -> Optional[float]:
        """Calcola la persistenza della varianza.

        Per GARCH e GJR-GARCH la persistenza è α + β (+ γ/2 per GJR).
        Per EGARCH è β (nel log-varianza).

        Returns
        -------
        float | None
            Persistenza stimata, o ``None`` se il modello non la supporta.
        """
        try:
            params = self.result.params
            # GARCH / GJR: persistence = alpha[1] + beta[1] (+ gamma[1]/2)
            if "alpha[1]" in params and "beta[1]" in params:
                p = params["alpha[1]"] + params["beta[1]"]
                # gamma può non esserci: .get() default 0
                gamma = params.get("gamma[1]", 0.0)
                if gamma is not None:
                    p += gamma / 2.0
                return float(p)
            # EGARCH: persistence ≈ beta[1] in log-variance space
            if "beta[1]" in params:
                return float(params["beta[1]"])
        except Exception:
            return None
        return None


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------

class GARCHModel(BaseGARCHModel):
    """Modello GARCH(1,1) simmetrico.

    Specifica::

        r_t = μ + ε_t,   ε_t = σ_t z_t,   z_t ~ N(0,1)
        σ²_t = ω + α ε²_{t-1} + β σ²_{t-1}
    """

    def fit(self, returns: pd.Series) -> None:
        # Scale returns x100 per stabilità numerica
        scaled = returns * 100.0
        am = arch_model(scaled, vol="Garch", p=1, q=1, dist="normal")
        try:
            self.result = am.fit(disp="off", show_warning=False)
        except Exception as exc:
            raise RuntimeError(
                f"GARCH(1,1) non convergiuto: {exc}"
            ) from exc


class EGARCHModel(BaseGARCHModel):
    """Modello EGARCH(1,1) — esponenziale, cattura asimmetria.

    Specifica (log-varianza)::

        ln(σ²_t) = ω + α (|z_{t-1}| - E|z|) + γ z_{t-1} + β ln(σ²_{t-1})
    """

    def fit(self, returns: pd.Series) -> None:
        scaled = returns * 100.0
        am = arch_model(scaled, vol="EGARCH", p=1, q=1, dist="normal")
        try:
            self.result = am.fit(disp="off", show_warning=False)
        except Exception as exc:
            raise RuntimeError(
                f"EGARCH(1,1) non convergiuto: {exc}"
            ) from exc


class GJRGARCHModel(BaseGARCHModel):
    """Modello GJR-GARCH(1,1) — cattura l'effetto leverage.

    Specifica::

        σ²_t = ω + (α + γ I_{ε<0}) ε²_{t-1} + β σ²_{t-1}

    dove I_{ε<0} = 1 se ε_{t-1} < 0 (shock negativo amplifica la volatilità).
    """

    def fit(self, returns: pd.Series) -> None:
        scaled = returns * 100.0
        am = arch_model(scaled, vol="Garch", p=1, o=1, q=1, dist="normal")
        try:
            self.result = am.fit(disp="off", show_warning=False)
        except Exception as exc:
            raise RuntimeError(
                f"GJR-GARCH(1,1) non convergiuto: {exc}"
            ) from exc
