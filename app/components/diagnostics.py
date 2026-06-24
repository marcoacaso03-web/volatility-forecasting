"""
Componente per la visualizzazione della diagnostica dei modelli GARCH.

Mostra:
- tabella di fit statistics (Log-Likelihood, AIC, BIC)
- parametri stimati per ciascun modello in un expander
- persistenza della varianza con warning se > 0.97 o > 1.0
"""

from typing import Dict, Optional

import pandas as pd
import streamlit as st

from app.core.models import BaseGARCHModel


def render_diagnostics(models_fitted: Dict[str, BaseGARCHModel]) -> None:
    """Renderizza la diagnostica dei modelli fittati.

    Parameters
    ----------
    models_fitted : dict[str, BaseGARCHModel]
        Dizionario ``{"GARCH": model_instance, "EGARCH": model_instance, ...}``.
    """
    if not models_fitted:
        st.info("Nessun modello fittato da analizzare.")
        return

    # ------------------------------------------------------------------
    # 1. Tabella fit statistics
    # ------------------------------------------------------------------
    rows = []
    for name, model in models_fitted.items():
        try:
            summary = model.get_fit_summary()
            rows.append({
                "Modello": name,
                "Log-Likelihood": round(summary["log_likelihood"], 4),
                "AIC": round(summary["AIC"], 4),
                "BIC": round(summary["BIC"], 4),
            })
        except Exception as exc:
            rows.append({
                "Modello": name,
                "Log-Likelihood": "Errore",
                "AIC": "Errore",
                "BIC": "Errore",
            })
            st.warning(f"Impossibile ottenere statistiche per {name}: {exc}")

    if rows:
        stats_df = pd.DataFrame(rows)
        st.dataframe(
            stats_df,
            column_config={
                "Modello": st.column_config.TextColumn("Modello"),
                "Log-Likelihood": st.column_config.NumberColumn("Log-Likelihood", format="%.4f"),
                "AIC": st.column_config.NumberColumn("AIC", format="%.4f"),
                "BIC": st.column_config.NumberColumn("BIC", format="%.4f"),
            },
            hide_index=True,
            use_container_width=True,
        )

    # ------------------------------------------------------------------
    # 2. Parametri stimati per ciascun modello
    # ------------------------------------------------------------------
    for name, model in models_fitted.items():
        with st.expander(f"📊 Parametri — {name}", expanded=False):
            try:
                params = model.get_params()
                std_err = model.result.std_err
                t_stats = model.result.tvalues
                p_values = model.result.pvalues

                param_rows = []
                for param_name in params:
                    param_rows.append({
                        "Parametro": param_name,
                        "Valore": f"{params[param_name]:.6f}",
                        "Std Error": f"{std_err[param_name]:.6f}",
                        "t-stat": f"{t_stats[param_name]:.4f}",
                        "p-value": f"{p_values[param_name]:.6f}",
                    })

                param_df = pd.DataFrame(param_rows)
                st.dataframe(param_df, hide_index=True, use_container_width=True)

            except Exception as exc:
                st.error(f"Errore nel recupero dei parametri per {name}: {exc}")

    # ------------------------------------------------------------------
    # 3. Persistenza della varianza
    # ------------------------------------------------------------------
    st.markdown("**Persistenza della varianza**")
    for name, model in models_fitted.items():
        try:
            persistence = model.get_persistence()
            if persistence is None:
                st.info(f"{name}: persistenza non calcolabile.")
                continue

            st.write(f"**{name}**: persistenza = **{persistence:.4f}**")

            if persistence > 1.0:
                st.error(
                    f"⚠️ {name}: modello **non stazionario** "
                    f"(persistenza = {persistence:.4f} > 1.0). "
                    "La varianza esplode nel lungo periodo."
                )
            elif persistence > 0.97:
                st.warning(
                    f"⚠️ {name}: **alta persistenza** "
                    f"(persistenza = {persistence:.4f} > 0.97). "
                    "La volatilità decade lentamente — shock persistenti."
                )
            else:
                st.success(
                    f"✅ {name}: persistenza = {persistence:.4f} "
                    f"(modello stazionario, mean-reverting)."
                )

        except Exception as exc:
            st.warning(f"Impossibile calcolare la persistenza per {name}: {exc}")
