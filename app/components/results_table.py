"""
Componente per la visualizzazione tabulare dei risultati di forecast.

Accetta un dict ``{model_name: DataFrame}`` e produce:
- metric cards per l'orizzonte più breve (GARCH come riferimento)
- tabella comparativa multi-modello
- indicazione del modello con miglior AIC
"""

from typing import Dict, Optional

import pandas as pd
import streamlit as st


def render_results(
    results: Dict[str, pd.DataFrame],
    models_fitted: Optional[Dict] = None,
) -> None:
    """Renderizza i risultati del forecast della volatilità.

    Parameters
    ----------
    results : dict[str, pd.DataFrame]
        Dizionario ``{"GARCH": df, "EGARCH": df, ...}`` dove ogni DataFrame
        ha colonne ``["Orizzonte", "Trading Days", "Vol Forecast (%)"]``.
    models_fitted : dict, optional
        Dizionario ``{"GARCH": model_instance, ...}`` usato per mostrare
        il miglior AIC.
    """
    if not results:
        st.warning("Nessun risultato da mostrare.")
        return

    # ------------------------------------------------------------------
    # 1. Metric cards usando GARCH come riferimento
    # ------------------------------------------------------------------
    garch_key = next((k for k in results if "GARCH" in k and "EGARCH" not in k and "GJR" not in k), None)

    if garch_key and not results[garch_key].empty:
        st.markdown(f"**Riferimento: {garch_key}**")
        garch_df = results[garch_key]
        cols = st.columns(len(garch_df))
        for col, (_, row) in zip(cols, garch_df.iterrows()):
            col.metric(
                label=f"📅 {row['Orizzonte']}",
                value=f"{row['Vol Forecast (%)']:.2f}%",
            )

    # ------------------------------------------------------------------
    # 2. Tabella comparativa: righe = orizzonti, colonne = modelli
    # ------------------------------------------------------------------
    st.markdown("**Confronto modelli**")

    # Merge di tutti i DataFrame su "Orizzonte"
    merged: Optional[pd.DataFrame] = None
    for model_name, df in results.items():
        subset = df[["Orizzonte", "Vol Forecast (%)"]].copy()
        subset.rename(columns={"Vol Forecast (%)": model_name}, inplace=True)
        if merged is None:
            merged = subset
        else:
            merged = merged.merge(subset, on="Orizzonte", how="outer")

    if merged is not None and not merged.empty:
        # Formatta i valori con 2 decimali e suffisso %
        formatters = {
            col: lambda x, _c=col: f"{x:.2f}%"
            for col in merged.columns
            if col != "Orizzonte"
        }
        st.dataframe(
            merged,
            column_config={
                "Orizzonte": st.column_config.TextColumn("Orizzonte"),
                **{
                    col: st.column_config.NumberColumn(col, format="%.2f%%")
                    for col in merged.columns
                    if col != "Orizzonte"
                },
            },
            hide_index=True,
            use_container_width=True,
        )

    # ------------------------------------------------------------------
    # 3. Modello con miglior AIC
    # ------------------------------------------------------------------
    if models_fitted:
        best_model = None
        best_aic = float("inf")
        for name, model in models_fitted.items():
            try:
                summary = model.get_fit_summary()
                if summary["AIC"] < best_aic:
                    best_aic = summary["AIC"]
                    best_model = name
            except Exception:
                continue

        if best_model:
            st.success(
                f"🏆 Modello con miglior AIC: **{best_model}** "
                f"(AIC = {best_aic:.2f})"
            )
