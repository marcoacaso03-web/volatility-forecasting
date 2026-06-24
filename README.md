# 📈 Volatility Forecasting — GARCH Family

Web app Streamlit per la previsione della volatilità di singoli asset
finanziari utilizzando modelli della famiglia **GARCH**.

I dati di mercato vengono scaricati in tempo reale da **Yahoo Finance**
e i modelli vengono fittati automaticamente sui log-return giornalieri.

---

## Modelli implementati

| Modello | Specifica | Quando usarlo |
|---|---|---|
| **GARCH(1,1)** | σ²_t = ω + α ε²_{t-1} + β σ²_{t-1} | Riferimento base. Simmetrico: shock positivi e negativi hanno lo stesso effetto sulla volatilità. |
| **EGARCH(1,1)** | ln(σ²_t) = ω + α(\|z\|-E\|z\|) + γ z + β ln(σ²_{t-1}) | Quando c'è **asimmetria** (leverage effect): shock negativi aumentano la volatilità più di quelli positivi. |
| **GJR-GARCH(1,1)** | σ²_t = ω + (α + γ·I_{ε<0}) ε²_{t-1} + β σ²_{t-1} | Alternativa a EGARCH per catturare il leverage effect. Il termine γ misura l'extra-volatilità da shock negativi. |

### Come scegliere
- **GARCH**: buon modello di partenza, robusto e semplice.
- **EGARCH**: preferito quando si vuole modellare l'asimmetria nel log-varianza (più stabile numericamente).
- **GJR-GARCH**: interpretazione intuitiva del parametro γ (effetto leverage diretto).

---

## Asset supportati

I ticker di default sono:

- **SPY** — SPDR S&P 500 ETF
- **NVDA** — NVIDIA Corporation
- **AAPL** — Apple Inc.
- **MSFT** — Microsoft Corporation

È possibile inserire **qualsiasi ticker Yahoo Finance** tramite il campo
libero nella sidebar (es. `TSLA`, `BTC-USD`, `^GSPC`, `EURUSD=X`).

---

## Orizzonti di previsione

| Etichetta | Trading Days | Equivalente |
|---|---|---|
| 1 giorno | 1 | 1 giorno |
| 1 mese | 21 | ≈ 1 mese lavorativo |
| 3 mesi | 63 | ≈ 1 trimestre |
| 6 mesi | 126 | ≈ 1 semestre |

> **Nota**: 1 anno = 252 trading days. La volatilità è sempre annualizzata.

---

## Installazione

### Prerequisiti
- Python 3.10+
- pip

### Step

```bash
# 1. Clona la repo
git clone https://github.com/<tuo-username>/volatility-forecasting.git
cd volatility-forecasting

# 2. Crea un ambiente virtuale (consigliato)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 3. Installa le dipendenze
pip install -r requirements.txt

# 4. Avvia l'app
streamlit run app/main.py
```

L'app sarà disponibile su `http://localhost:8501`.

---

## Struttura del progetto

```
volatility-forecasting/
│
├── app/
│   ├── __init__.py
│   ├── main.py                  # Entry point Streamlit — orchestratore principale
│   │
│   ├── components/
│   │   ├── __init__.py
│   │   ├── sidebar.py           # UI sidebar: ticker, anni, modelli, orizzonti
│   │   ├── results_table.py     # Tabella comparativa forecast + metric cards
│   │   └── diagnostics.py       # Fit statistics, parametri, persistenza
│   │
│   └── core/
│       ├── __init__.py
│       ├── data_loader.py       # Download dati Yahoo Finance con caching
│       ├── returns.py           # Log-return, volatilità storica e realizzata
│       ├── models.py            # GARCH, EGARCH, GJR-GARCH (interfaccia comune)
│       └── forecaster.py        # Logica multi-horizon forecast
│
├── config/
│   ├── __init__.py
│   └── settings.py              # Costanti globali (ticker, orizzonti, fattori)
│
├── requirements.txt             # Dipendenze pinned
└── README.md                    # Questo file
```

---

## Come interpretare i risultati

### AIC / BIC (Akaike / Bayesian Information Criterion)
- **Valori più bassi = modello migliore**.
- Penalizzano la complessità: un modello con più parametri deve
  "meritarsi" la complessità con un fit significativamente migliore.
- L'app evidenzia automaticamente il modello con **miglior AIC**.

### Persistenza della varianza
- **GARCH**: persistenza = α + β
- **GJR-GARCH**: persistenza = α + β + γ/2
- **EGARCH**: persistenza ≈ β (nel log-varianza)

| Valore | Interpretazione |
|---|---|
| < 0.97 | ✅ Modello stazionario, mean-reverting. Shock si dissolvono rapidamente. |
| 0.97 – 1.0 | ⚠️ Alta persistenza. La volatilità decade lentamente, shock molto persistenti. |
| > 1.0 | ❌ Modello non stazionario. La varianza esplode nel lungo periodo. |

### Volatilità annualizzata
- Espressa in **percentuale**.
- Formula: σ_annual = σ_giornaliera × √252
- Esempio: 20% annualizzato ≈ 1.26% giornaliero.

---

## Note teoriche

### Forecast multi-step e mean-reversion

Il forecast h-step ahead della varianza condizionale converge al valore
di lungo periodo:

```
σ²_LR = ω / (1 - α - β)      [GARCH]
```

Per orizzonti brevi il forecast riflette le condizioni attuali di mercato;
per orizzonti lunghi converge alla varianza incondizionata.

La volatilità annualizzata per orizzonte h è calcolata come:

```
σ_h = √( Σᵢ₌₁ʰ σ²_{t+i} / h  × 252 )
```

dove σ²_{t+i} è la varianza condizionale prevista allo step i.

---

## Limitazioni

1. **Distribuzione normale**: tutti i modelli usano `dist='normal'`.
   I rendimenti finanziari hanno tipicamente **fat tails** (kurtosi > 3).
   Per un modello più realistico si può usare `dist='t'` (Student-t).

2. **Nessun regime-switching**: i modelli assumono parametri costanti.
   In periodi di crisi la struttura della volatilità può cambiare.

3. **Singolo asset**: non c'è modellazione della correlazione cross-asset
   (per quello servirebbero modelli multivariate come DCC-GARCH).

4. **Look-ahead bias**: il forecast è basato su dati storici e non
   incorpora informazioni forward-looking.

---

## Licenza

MIT License — vedi [LICENSE](LICENSE) per i dettagli.
