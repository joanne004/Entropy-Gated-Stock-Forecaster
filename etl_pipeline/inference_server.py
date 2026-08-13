"""
FastAPI Inference Server
========================
Serves predictions from the trained XGBoost and LSTM models.

Endpoints:
  GET /health           — confirms the server is running and which tickers are loaded
  GET /predict?ticker=  — returns next-day direction prediction for any supported ticker
                          (defaults to AAPL if no ticker supplied)

Run with:
  python -m uvicorn inference_server:app --reload --port 8001
"""

import os
import joblib # saves/loads files
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException # web framework
from sqlalchemy import create_engine
from xgboost import XGBClassifier
from tensorflow.keras.models import load_model as keras_load_model


# ── Database URL ──────────────────────────────────────────────────────────────
# Cloud (Neon): set DATABASE_URL in the environment, e.g.:
#   postgresql://user:pass@ep-xxx.neon.tech/dbname?sslmode=require
# Local (Docker): set POSTGRES_HOST (defaults to localhost)
DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL:
    DB_URL = DATABASE_URL
else:
    POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
    DB_URL = f"postgresql://stockuser:stockpass@{POSTGRES_HOST}:5432/stockdb"

# All tickers we try to load models for at startup.
# If a ticker's model files don't exist yet it is skipped gracefully.
TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"]

FEATURES = [
    "returns", "log_returns",
    "rsi_14", "adx_14", "macd", "vix",
    "sentiment", "entropy", "alpha", "post_count",
]
WINDOW = 20

# XGBoost gets more weight — LSTM collapses to majority-class on price-only data.
# Increasing XGB weight to 0.85 lets XGBoost drive the combined probability
# while the LSTM's agreement check still acts as a confidence gate.
XGB_WEIGHT     = 0.85
LSTM_WEIGHT    = 0.15
LSTM_THRESHOLD = 0.50  # standard threshold — LSTM is confirmation signal at 15% weight


# ── Database engine (module-level — reuse pool across all requests) ───────────
engine = create_engine(DB_URL, pool_pre_ping=True)

# ── In-memory model store ─────────────────────────────────────────────────────
# models[ticker] = {"xgb": ..., "lstm": ..., "scaler": ...}
# Every request reads from this dict — no disk access after startup.
models = {}


# ── Lifespan — runs once on startup, once on shutdown ────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Everything ABOVE yield runs when the server starts.
    Everything BELOW yield runs when the server stops.

    We load all ticker model trios into models[ticker] here so every
    request can use them instantly without touching disk again.
    Tickers whose saved files don't exist yet are skipped with a warning.
    """
    # Limit TF to only allocate RAM it actually needs (not grab everything)
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    print("\nLoading models...")

    for ticker in TICKERS:
        xgb_path    = f"models/saved/xgboost_{ticker}.json"
        lstm_path   = f"models/saved/lstm_{ticker}.keras"
        scaler_path = f"models/saved/lstm_scaler_{ticker}.pkl"

        # Skip gracefully if this ticker hasn't been trained yet
        missing = [p for p in [xgb_path, lstm_path, scaler_path] if not os.path.exists(p)]
        if missing:
            print(f"  ⚠ Skipping {ticker} — missing: {missing}")
            continue

        xgb = XGBClassifier()
        xgb.load_model(xgb_path)
        models[ticker] = {
            "xgb"   : xgb,
            "lstm"  : keras_load_model(lstm_path),
            "scaler": joblib.load(scaler_path),
        }
        print(f"  ✓ {ticker} — XGBoost + LSTM + Scaler loaded")

    if not models:
        print("  ⚠ No models loaded — train the models first.")
    else:
        print(f"\nAll models ready ({list(models.keys())}) — warming up TF...")
        # Run a dummy inference on the first loaded ticker so TF builds its
        # computation graph now (at startup) instead of on the first real request.
        # This prevents the first-request memory spike that causes 502s.
        first = next(iter(models.values()))
        dummy = np.zeros((1, WINDOW, len(FEATURES)), dtype="float32")
        first["lstm"].predict(dummy, verbose=0)
        print("TF warmup done — server is live.\n")

    yield   # <-- server runs here, handling incoming requests

    # Shutdown: free the memory
    models.clear()
    print("Models unloaded. Server stopped.")


# ── App ───────────────────────────────────────────────────────────────────────
# We pass the lifespan function to FastAPI so it knows to call it on
# startup/shutdown. title and description show up in /docs automatically.
app = FastAPI( #  creates an instance of the FastAPI class
    title       = "AAPL Direction Predictor",
    description = "Entropy-Gated Adaptive Hybrid Forecasting System",
    version     = "1.0.0",
    lifespan    = lifespan, # tells FastAPI to use this function to manage startups and shutdowns
)


# ── GET /health ───────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """
    Simple liveness check — confirms the server is up and which tickers
    have models loaded in memory.
    """
    return {
        "status"          : "ok",
        "tickers_loaded"  : list(models.keys()),
    }


# ── GET /predict ──────────────────────────────────────────────────────────────
@app.get("/predict")
def predict(ticker: str = "AAPL"):
    """
    Pulls the latest 20 rows from fused_features for the requested ticker,
    runs both models, and returns a combined directional prediction.

    Query param: ?ticker=AAPL  (default: AAPL)

    Flow:
      1. Validate ticker has loaded models
      2. Query latest 20 rows from PostgreSQL (most recent first, then reversed)
      3. Validate we have enough data
      4. Run XGBoost on the single latest row  (no scaling needed)
      5. Scale the 20-row window, run LSTM     (scaling required)
      6. Weighted combination: XGBoost 70%, LSTM 30%
      7. Return JSON
    """
    ticker = ticker.upper()

    # ── 1. Validate ticker ────────────────────────────────────────────────────
    if ticker not in models:
        raise HTTPException(
            status_code = 404,
            detail      = f"No models loaded for '{ticker}'. Loaded: {list(models.keys())}",
        )

    ticker_models = models[ticker]

    # ── 2. Fetch latest rows ──────────────────────────────────────────────────
    # DESC + LIMIT 20 gets the 20 most recent rows newest-first.
    # We immediately reverse them so the oldest row is first — both the
    # scaler and the LSTM window expect chronological (oldest → newest) order.
    df = pd.read_sql(
        f"SELECT * FROM fused_features WHERE ticker = '{ticker}' ORDER BY date DESC LIMIT {WINDOW}",
        engine,
    )
    df = df.iloc[::-1].reset_index(drop=True)   # reverse to oldest → newest

    # ── 3. Validate ───────────────────────────────────────────────────────────
    if len(df) < WINDOW:
        raise HTTPException(
            status_code = 400,
            detail      = f"Not enough rows for {ticker}: need {WINDOW}, found {len(df)}",
        )

    as_of_date = str(df["date"].iloc[-1])[:10]

    # ── 4. XGBoost — single row ───────────────────────────────────────────────
    # XGBoost is tree-based — it splits on thresholds, not distances.
    # Scale doesn't matter to it, so we feed raw (unscaled) features.
    # iloc[[-1]] — double brackets keep the result as a DataFrame (shape 1×N)
    # rather than a Series. XGBoost's predict_proba expects 2-D input.
    latest_row = df[FEATURES].iloc[[-1]]

    xgb_proba     = ticker_models["xgb"].predict_proba(latest_row)[0]
    xgb_prob_up   = float(xgb_proba[1])
    xgb_direction = "UP" if xgb_prob_up > 0.5 else "DOWN"

    # ── 5. LSTM — 20-row window ───────────────────────────────────────────────
    # LSTM is sensitive to scale — values must be in [0,1].
    # We use the same scaler that was fitted on this ticker's training data.
    X_raw    = df[FEATURES].values.astype(float)
    X_scaled = ticker_models["scaler"].transform(X_raw)

    X_window = X_scaled[np.newaxis, :, :]          # shape: (1, 20, N_features)

    lstm_prob_up   = float(ticker_models["lstm"].predict(X_window, verbose=0)[0][0])
    lstm_direction = "UP" if lstm_prob_up > LSTM_THRESHOLD else "DOWN"

    # ── 5. Weighted combination ───────────────────────────────────────────────
    # We weight XGBoost at 70% because its DOWN recall was 40% vs LSTM's 4%.
    # A model that can actually detect DOWN days is more useful in practice.
    combined_prob = XGB_WEIGHT * xgb_prob_up + LSTM_WEIGHT * lstm_prob_up

    # Agreement-based confidence — backtest showed that when both models
    # agree on direction, accuracy jumps from 48.4% to 53.2%.
    # When they disagree the signal is unreliable, so we return UNCERTAIN.
    # Probability-distance thresholding was ineffective because combined
    # probabilities never stray more than ~0.03 from 0.5.
    if xgb_direction == lstm_direction:
        combined_direction = xgb_direction   # UP or DOWN — both agree
    else:
        combined_direction = "UNCERTAIN"     # models disagree, don't commit

    # ── 7. Return ─────────────────────────────────────────────────────────────
    return {
        "ticker"               : ticker,
        "as_of_date"           : as_of_date,
        "xgb_probability_up"   : round(xgb_prob_up,   4),
        "xgb_direction"        : xgb_direction,
        "lstm_probability_up"  : round(lstm_prob_up,  4),
        "lstm_direction"       : lstm_direction,
        "combined_probability" : round(combined_prob, 4),
        "combined_direction"   : combined_direction,
        "model_agreement"      : xgb_direction == lstm_direction,
    }
