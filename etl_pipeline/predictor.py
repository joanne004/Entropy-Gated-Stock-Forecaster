"""
predictor.py — Direct model inference for Streamlit
====================================================
Loads models once at startup (cached via st.cache_resource) and runs
inference directly inside the Streamlit process. No separate server needed.

TensorFlow/LSTM is optional — if TF is not installed (e.g. on Streamlit Cloud),
the predictor falls back to XGBoost-only mode automatically.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

# Check TF availability at import time — no crash if missing
try:
    from tensorflow.keras.models import load_model as keras_load_model
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL:
    DB_URL = DATABASE_URL
else:
    POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
    DB_URL = f"postgresql://stockuser:stockpass@{POSTGRES_HOST}:5432/stockdb"

TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"]

FEATURES = [
    "returns", "log_returns",
    "rsi_14", "adx_14", "macd", "vix",
    "sentiment", "entropy", "alpha", "post_count",
]
WINDOW = 20


@st.cache_resource
def _load_models():
    """
    Load all ticker models from disk. Called once via st.cache_resource.

    If TF is available: loads XGBoost + LSTM + Scaler for each ticker (full hybrid).
    If TF is not available: loads XGBoost only (deployed/cloud mode).
    """
    from xgboost import XGBClassifier

    store = {}
    base  = os.path.dirname(__file__)   # etl_pipeline/

    for ticker in TICKERS:
        xgb_path = os.path.join(base, f"models/saved/xgboost_{ticker}.json")

        if not os.path.exists(xgb_path):
            continue

        xgb = XGBClassifier()
        xgb.load_model(xgb_path)
        entry = {"xgb": xgb, "lstm": None, "scaler": None}

        if TF_AVAILABLE:
            lstm_path   = os.path.join(base, f"models/saved/lstm_{ticker}.keras")
            scaler_path = os.path.join(base, f"models/saved/lstm_scaler_{ticker}.pkl")
            if os.path.exists(lstm_path) and os.path.exists(scaler_path):
                try:
                    entry["lstm"]   = keras_load_model(lstm_path)
                    entry["scaler"] = joblib.load(scaler_path)
                except Exception:
                    pass  # LSTM load failed — stay in XGBoost-only mode

        store[ticker] = entry

    return store


def predict(ticker: str, models: dict) -> dict:
    """Run inference for one ticker using pre-loaded models."""
    ticker = ticker.upper()
    if ticker not in models:
        return {"error": f"No models loaded for {ticker}"}

    m      = models[ticker]
    engine = create_engine(DB_URL)
    df     = pd.read_sql(
        f"SELECT * FROM fused_features WHERE ticker = '{ticker}' ORDER BY date DESC LIMIT {WINDOW}",
        engine,
    )
    df = df.iloc[::-1].reset_index(drop=True)

    if len(df) < WINDOW:
        return {"error": f"Not enough data for {ticker}: need {WINDOW}, got {len(df)}"}

    as_of_date = str(df["date"].iloc[-1])[:10]

    # ── XGBoost (always runs) ─────────────────────────────────────────────────
    xgb_proba     = m["xgb"].predict_proba(df[FEATURES].iloc[[-1]])[0]
    xgb_prob_up   = float(xgb_proba[1])
    xgb_direction = "UP" if xgb_prob_up > 0.5 else "DOWN"

    # ── LSTM (only when TF is available) ─────────────────────────────────────
    lstm_available = m["lstm"] is not None and m["scaler"] is not None

    # ── LSTM (agreement gate — confirmation only, not blended) ───────────────
    # XGBoost is the primary predictor. LSTM acts as a binary gate:
    #   when both models predict the same direction → HIGH confidence
    #   when they disagree → flag as UNCERTAIN (but still use XGBoost signal)
    if lstm_available:
        X_scaled       = m["scaler"].transform(df[FEATURES].values.astype(float))
        lstm_prob_up   = float(m["lstm"].predict(X_scaled[np.newaxis, :, :], verbose=0)[0][0])
        lstm_direction = "UP" if lstm_prob_up > 0.5 else "DOWN"
        lstm_confirms  = xgb_direction == lstm_direction
    else:
        lstm_prob_up   = None
        lstm_direction = "N/A"
        lstm_confirms  = None  # LSTM not available

    # Primary prediction always comes from XGBoost
    return {
        "ticker"             : ticker,
        "as_of_date"         : as_of_date,
        # XGBoost — primary predictor
        "xgb_probability_up" : round(xgb_prob_up, 4),
        "xgb_direction"      : xgb_direction,
        # LSTM — confirmation gate
        "lstm_probability_up": round(lstm_prob_up, 4) if lstm_prob_up is not None else None,
        "lstm_direction"     : lstm_direction,
        "lstm_confirms"      : lstm_confirms,   # True=agree(high conf), False=disagree, None=unavailable
        "lstm_available"     : lstm_available,
    }
