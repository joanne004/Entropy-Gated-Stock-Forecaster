"""
predictor.py — Direct model inference for Streamlit
====================================================
Loads models once at startup (cached via st.cache_resource) and runs
inference directly inside the Streamlit process. No separate server needed.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

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
WINDOW      = 20
XGB_WEIGHT  = 0.85
LSTM_WEIGHT = 0.15


@st.cache_resource
def _load_models():
    """Load all ticker model trios from disk. Called once via st.cache_resource."""
    from xgboost import XGBClassifier
    from tensorflow.keras.models import load_model as keras_load_model

    store = {}
    base = os.path.dirname(__file__)   # etl_pipeline/
    for ticker in TICKERS:
        xgb_path    = os.path.join(base, f"models/saved/xgboost_{ticker}.json")
        lstm_path   = os.path.join(base, f"models/saved/lstm_{ticker}.keras")
        scaler_path = os.path.join(base, f"models/saved/lstm_scaler_{ticker}.pkl")

        missing = [p for p in [xgb_path, lstm_path, scaler_path] if not os.path.exists(p)]
        if missing:
            continue

        xgb = XGBClassifier()
        xgb.load_model(xgb_path)
        store[ticker] = {
            "xgb"   : xgb,
            "lstm"  : keras_load_model(lstm_path),
            "scaler": joblib.load(scaler_path),
        }
    return store


def predict(ticker: str, models: dict) -> dict:
    """Run inference for one ticker using pre-loaded models."""
    ticker = ticker.upper()
    if ticker not in models:
        return {"error": f"No models loaded for {ticker}"}

    m      = models[ticker]
    engine = create_engine(DB_URL)
    df = pd.read_sql(
        f"SELECT * FROM fused_features WHERE ticker = '{ticker}' ORDER BY date DESC LIMIT {WINDOW}",
        engine,
    )
    df = df.iloc[::-1].reset_index(drop=True)

    if len(df) < WINDOW:
        return {"error": f"Not enough data for {ticker}: need {WINDOW}, got {len(df)}"}

    as_of_date = str(df["date"].iloc[-1])[:10]

    # XGBoost — single latest row
    xgb_proba     = m["xgb"].predict_proba(df[FEATURES].iloc[[-1]])[0]
    xgb_prob_up   = float(xgb_proba[1])
    xgb_direction = "UP" if xgb_prob_up > 0.5 else "DOWN"

    # LSTM — 20-row scaled window
    X_scaled       = m["scaler"].transform(df[FEATURES].values.astype(float))
    lstm_prob_up   = float(m["lstm"].predict(X_scaled[np.newaxis, :, :], verbose=0)[0][0])
    lstm_direction = "UP" if lstm_prob_up > 0.5 else "DOWN"

    combined_prob      = XGB_WEIGHT * xgb_prob_up + LSTM_WEIGHT * lstm_prob_up
    combined_direction = xgb_direction if xgb_direction == lstm_direction else "UNCERTAIN"

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
