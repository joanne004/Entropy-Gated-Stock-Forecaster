"""
FastAPI Inference Server
========================
Serves predictions from the trained XGBoost and LSTM models.

Endpoints:
  GET /health   — confirms the server is running and models are loaded
  GET /predict  — returns next-day direction prediction for AAPL

Run with:
  uvicorn inference_server:app --reload --port 8000 # uvicorn listens on port 8000 for requests and sends the request to FastAPI. Uvicorn is a web server that runs the FastAPI app and listens for incoming HTTP requests
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


# ── Constants ─────────────────────────────────────────────────────────────────
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
DB_URL        = f"postgresql://stockuser:stockpass@{POSTGRES_HOST}:5432/stockdb"

XGB_PATH    = "models/saved/xgboost_model.json"
LSTM_PATH   = "models/saved/lstm_model.keras"
SCALER_PATH = "models/saved/lstm_scaler.pkl"

FEATURES = [
    "returns", "log_returns",
    "rsi_14", "adx_14", "macd",
    "sentiment", "entropy", "alpha", "post_count",
]
WINDOW = 20

# XGBoost gets more weight — it showed 40% DOWN recall vs LSTM's 4%
XGB_WEIGHT  = 0.7
LSTM_WEIGHT = 0.3


# ── In-memory model store ─────────────────────────────────────────────────────
# A plain Python dict that holds the three loaded artifacts.
# Every request reads from this dict — no disk access after startup.
models = {} # plain dictionary that will hold the loaded XGBoost object, LSTM object and the scaler


# ── Lifespan — runs once on startup, once on shutdown ────────────────────────
@asynccontextmanager 
async def lifespan(app: FastAPI): # async means the function can pause and wait for something(like a DB query)
    """
    Everything ABOVE yield runs when the server starts.
    Everything BELOW yield runs when the server stops.

    We load all three artifacts into the models dict here so every
    request can use them instantly without touching disk again.

    We use lifespan because without lifespan the loading of the models will be done inside predict and every time we need a prediction the models will be loaded which takes some extra time
    """
    print("\nLoading models...")

    # Load XGBoost — create an empty classifier then load weights from JSON
    xgb = XGBClassifier() # creates an empty classifier / blank object with no weights or trees
    xgb.load_model(XGB_PATH) # this fills the empty shell with everything from the saved JSON file. When saving the XGBoost model it gets saved as a JSON file
    models["xgb"] = xgb
    print(f"  ✓ XGBoost  loaded  ({XGB_PATH})")

    # Load LSTM — keras reconstructs the full architecture + weights from file
    models["lstm"] = keras_load_model(LSTM_PATH) # load_model() it reads the architecture, rebuilds the layers and then loads the weights into them
    print(f"  ✓ LSTM     loaded  ({LSTM_PATH})")

    # Load scaler — joblib deserialises the fitted MinMaxScaler object
    models["scaler"] = joblib.load(SCALER_PATH) # joblib.dump() is general-purpose Python object serialiser. It can freeze any Python object to a file
    print(f"  ✓ Scaler   loaded  ({SCALER_PATH})")

    print("All models ready — server is live.\n")

    # if any of the load calls fail, python raises an exception and the lifespan function crashes and uvicorn sees the crash and refuses to start
    yield   # <-- server runs here, it starts handling incoming requests here

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
@app.get("/health") # returns if the server is live and running
def health():
    """
    Simple liveness check — confirms the server is up and which models
    are currently loaded in memory.
    """
    return {
        "status"        : "ok",
        "models_loaded" : list(models.keys()),
    }


# ── GET /predict ──────────────────────────────────────────────────────────────
@app.get("/predict") # tells FastAPI when a request is made to predicy endpoint runs this function
def predict():
    """
    Pulls the latest 20 rows from fused_features, runs both models,
    and returns a combined directional prediction for the next trading day.

    Flow:
      1. Query latest 20 rows from PostgreSQL (most recent first, then reversed)
      2. Validate we have enough data
      3. Run XGBoost on the single latest row  (no scaling needed)
      4. Scale the 20-row window, run LSTM     (scaling required)
      5. Weighted combination: XGBoost 70%, LSTM 30%
      6. Return JSON
    """
    engine = create_engine(DB_URL) # creates a connection to PostgresSQL

    # ── 1. Fetch latest rows ──────────────────────────────────────────────────
    # DESC + LIMIT 20 gets the 20 most recent rows newest-first.
    # We immediately reverse them so the oldest row is first — both the
    # scaler and the LSTM window expect chronological (oldest → newest) order.
    df = pd.read_sql(
        f"SELECT * FROM fused_features ORDER BY date DESC LIMIT {WINDOW}",
        engine,
    )
    df = df.iloc[::-1].reset_index(drop=True)   # reverse to oldest → newest

    # ── 2. Validate ───────────────────────────────────────────────────────────
    if len(df) < WINDOW:
        # HTTPException sends an error response back to the caller instead of
        # crashing the server. 400 = Bad Request — the data isn't ready yet.
        raise HTTPException(
            status_code = 400,
            detail      = f"Not enough rows: need {WINDOW}, found {len(df)}",
        )

    as_of_date = str(df["date"].iloc[-1])[:10]   # e.g. "2024-12-30" .iloc[-1] gets the last date convert it from a pandas timestamp to a string and slices the first 10 characters

    # ── 3. XGBoost — single row ───────────────────────────────────────────────
    # XGBoost is tree-based — it splits on thresholds, not distances.
    # Scale doesn't matter to it, so we feed raw (unscaled) features.
    # iloc[[-1]] — double brackets keep the result as a DataFrame (shape 1×8)
    # rather than a Series (shape 8,). XGBoost's predict_proba expects a 2-D input.
    latest_row = df[FEATURES].iloc[[-1]]          # shape: (1, 8) double brackets returns a dataframe and a single bracket returns a series

    xgb_proba     = models["xgb"].predict_proba(latest_row)[0]
    # predict_proba returns a 2-D array: [[prob_DOWN, prob_UP]]
    # [0] takes the first (only) row → [prob_DOWN, prob_UP]
    # [1] takes prob_UP specifically
    xgb_prob_up   = float(xgb_proba[1]) # float converts the numpy's float32 to python's plain float
    xgb_direction = "UP" if xgb_prob_up > 0.5 else "DOWN"

    # ── 4. LSTM — 20-row window ───────────────────────────────────────────────
    # LSTM is sensitive to scale — values must be in [0,1].
    # We use the same scaler that was fitted on training data.
    # .transform() applies the training min/max without re-learning from new data.
    X_raw    = df[FEATURES].values.astype(float)   # shape: (20, 8)
    X_scaled = models["scaler"].transform(X_raw)   # shape: (20, 8)  scaled to [0,1]

    # LSTM expects a 3-D input: (batch_size, time_steps, features)
    # np.newaxis inserts a new axis at position 0, turning (20, 8) into (1, 20, 8)
    # The 1 = batch size of 1 (we are predicting for one window at a time)
    X_window = X_scaled[np.newaxis, :, :]          # shape: (1, 20, 8)

    # model.predict returns shape (1, 1) — one probability for one window.
    # [0][0] unpacks it to a plain float.
    # verbose=0 silences the per-prediction progress bar.
    lstm_prob_up   = float(models["lstm"].predict(X_window, verbose=0)[0][0])
    lstm_direction = "UP" if lstm_prob_up > 0.5 else "DOWN"

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

    # ── 6. Return ─────────────────────────────────────────────────────────────
    # FastAPI automatically converts this dict to a JSON response.
    # round() keeps the probabilities to 4 decimal places — clean output.
    # float() is needed because numpy floats are not JSON-serialisable by default.
    return {
        "as_of_date"           : as_of_date,
        "xgb_probability_up"   : round(xgb_prob_up,    4),
        "xgb_direction"        : xgb_direction,
        "lstm_probability_up"  : round(lstm_prob_up,   4),
        "lstm_direction"       : lstm_direction,
        "combined_probability" : round(combined_prob,  4),
        "combined_direction"   : combined_direction,
        "model_agreement"      : xgb_direction == lstm_direction,
    }
