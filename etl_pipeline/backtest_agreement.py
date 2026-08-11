"""
Backtest — Model Agreement Analysis
=====================================
Runs XGBoost and LSTM on the test set, finds days where both models
agreed on direction, and checks how often the actual market agreed too.

This validates the confidence thresholding idea:
  "When both models agree, are they more accurate than average?"
"""

import os
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from xgboost import XGBClassifier
from tensorflow.keras.models import load_model as keras_load_model

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


def main():
    engine = create_engine(DB_URL)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading fused_features...")
    df = pd.read_sql("SELECT * FROM fused_features ORDER BY date ASC", engine)
    df["date"] = pd.to_datetime(df["date"])

    # Build target: 1 if next day close > today close
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna(subset=["target"] + FEATURES)

    # Same 80/20 split as training
    cutoff_idx  = int(len(df) * 0.8)
    cutoff_date = df["date"].iloc[cutoff_idx]
    test = df.iloc[cutoff_idx:].reset_index(drop=True)
    print(f"Test set: {len(test)} rows ({test['date'].min().date()} → {test['date'].max().date()})")

    # ── Load models ────────────────────────────────────────────────────────────
    print("\nLoading models...")
    xgb = XGBClassifier()
    xgb.load_model(XGB_PATH)
    lstm   = keras_load_model(LSTM_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("  Models loaded.")

    # ── XGBoost predictions on test set ───────────────────────────────────────
    X_test    = test[FEATURES]
    xgb_proba = xgb.predict_proba(X_test)[:, 1]   # prob UP for each row

    # ── LSTM predictions — need full df for the 20-day window context ─────────
    # Scale the entire dataset using the training scaler
    all_scaled = scaler.transform(df[FEATURES].values.astype(float))

    lstm_proba = np.full(len(df), np.nan)
    for i in range(WINDOW, len(df)):
        window     = all_scaled[i - WINDOW:i]          # shape (20, 9)
        x          = window[np.newaxis, :, :]          # shape (1, 20, 9)
        lstm_proba[i] = float(lstm.predict(x, verbose=0)[0][0])

    # Slice to test set indices (offset by cutoff_idx)
    test_start_idx = cutoff_idx
    lstm_proba_test = lstm_proba[test_start_idx: test_start_idx + len(test)]

    # ── Combine and analyse ───────────────────────────────────────────────────
    results = pd.DataFrame({
        "date"           : test["date"].values,
        "actual"         : test["target"].values,
        "xgb_prob_up"    : xgb_proba,
        "lstm_prob_up"   : lstm_proba_test,
    })

    results["xgb_pred"]      = (results["xgb_prob_up"]  > 0.5).astype(int)
    results["lstm_pred"]     = (results["lstm_prob_up"] > 0.5).astype(int)
    results["agreement"]     = results["xgb_pred"] == results["lstm_pred"]
    results["combined_prob"] = 0.7 * results["xgb_prob_up"] + 0.3 * results["lstm_prob_up"]
    results["confident"]     = results["combined_prob"].apply(
        lambda p: abs(p - 0.5) >= 0.05
    )
    results["combined_pred"] = (results["combined_prob"] > 0.5).astype(int)
    results["correct"]       = results["combined_pred"] == results["actual"]

    # ── Print results ──────────────────────────────────────────────────────────
    total      = len(results)
    agree_mask = results["agreement"]
    conf_mask  = results["confident"]

    print(f"\n{'='*55}")
    print(f"  BACKTEST RESULTS — TEST SET ({test['date'].min().date()} → {test['date'].max().date()})")
    print(f"{'='*55}")

    # Overall accuracy
    overall_acc = results["correct"].mean()
    print(f"\n  Total test days       : {total}")
    print(f"  Overall accuracy      : {overall_acc:.1%}")

    # Agreement accuracy
    agree_df    = results[agree_mask]
    agree_acc   = agree_df["correct"].mean() if len(agree_df) else 0
    print(f"\n  Days both models agreed : {len(agree_df)} / {total}  ({len(agree_df)/total:.1%} of test set)")
    print(f"  Accuracy on agreed days : {agree_acc:.1%}")

    # Confident + agreement
    conf_agree_df  = results[agree_mask & conf_mask]
    conf_agree_acc = conf_agree_df["correct"].mean() if len(conf_agree_df) else 0
    print(f"\n  Days agreed AND confident (|prob-0.5| ≥ 0.05) : {len(conf_agree_df)} / {total}  ({len(conf_agree_df)/total:.1%})")
    print(f"  Accuracy on those days                        : {conf_agree_acc:.1%}")

    # UP agreement vs DOWN agreement
    up_agree   = results[agree_mask & (results["combined_pred"] == 1)]
    down_agree = results[agree_mask & (results["combined_pred"] == 0)]
    print(f"\n  UP agreement days   : {len(up_agree)}  →  accuracy {up_agree['correct'].mean():.1%}" if len(up_agree) else "  No UP agreement days")
    print(f"  DOWN agreement days : {len(down_agree)}  →  accuracy {down_agree['correct'].mean():.1%}" if len(down_agree) else "  No DOWN agreement days")

    print(f"\n{'='*55}")

    # Show the top agreement days
    print(f"\n  Sample of days where both models agreed correctly:")
    sample = results[agree_mask & results["correct"]].head(10)[[
        "date", "actual", "combined_pred", "combined_prob"
    ]]
    sample["direction"] = sample["combined_pred"].map({1: "UP", 0: "DOWN"})
    sample["actual_dir"] = sample["actual"].map({1: "UP", 0: "DOWN"})
    print(sample[["date", "direction", "actual_dir", "combined_prob"]].to_string(index=False))


if __name__ == "__main__":
    main()
