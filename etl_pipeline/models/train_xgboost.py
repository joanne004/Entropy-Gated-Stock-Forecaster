"""
XGBoost Model Training
======================
Predicts next-day AAPL stock direction: UP (1) or DOWN (0)

Input:  fused_features table in PostgreSQL
        (price indicators + entropy-gated sentiment)
Output: trained model saved to models/saved/xgboost_model.json

Train period: 2018 - 2022
Test  period: 2023 - 2024
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

# ── Database connection ───────────────────────────────────────────────────────
# POSTGRES_HOST comes from the environment when running inside Docker (= "postgres")
# Falls back to "localhost" when running on your own machine
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
DB_URL        = f"postgresql://stockuser:stockpass@{POSTGRES_HOST}:5432/stockdb"
engine        = create_engine(DB_URL) # sets up the connection to Postgres

TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"]


# ── Feature columns ───────────────────────────────────────────────────────────
# These are all the columns from fused_features we will feed into the model.
# We do NOT include "date" or "ticker" — those are identifiers, not signals.
# We do NOT include "target" — that's what we're trying to predict.
FEATURES = [
    # We remove raw OHLC prices (open, high, low, close) and volume.
    # They are non-stationary — AAPL was $40 in 2018 and $180 in 2022.
    # The model would learn "when close > 150 do X" which only holds for
    # certain years, so predictions fall apart on 2023-2024 test data.
    # Instead we keep only features that mean the same thing regardless
    # of what year or price level we are at.

    "returns", "log_returns",                     # daily price movement  (stationary)
    "rsi_14",                                     # momentum 0-100        (bounded)
    "adx_14",                                     # trend strength 0-100  (bounded)
    "macd",                                       # MACD histogram — captures momentum reversals
    "vix",                                        # market fear index — regime signal
    # obv removed — it is cumulative and grows over time (non-stationary)
    # the values during 2018-2022 training are completely different scales
    # to 2023-2024 test, so any split the model learns won't transfer
    "sentiment", "entropy", "alpha", "post_count" # entropy-gated sentiment
]


# ── Step 1: Load data ─────────────────────────────────────────────────────────
def load_data(ticker: str) -> pd.DataFrame:
    """
    Pulls fused_features for a single ticker from PostgreSQL.
    Sorts oldest to newest so the chronological split works correctly.
    """
    print(f"Loading fused_features for {ticker}...")
    df = pd.read_sql(
        f"SELECT * FROM fused_features WHERE ticker = '{ticker}' ORDER BY date ASC",
        engine,
    )
    df["date"] = pd.to_datetime(df["date"])
    print(f"  Loaded {len(df)} rows ({df['date'].min().date()} → {df['date'].max().date()})")
    return df


# ── Step 2: Create target variable ───────────────────────────────────────────
def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Target = 1 if tomorrow's close > today's close (price went UP)
    Target = 0 if tomorrow's close <= today's close (price went DOWN)

    shift(-1) moves the NEXT row's close value into the CURRENT row.
    So on any given day, "target" tells us what happened the NEXT day.
    
    The very last row has no "tomorrow" so we drop it.
    """
    df = df.copy()
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna(subset=["target"]) # last row has NaN target so we remove it

    up_days = df["target"].sum()
    down_days = len(df) - up_days
    print(f" \n Target Distribution: UP = {up_days} ({up_days/len(df):.1%}) | DOWN = {down_days} ({down_days/len(df):.1%})")
    return df


# ── Step 3: Chronological train/test split ────────────────────────────────────
def split_data(df: pd.DataFrame):
    """
    Time-based 80/20 split — never shuffle financial time-series data.
    Shuffling would let the model train on 2024 data and test on 2019 data
    which means it has already seen the future — that's data leakage.

    Instead we sort by date, take the first 80% of rows as training and
    the last 20% as test. The cutoff date is computed from the data so the
    split automatically adjusts whenever new data is added.
    """
    df = df.sort_values("date").reset_index(drop=True)

    # compute the row index at the 80% mark
    cutoff_idx  = int(len(df) * 0.8)
    cutoff_date = df["date"].iloc[cutoff_idx]   # the date at that index

    train = df.iloc[:cutoff_idx]
    test  = df.iloc[cutoff_idx:]

    X_train, y_train = train[FEATURES], train["target"]
    X_test,  y_test  = test[FEATURES],  test["target"]

    print(f"\n  Train: {len(train)} rows ({train['date'].min().date()} → {train['date'].max().date()})")
    print(f"  Test:  {len(test)} rows  ({test['date'].min().date()} → {test['date'].max().date()})")
    print(f"  Split date: {cutoff_date.date()}  (80/20)")
    return X_train, y_train, X_test, y_test


# ── Step 4: Train XGBoost ─────────────────────────────────────────────────────
def train_model(X_train, y_train, X_test, y_test) -> XGBClassifier:
    """
    XGBClassifier key settings:

    n_estimators=300        — build up to 300 trees (early stopping may stop sooner)
    learning_rate=0.01       — each tree's correction is scaled by 1%
                              (small steps = less risk of overshooting)
    max_depth=4             — each tree can ask at most 4 yes/no questions deep
                              (deeper = more complex = more overfitting risk)
    subsample=0.8           — each tree sees a random 80% of training rows
    colsample_bytree=0.8    — each tree sees a random 80% of features
                              (both add randomness to reduce overfitting)
    early_stopping_rounds=20 — stop if test error hasn't improved for 20 rounds
    eval_metric="logloss"   — the error metric we're minimising (log loss works
                              well for binary classification with probabilities)
    """
    print("\nTraining XGBoost...")

    # scale_pos_weight = DOWN days / UP days
    # XGBoost uses this to penalise missing a DOWN day more heavily,
    # so the model stops defaulting to UP for everything uncertain.
    down_days = int((y_train == 0).sum())
    up_days   = int((y_train == 1).sum())
    # scale_pos_weight is kept at 1.0 (neutral) when using logloss.
    # scale_pos_weight modifies the gradient computation — with logloss this
    # distorts the probability targets and early stopping fires immediately (best=0).
    # Instead we use sample_weight in fit() which reweights individual sample losses.
    # compute_sample_weight('balanced') gives DOWN and UP equal total weight,
    # computed from the actual class split — generic, no per-ticker hardcoding.
    spw = 1.0
    sample_weights = compute_sample_weight('balanced', y_train)
    sw_down = sample_weights[y_train == 0][0]
    sw_up   = sample_weights[y_train == 1][0]
    print(f"  scale_pos_weight = {spw:.3f}  (neutral — class balance via sample_weight)")
    print(f"  sample_weight: DOWN={sw_down:.4f}  UP={sw_up:.4f}  (balanced from data)")
    print(f"  Class distribution: DOWN={down_days}, UP={up_days}")

    model = XGBClassifier(
        n_estimators          = 500,
        learning_rate         = 0.01,   # small steps so more trees can train
        max_depth             = 4,      # one level deeper than before — captures 2-way feature
                                        # interactions (e.g. RSI high AND VIX rising → DOWN)
                                        # without going so deep the model memorises noise
        min_child_weight      = 5,      # relaxed from 10 → 5: each leaf needs 5 samples minimum
                                        # diagnosis showed MSFT has real MI signal in RSI/VIX/sentiment;
                                        # 10 was too conservative and prevented those splits from forming
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        early_stopping_rounds = 50,     # increased from 30 → 50: gives logloss more room to improve
                                        # before stopping — AUC plateaus immediately on weak signal,
                                        # logloss keeps improving gradually so needs more patience
        scale_pos_weight      = spw,    # kept at 1.0 — class balance is handled by sample_weight
                                        # in fit() which doesn't conflict with logloss
        eval_metric           = "logloss",  # changed from "auc" — this was the root cause of
                                            # 1-7 trees: AUC on a near-50/50 dataset is flat and
                                            # triggers early stopping immediately. logloss rewards
                                            # small probability improvements each tree so the model
                                            # keeps growing and learning the weak signal that exists
        random_state          = 42,
        verbosity             = 1,
    )

    # eval_set lets XGBoost check test error after every tree
    # verbose=50 prints a status update every 50 trees
    # sample_weight tells XGBoost to penalise misclassifying DOWN days equally
    # to UP days during training. The validation set (eval_set) stays unweighted
    # so early stopping measures true generalisation, not the weighted training loss.
    model.fit(
        X_train, y_train,
        sample_weight = sample_weights,  # balanced weights — DOWN gets extra emphasis
        eval_set = [(X_test, y_test)],   # unweighted validation — fair early stopping
        verbose  = 50,
    )

    print(f"\n  Best number of trees: {model.best_iteration}") # tells you which tree number had the lowest test error
    return model


# ── Step 5: Evaluate ──────────────────────────────────────────────────────────
def evaluate(model, X_train, y_train, X_test, y_test):
    """
    We check both train and test accuracy.

    If train accuracy is much higher than test accuracy (gap > 10%)
    the model has memorised the training data instead of learning
    general patterns — that's overfitting.

    Classification report breaks down accuracy separately for:
      - UP predictions:   when the model says UP, how often is it right?
      - DOWN predictions: when the model says DOWN, how often is it right?
    This matters because a model that always says UP would get ~53% accuracy
    but would be completely useless for DOWN days.
    """
    train_preds = model.predict(X_train) # runs each row through all the trained trees
    test_preds  = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_preds) # accuracy_score counts how many predictions matched the true answer and divides by the total
    test_acc  = accuracy_score(y_test,  test_preds)

    print(f"\n{'='*45}")
    print(f"  Train accuracy : {train_acc:.4f}  ({train_acc*100:.1f}%)")
    print(f"  Test  accuracy : {test_acc:.4f}  ({test_acc*100:.1f}%)")
    if train_acc - test_acc > 0.10:
        print(f"  Train-test gap: {(train_acc - test_acc)*100:.1f}%")
    print(f"{'='*45}")

    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, test_preds, target_names=["DOWN (0)", "UP (1)"]))

    # confusion matrix — rows = actual, cols = predicted
    cm   = confusion_matrix(y_test, test_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["DOWN", "UP"])
    disp.plot(cmap="Blues")
    plt.title("XGBoost — Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig("models/saved/xgboost_confusion_matrix.png")
    plt.close()
    print("  Confusion matrix saved.")

    return test_preds


# ── Step 6: Feature importance ────────────────────────────────────────────────
def plot_feature_importance(model):
    """
    XGBoost tracks how often each feature was used for a split
    and how much it reduced the error on average.
    This tells us which features the model relied on most.
    """
    importance = pd.Series(model.feature_importances_, index=FEATURES)
    importance = importance.sort_values(ascending=True)  # ascending for horizontal bar

    plt.figure(figsize=(10, 7))
    importance.plot(kind="barh", color="steelblue")
    plt.title("XGBoost — Feature Importance")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("models/saved/xgboost_feature_importance.png")
    plt.close()

    print("\nFeature importance (highest first):")
    print(importance.sort_values(ascending=False).to_string())
    print("\n  Feature importance plot saved.")


# ── Step 7: Save model ────────────────────────────────────────────────────────
def save_model(model, ticker: str):
    """
    Saves the trained model as a JSON file named by ticker.
    The inference server loads the correct file per ticker.
    """
    path = f"models/saved/xgboost_{ticker}.json"
    model.save_model(path)
    print(f"\n  Model saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    for ticker in TICKERS:
        print(f"\n{'='*55}")
        print(f"  XGBoost — Training for {ticker}")
        print(f"{'='*55}")

        # 1. Load
        df = load_data(ticker)
        if len(df) < 100:
            print(f"  Skipping {ticker} — not enough rows ({len(df)})")
            continue

        # 2. Target
        df = create_target(df)

        # 3. Split
        X_train, y_train, X_test, y_test = split_data(df)

        # 4. Train
        model = train_model(X_train, y_train, X_test, y_test)

        # 5. Evaluate
        evaluate(model, X_train, y_train, X_test, y_test)

        # 6. Feature importance
        plot_feature_importance(model)

        # 7. Save
        save_model(model, ticker)

        print(f"\n  Done. {ticker} XGBoost complete.")

    print(f"\n{'='*55}")
    print(f"  All {len(TICKERS)} tickers trained.")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
