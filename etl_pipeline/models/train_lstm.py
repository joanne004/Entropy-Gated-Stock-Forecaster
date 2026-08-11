"""
LSTM Model Training
===================
Predicts next-day AAPL stock direction: UP (1) or DOWN (0)

Key differences from XGBoost:
  1. Input is a 20-day sequence, not a single row
  2. Features must be scaled to 0-1 (LSTM is sensitive to scale)
  3. Both model AND scaler are saved (scaler needed at inference time)

Train period: 2018 - 2022
Test  period: 2023 - 2024
"""

import os
import joblib                          # saves/loads Python objects to files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler        # scales features to 0-1
from sklearn.metrics import (
    accuracy_score, classification_report,
    ConfusionMatrixDisplay, confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight  # computes class weights to handle class imbalance — same idea as scale_pos_weight in XGBoost
from tensorflow.keras.models import Sequential        # layers stacked in order
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping  # stops training early
from tensorflow.keras.optimizers import Adam          # adaptive gradient descent
import random, tensorflow as tf
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW = 20   # how many past days LSTM looks at for each prediction

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
DB_URL        = f"postgresql://stockuser:stockpass@{POSTGRES_HOST}:5432/stockdb"
engine        = create_engine(DB_URL)

# Same stationary features as XGBoost — no raw prices, no OBV
FEATURES = [
    "returns", "log_returns",
    "rsi_14", "adx_14", "macd",
    "sentiment", "entropy", "alpha", "post_count",
]


# ── Step 1: Load data ─────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """Same as XGBoost — pull fused_features, sort oldest to newest."""
    print("Loading fused_features from PostgreSQL...")
    df = pd.read_sql("SELECT * FROM fused_features ORDER BY date ASC", engine)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  Loaded {len(df)} rows  ({df['date'].min().date()} → {df['date'].max().date()})")
    return df


# ── Step 2: Create target variable ───────────────────────────────────────────
def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Same as XGBoost — 1 if tomorrow's close > today's close, else 0."""
    df = df.copy()
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna(subset=["target"])
    up   = df["target"].sum()
    down = len(df) - up
    print(f"  Target: UP={up} ({up/len(df):.1%})  DOWN={down} ({down/len(df):.1%})")
    return df


# ── Step 3: Scale features ────────────────────────────────────────────────────
def scale_features(df: pd.DataFrame):
    """
    MinMaxScaler transforms each feature column to the range [0, 1].
    Formula for each value:  scaled = (value - column_min) / (column_max - column_min)

    Example — RSI column has min=20, max=85:
        RSI=20 → 0.0
        RSI=52 → 0.49
        RSI=85 → 1.0

    Why we can only fit on training data:
        fit_transform() learns the min and max from the data it sees.
        If we included test data, the scaler would know future min/max values
        and use them to scale training data — that's data leakage.
        So we fit on training rows only, then apply (transform) to test rows
        using the same min/max values learned from training.
    """
    print("\nScaling features...")

    # compute the 80% cutoff date from the data so the split adjusts automatically
    cutoff_idx  = int(len(df) * 0.8)
    cutoff_date = df["date"].iloc[cutoff_idx]
    print(f"  Split date: {cutoff_date.date()}  (80/20)")
    split_mask = df["date"] < cutoff_date            # True for training rows
    X_all      = df[FEATURES].values.copy().astype(float)

    scaler = MinMaxScaler()
    # fit_transform: learns min/max from training rows AND scales them
    X_all[split_mask]  = scaler.fit_transform(X_all[split_mask])
    # transform only: applies the same min/max to test rows (no new learning)
    X_all[~split_mask] = scaler.transform(X_all[~split_mask])

    print(f"  Scaler fitted on {split_mask.sum()} training rows")
    print(f"  Features scaled to [0, 1] range")
    return X_all, scaler


# ── Step 4: Create 20-day sliding windows ─────────────────────────────────────
def make_windows(X_scaled: np.ndarray, df: pd.DataFrame):
    """
    Slides a 20-day window through the full dataset.

    For each day i (starting at day 20):
        X window = rows [i-20 : i]   → the 20 days of history
        y target = target on day i   → what happened that day
        date     = date on day i     → so we can split train/test later

    Example with WINDOW=20:
        i=20:  X = rows[0:20],   y = target[20]
        i=21:  X = rows[1:21],   y = target[21]
        i=22:  X = rows[2:22],   y = target[22]

    Result shape:
        X_windows → (n_samples, 20, 8)
                     n_samples predictions
                     each with 20 time steps
                     each time step has 8 features
        y_windows → (n_samples,)  — one 0 or 1 per window
    """
    print("\nCreating 20-day sliding windows...")

    y_all  = df["target"].values
    dates  = df["date"].values

    X_windows, y_windows, date_windows = [], [], []

    for i in range(WINDOW, len(X_scaled)):
        X_windows.append(X_scaled[i - WINDOW : i])  # 20 rows × 8 features
        y_windows.append(y_all[i])                   # the answer for day i
        date_windows.append(dates[i])                # the date for day i

    X_windows = np.array(X_windows)          # shape: (n_samples, 20, 8)
    y_windows = np.array(y_windows)          # shape: (n_samples,)
    dates_arr = pd.to_datetime(date_windows)

    print(f"  Created {len(X_windows)} windows  —  shape: {X_windows.shape}")
    return X_windows, y_windows, dates_arr


# ── Step 5: Train/test split ──────────────────────────────────────────────────
def split_windows(X_windows, y_windows, dates_arr):
    """
    We split by the TARGET DATE of each window — the day being predicted.

    A window targeting a date before 2023  → training
    A window targeting a date from 2023+   → testing

    The first test window's look-back (the 20 days before 2023-01-01)
    naturally uses the last 20 training-period rows — that's fine,
    those are real historical data points, not future data.
    """
    # compute 80/20 cutoff from the windows themselves (df not in scope here)
    cutoff_idx  = int(len(dates_arr) * 0.8)
    cutoff_date = dates_arr[cutoff_idx]
    train_mask  = dates_arr < cutoff_date            # 80% of windows go to train

    X_train = X_windows[train_mask]
    y_train = y_windows[train_mask]
    X_test  = X_windows[~train_mask]
    y_test  = y_windows[~train_mask]

    print(f"\nTrain: {len(X_train)} windows")
    print(f"Test:  {len(X_test)}  windows")
    return X_train, y_train, X_test, y_test


# ── Step 6: Build the model ───────────────────────────────────────────────────
def build_model(n_features: int) -> Sequential:
    """
    Sequential: layers are stacked in order — output of one feeds into next.

    LSTM(64):
        Reads the 20-day sequence one day at a time.
        Maintains a memory (cell state) across all 20 steps.
        After reading all 20 days, outputs 64 numbers — a compressed
        summary of the pattern it detected.
        input_shape=(20, 8) tells it: 20 time steps, 8 features each.

    Dropout(0.3):
        During each training pass, 30% of the 64 LSTM outputs are
        randomly set to 0. Forces the network not to depend on any
        single neuron — reduces overfitting.
        Switched OFF at prediction time (all neurons active).

    Dense(32, activation="relu"):
        A regular fully-connected layer with 32 neurons.
        Takes the 64 LSTM numbers, learns weighted combinations of them.
        relu activation: any negative value becomes 0. Adds non-linearity
        so the layer can learn complex patterns, not just straight lines.

    Dropout(0.2):
        Same idea — randomly zero 20% of the 32 dense outputs.

    Dense(1, activation="sigmoid"):
        Final layer — 1 neuron.
        sigmoid squishes the output to 0-1 → the probability of UP.
        > 0.5 = predict UP, <= 0.5 = predict DOWN.
    """
    model = Sequential([
        LSTM(64, input_shape=(WINDOW, n_features)),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])

    # Adam: adaptive learning rate optimizer
    #       adjusts step size per weight based on how much it has been changing
    # binary_crossentropy: the loss function for 0/1 classification problems
    # accuracy: what we want printed during training so we can watch progress
    model.compile(
        optimizer = Adam(learning_rate=0.0005),  # 0.0005 is lower than default 0.001 — smaller steps reduce the epoch-to-epoch flipping we saw where the model alternated between predicting all-UP and all-DOWN
        loss      = "binary_crossentropy",
        metrics   = ["accuracy"],
    )

    model.summary()  # prints the architecture — layer by layer with parameter counts
    return model


# ── Step 7: Train ─────────────────────────────────────────────────────────────
def train_model(model, X_train, y_train, X_test, y_test):
    """
    epoch: one full pass through ALL training windows.
           In each epoch the model sees every window once, computes the
           error, and updates its weights.

    batch_size=32: instead of updating after every single window,
                   the model processes 32 windows at a time, averages
                   their errors, then takes one gradient step.
                   Faster and more stable than updating after every row.

    EarlyStopping:
        Watches val_loss (error on test set) after each epoch.
        If val_loss hasn't improved for 15 epochs → stop.
        restore_best_weights=True: when it stops, it rewinds to the
        epoch that had the lowest val_loss, not the last epoch.
        This means the saved model is the best one, not the most recent.
    """
    print("\nTraining LSTM...")

    early_stop = EarlyStopping(
        monitor              = "val_loss",
        patience             = 15,            # wait 15 epochs before stopping
        restore_best_weights = True,          # rewind to best epoch on stop
        verbose              = 1,
    )

    # Manual weights instead of balanced — balanced gives DOWN only 1.059 vs UP 0.947,
    # too close to reliably push the model away from the all-UP trap.
    # 1.8 for DOWN means errors on DOWN days cost 1.8× more than UP errors.
    cw = {0: 1.2, 1: 1.0}                                           # 0=DOWN gets heavier penalty, 1=UP is baseline
    print(f"  Class weights — DOWN: {cw[0]:.3f}  UP: {cw[1]:.3f}")

    history = model.fit(
        X_train, y_train,
        epochs          = 100,                # max epochs (early stopping may stop sooner)
        batch_size      = 32,
        validation_data = (X_test, y_test),  # monitored after each epoch
        callbacks       = [early_stop],
        class_weight    = cw,                 # penalise DOWN errors more — same idea as scale_pos_weight in XGBoost
        verbose         = 1,                  # print one line per epoch
    )

    return history


# ── Step 8: Evaluate ──────────────────────────────────────────────────────────
def evaluate(model, X_train, y_train, X_test, y_test):
    """
    model.predict() returns a probability for each window (0-1).
    We convert to 0 or 1 by thresholding at 0.5.
    (model.predict() > 0.5) gives True/False, .astype(int) converts to 1/0.
    .flatten() removes the extra dimension — predict returns shape (n,1),
    we need shape (n,).
    """
    train_preds = (model.predict(X_train) > 0.5).astype(int).flatten()
    test_preds  = (model.predict(X_test)  > 0.5).astype(int).flatten()

    train_acc = accuracy_score(y_train, train_preds)
    test_acc  = accuracy_score(y_test,  test_preds)

    print(f"\n{'='*45}")
    print(f"  Train accuracy : {train_acc:.4f}  ({train_acc*100:.1f}%)")
    print(f"  Test  accuracy : {test_acc:.4f}  ({test_acc*100:.1f}%)")
    if train_acc - test_acc > 0.10:
        print("  ⚠  Gap > 10% — model may be overfitting")
    print(f"{'='*45}")

    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, test_preds, target_names=["DOWN (0)", "UP (1)"]))

    cm   = confusion_matrix(y_test, test_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["DOWN", "UP"])
    disp.plot(cmap="Greens")
    plt.title("LSTM — Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig("models/saved/lstm_confusion_matrix.png")
    plt.close()
    print("  Confusion matrix saved.")


# ── Step 9: Plot training history ─────────────────────────────────────────────
def plot_history(history):
    """
    history.history is a dict of lists — one value per epoch:
        "loss"         → training loss each epoch
        "val_loss"     → validation loss each epoch
        "accuracy"     → training accuracy each epoch
        "val_accuracy" → validation accuracy each epoch

    If val_loss and train_loss track each other → good generalisation.
    If they diverge (train improves, val gets worse) → overfitting.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["loss"],     label="Train loss")
    ax1.plot(history.history["val_loss"], label="Val loss")
    ax1.set_title("Loss over epochs")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history.history["accuracy"],     label="Train accuracy")
    ax2.plot(history.history["val_accuracy"], label="Val accuracy")
    ax2.set_title("Accuracy over epochs")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("models/saved/lstm_training_history.png")
    plt.close()
    print("  Training history plot saved.")


# ── Step 10: Save model and scaler ────────────────────────────────────────────
def save_model_and_scaler(model, scaler):
    """
    We save BOTH the model and the scaler.

    At inference time, new data must be scaled the SAME WAY before predicting.
    The scaler remembers the min/max it learned from training data.
    If we only saved the model, the inference server wouldn't know how to
    scale new inputs — predictions would be completely wrong.

    joblib.dump() serialises any Python object to a file.
    joblib.load() reads it back — we'll use this in the inference server.
    """
    model.save("models/saved/lstm_model.keras")
    joblib.dump(scaler, "models/saved/lstm_scaler.pkl")
    print("\n  Model  saved → models/saved/lstm_model.keras")
    print("  Scaler saved → models/saved/lstm_scaler.pkl")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    df                               = load_data()
    df                               = create_target(df)
    # Drop any rows where features are NaN (e.g. the first row has no previous
    # close so returns/log_returns are NaN). A single NaN inside a 20-day window
    # propagates through the LSTM cell state and corrupts every gradient update.
    before = len(df)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)  # the first row has no previous close so returns/log_returns are NaN — one NaN inside a 20-day window corrupts the LSTM hidden state
    print(f"  Dropped {before - len(df)} NaN row(s) from features")
    X_scaled, scaler                 = scale_features(df)
    X_windows, y_windows, dates_arr  = make_windows(X_scaled, df)
    X_train, y_train, X_test, y_test = split_windows(X_windows, y_windows, dates_arr)
    model                            = build_model(n_features=len(FEATURES))
    history                          = train_model(model, X_train, y_train, X_test, y_test)
    evaluate(model, X_train, y_train, X_test, y_test)
    plot_history(history)
    save_model_and_scaler(model, scaler)
    print("\nDone. LSTM training complete.")


if __name__ == "__main__":
    main()
