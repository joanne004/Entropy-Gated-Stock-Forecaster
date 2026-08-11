"""
LSTM Diagnostic Script
======================
Checks for data quality issues that could cause LSTM to fail:
  1. NaN / Inf values in features
  2. Features out of [0,1] range after scaling (test set clipping)
  3. Feature-target correlations (how much signal is actually there?)
  4. Autocorrelation of features (do past values predict future ones?)
  5. Window label distribution (are train windows balanced?)
"""

import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
DB_URL        = f"postgresql://stockuser:stockpass@{POSTGRES_HOST}:5432/stockdb"
engine        = create_engine(DB_URL)

FEATURES = ["returns", "log_returns", "rsi_14", "adx_14",
            "sentiment", "entropy", "alpha", "post_count"]
WINDOW = 20

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_sql("SELECT * FROM fused_features ORDER BY date ASC", engine)
df["date"] = pd.to_datetime(df["date"])
df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
df = df.dropna(subset=["target"])

# ── 1. NaN in raw features ────────────────────────────────────────────────────
print("=" * 55)
print("1. NaN COUNTS PER FEATURE (raw)")
print("=" * 55)
nan_counts = df[FEATURES].isna().sum()
print(nan_counts)
if nan_counts.sum() > 0:
    print("  ⚠  NaNs found — these will silently break LSTM training")
else:
    print("  ✓  No NaNs")

# ── 2. Post_count = 0 days ────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("2. DAYS WITH 0 REDDIT POSTS")
print("=" * 55)
zero_post = df[df["post_count"] == 0]
print(f"  Count: {len(zero_post)} / {len(df)} days ({len(zero_post)/len(df):.1%})")
if len(zero_post) > 0:
    print("  Sample entropy/sentiment on 0-post days:")
    print(zero_post[["date", "entropy", "sentiment", "alpha"]].head(5).to_string(index=False))

# ── 3. After scaling: NaN / Inf / out-of-range ───────────────────────────────
print("\n" + "=" * 55)
print("3. AFTER MINMAXSCALER")
print("=" * 55)
split_mask = df["date"] < "2023-01-01"
X_all = df[FEATURES].values.copy().astype(float)
scaler = MinMaxScaler()
X_all[split_mask]  = scaler.fit_transform(X_all[split_mask])
X_all[~split_mask] = scaler.transform(X_all[~split_mask])

nan_post = np.isnan(X_all).sum()
inf_post = np.isinf(X_all).sum()
out_of_range = ((X_all[~split_mask] < 0) | (X_all[~split_mask] > 1)).sum()
print(f"  NaN after scaling : {nan_post}")
print(f"  Inf after scaling : {inf_post}")
print(f"  Test values outside [0,1]: {out_of_range}")

if nan_post > 0 or inf_post > 0:
    print("  ⚠  FOUND — these will corrupt LSTM hidden state")
else:
    print("  ✓  Clean")

# Per-feature range after scaling
print("\n  Per-feature range after scaling (full dataset):")
for i, f in enumerate(FEATURES):
    col = X_all[:, i]
    print(f"    {f:15s}: min={col.min():.4f}  max={col.max():.4f}  "
          f"mean={col.mean():.4f}  std={col.std():.4f}")

# ── 4. Feature-target correlation ────────────────────────────────────────────
print("\n" + "=" * 55)
print("4. FEATURE → TARGET CORRELATION")
print("   (how much signal does each feature carry?)")
print("=" * 55)
train_df = df[split_mask.values]
for f in FEATURES:
    corr = train_df[f].corr(train_df["target"])
    bar = "█" * int(abs(corr) * 100)
    print(f"  {f:15s}: {corr:+.4f}  {bar}")

# ── 5. Autocorrelation of returns (the core signal) ──────────────────────────
print("\n" + "=" * 55)
print("5. AUTOCORRELATION OF RETURNS (lag 1–5)")
print("   (if near 0, past returns don't predict future — LSTM will struggle)")
print("=" * 55)
returns = df["returns"]
for lag in range(1, 6):
    ac = returns.autocorr(lag=lag)
    print(f"  Lag {lag}: {ac:+.4f}")

# ── 6. Window label balance ───────────────────────────────────────────────────
print("\n" + "=" * 55)
print("6. WINDOW LABEL BALANCE")
print("=" * 55)
y_all   = df["target"].values
dates   = df["date"].values
y_wins, d_wins = [], []
for i in range(WINDOW, len(X_all)):
    y_wins.append(y_all[i])
    d_wins.append(dates[i])
y_wins  = np.array(y_wins)
d_wins  = pd.to_datetime(d_wins)
train_m = d_wins < pd.Timestamp("2023-01-01")
up_tr   = y_wins[train_m].sum()
dn_tr   = (1 - y_wins[train_m]).sum()
up_te   = y_wins[~train_m].sum()
dn_te   = (1 - y_wins[~train_m]).sum()
print(f"  Train — UP: {up_tr} ({up_tr/(up_tr+dn_tr):.1%})  DOWN: {dn_tr} ({dn_tr/(up_tr+dn_tr):.1%})")
print(f"  Test  — UP: {up_te} ({up_te/(up_te+dn_te):.1%})  DOWN: {dn_te} ({dn_te/(up_te+dn_te):.1%})")

print("\nDiagnostic complete.")
