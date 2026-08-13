"""
Model Diagnosis Script
======================
Before tuning, answer: are the models failing because of:
  A) Weak/absent features (data problem)
  B) Poor model configuration (tuning problem)
  C) Out-of-distribution test period (regime shift)

Run with:
  python diagnose_models.py
"""

import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
DB_URL = f"postgresql://stockuser:stockpass@{POSTGRES_HOST}:5432/stockdb"
engine = create_engine(DB_URL)

FEATURES = [
    "returns", "log_returns", "rsi_14", "adx_14", "macd", "vix",
    "sentiment", "entropy", "alpha", "post_count",
]

# ─────────────────────────────────────────────────────────────────────────────
def diagnose(ticker: str):
    print(f"\n{'='*65}")
    print(f"  DIAGNOSIS — {ticker}")
    print(f"{'='*65}")

    df = pd.read_sql(
        f"SELECT * FROM fused_features WHERE ticker='{ticker}' ORDER BY date ASC",
        engine,
    )
    df["date"] = pd.to_datetime(df["date"])
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna(subset=["target"] + FEATURES).reset_index(drop=True)

    cutoff_idx = int(len(df) * 0.8)
    train = df.iloc[:cutoff_idx]
    test  = df.iloc[cutoff_idx:]
    print(f"  Train: {len(train)} rows  ({train['date'].min().date()} → {train['date'].max().date()})")
    print(f"  Test:  {len(test)} rows  ({test['date'].min().date()} → {test['date'].max().date()})")

    X_train = train[FEATURES].fillna(0).values
    y_train = train["target"].values
    X_test  = test[FEATURES].fillna(0).values
    y_test  = test["target"].values

    # ── 1. Baselines ─────────────────────────────────────────────────────────
    print(f"\n  [1] BASELINES (what any model must beat)")
    always_up  = accuracy_score(y_test, np.ones(len(y_test)))
    coin_flip  = 0.50
    print(f"     Always-UP accuracy : {always_up:.4f} ({always_up*100:.1f}%)")
    print(f"     Coin-flip accuracy : {coin_flip:.4f} (50.0%)")
    print(f"     → Any model not beating {max(always_up, coin_flip)*100:.1f}% is not learning")

    # ── 2. Feature variance ───────────────────────────────────────────────────
    print(f"\n  [2] FEATURE VARIANCE (zero variance = model cannot learn)")
    dead = []
    for f in FEATURES:
        v = train[f].var()
        flag = "  ⚠ DEAD FEATURE" if v < 0.0001 else ""
        print(f"     {f:<15} var={v:.6f}{flag}")
        if v < 0.0001:
            dead.append(f)
    if dead:
        print(f"\n     ⚠ Dead features detected: {dead}")
        print(f"     These carry zero information — remove them for this ticker")

    # ── 3. Correlation + p-value ──────────────────────────────────────────────
    print(f"\n  [3] LINEAR CORRELATION with next-day direction (train set)")
    print(f"     (point-biserial r + p-value)")
    rows = []
    for f in FEATURES:
        r, p = stats.pointbiserialr(train[f].fillna(0), y_train)
        rows.append((f, r, p))
    rows.sort(key=lambda x: abs(x[1]), reverse=True)
    any_sig = False
    for f, r, p in rows:
        sig = "✓ significant" if p < 0.05 else "✗ not significant"
        if p < 0.05:
            any_sig = True
        print(f"     {f:<15} r={r:+.4f}  p={p:.4f}  {sig}")
    if not any_sig:
        print(f"\n     ⚠ No feature has significant LINEAR correlation → EMH consistent")
    else:
        print(f"\n     ✓ Some features have significant (weak) linear signal")

    # ── 4. Mutual information (non-linear) ────────────────────────────────────
    print(f"\n  [4] MUTUAL INFORMATION (captures non-linear dependence, train set)")
    mi = mutual_info_classif(X_train, y_train, random_state=42)
    mi_pairs = sorted(zip(FEATURES, mi), key=lambda x: -x[1])
    total_mi = sum(mi)
    print(f"     Total MI across all features: {total_mi:.5f}")
    for f, m in mi_pairs:
        flag = "  ✓ some signal" if m > 0.002 else "  ≈ 0"
        print(f"     {f:<15} MI={m:.5f}{flag}")
    if total_mi < 0.05:
        print(f"\n     ⚠ Very low total MI → features contain minimal non-linear signal too")

    # ── 5. Target autocorrelation ─────────────────────────────────────────────
    print(f"\n  [5] TARGET AUTOCORRELATION (is there serial pattern to exploit?)")
    tgt = train["target"].values.astype(float)
    any_autocorr = False
    for lag in [1, 2, 3, 5, 10]:
        r, p = stats.pearsonr(tgt[lag:], tgt[:-lag])
        sig = "✓" if p < 0.05 else "✗"
        if p < 0.05:
            any_autocorr = True
        print(f"     lag={lag:<3}  r={r:+.4f}  p={p:.4f}  {sig}")
    if any_autocorr:
        print(f"     ✓ Serial pattern exists — LSTM should theoretically exploit this")
    else:
        print(f"     ✗ No serial pattern — LSTM has nothing sequential to learn")

    # ── 6. Regime check: rolling test accuracy ────────────────────────────────
    print(f"\n  [6] REGIME CHECK — is test period unusually hard?")
    test["year"] = test["date"].dt.year
    test["up_rate"] = test["target"]
    by_year = test.groupby("year")["target"].agg(["mean","count"])
    by_year.columns = ["up_rate","n_days"]
    print(f"     Year-by-year UP% in test set (tells you if regime shifted):")
    for yr, row in by_year.iterrows():
        print(f"     {yr}: UP={row['up_rate']:.1%}  ({int(row['n_days'])} trading days)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  ── SUMMARY for {ticker} ──")
    if dead:
        print(f"  → Problem A (data): {len(dead)} dead features — remove for this ticker")
    if not any_sig and total_mi < 0.05:
        print(f"  → Problem A (data): features have minimal linear AND non-linear signal")
        print(f"     Models performing near-50% is consistent with this finding")
        print(f"     Tuning won't fix a data problem — EMH is the likely explanation")
    elif any_sig or total_mi > 0.01:
        print(f"  → Signal exists but is weak — tuning may extract more of it")
        print(f"  → Recommend: XGBoost logloss metric + LSTM architecture changes")
    if any_autocorr:
        print(f"  → Serial structure exists — LSTM window tuning could help")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for ticker in ["AAPL", "MSFT"]:
        diagnose(ticker)

    print(f"\n\n{'='*65}")
    print(f"  NEXT STEP")
    print(f"{'='*65}")
    print(f"  If [3]+[4] show signal exists → run improve_models.py")
    print(f"  If [3]+[4] show no signal → results are valid, present as-is")
    print(f"  If [6] shows regime shift → consider walk-forward validation")
