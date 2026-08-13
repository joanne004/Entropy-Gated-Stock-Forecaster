"""
Step 1 -- Extract, Transform, Load: Price Data and Technical Indicators (X_t)

Production pipeline script. Pulls OHLCV price data from Yahoo Finance and
produces X_t -- the technical/price feature vector (SMA_20, RSI_14, OBV,
ADX_14) that later gets fused with sentiment (S_t) into F_t.

Runs through 7 stages: Extract, Validate, Transform, Clean, Assemble, Plot,
Load. See README.md in this folder for the full walkthrough.

All four indicators are computed with the `ta` library rather than by hand.
Each one was checked against a hand-rolled reference implementation first --
see verify_indicators_ta.py for the full comparison and the reasoning behind
every accept/reject decision:

  - SMA_20: exact match to the hand-rolled version, no caveats.
  - RSI_14: tiny seeding-related gap in the first ~20 rows that decays to
    ~0 -- a known, benign artifact, not a real disagreement. Accepted.
  - ADX_14: same kind of seeding gap as RSI, just decays more slowly
    (~150-200 rows) because of ta's stacked smoothing layers. Accepted.
  - OBV: structurally different from the textbook three-way definition --
    ta treats a "no price change" day as +volume instead of 0. Decided to
    accept ta's two-way version for consistency with the rest of this
    pipeline; the practical impact is tiny (rare exact-tie days) and washes
    out once OBV becomes a rate-of-change feature downstream.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from ta.trend import SMAIndicator, ADXIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volume import OnBalanceVolumeIndicator

TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"]
START  = "2018-01-01"
END    = datetime.today().strftime("%Y-%m-%d")   # always pulls up to today
SMA_WINDOW = 20
RSI_WINDOW = 14
ADX_WINDOW = 14

DATA_DIR = "data"

FEATURE_COLS = ["SMA_20", "RSI_14", "OBV", "ADX_14", "MACD", "VIX"]


def _mask_adx_warmup(adx: pd.Series) -> pd.Series:
    """ta's ADXIndicator fills its warm-up period with literal 0.0 instead
    of NaN (confirmed in verify_indicators_ta.py, Round 3 -- it builds the
    warm-up block with np.zeros). Left alone, those fake zeros survive the
    Clean stage's dropna() untouched and contaminate the start of X_t and
    the sanity-check plot. Find that leading all-zero run and convert it to
    real NaN so Clean actually drops it, same as every other indicator's
    warm-up period."""
    adx = adx.copy()
    nonzero = (adx != 0.0).to_numpy()
    if not nonzero.any():
        return adx  # nothing real anywhere -- leave as is
    first_real_pos = nonzero.argmax()
    adx.iloc[:first_real_pos] = float("nan")
    return adx


# ---------------------------------------------------------------------------
# Stage 1 -- Extract
# ---------------------------------------------------------------------------
def extract(ticker: str, start: str, end: str) -> pd.DataFrame:
    print(f"Pulling {ticker} from Yahoo Finance ({start} to {end})...")
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    # yf.download() returns a MultiIndex column header (price field, ticker)
    # for recent yfinance versions even with a single ticker -- flatten it
    # back down to plain column names like "Close", "Volume", etc.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
        raw.columns.name = None

    # Download VIX (CBOE Volatility Index) and merge on date.
    # VIX measures the market's expectation of 30-day volatility.
    # High VIX = fear/uncertainty → predictions less reliable.
    # Low VIX  = calm market    → trends more likely to hold.
    print(f"Pulling VIX from Yahoo Finance...")
    vix = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
        vix.columns.name = None
    vix = vix[["Close"]].rename(columns={"Close": "VIX"})

    # Left join — keeps all AAPL trading days, fills any VIX gaps forward
    raw = raw.join(vix, how="left")
    raw["VIX"] = raw["VIX"].ffill()

    return raw


# ---------------------------------------------------------------------------
# Stage 2 -- Validate
# ---------------------------------------------------------------------------
def validate(df: pd.DataFrame) -> None:
    """Look before you trust it -- basic data-quality checks before any
    feature gets built on top of this raw pull."""
    n_missing = int(df.isna().sum().sum())
    n_duplicates = int(df.index.duplicated().sum())
    gaps = df.index.to_series().diff().dt.days.dropna()
    max_gap = int(gaps.max()) if not gaps.empty else 0

    print(f"Missing values: {n_missing}")
    print(f"Duplicate dates: {n_duplicates}")
    print(f"Longest gap between consecutive rows: {max_gap} day(s) (long weekends/holidays expected)")

    if n_missing or n_duplicates:
        raise ValueError(
            "Data quality check failed (missing values or duplicate dates) -- "
            "inspect the raw pull before trusting any feature built on it."
        )


# ---------------------------------------------------------------------------
# Stage 3 -- Transform
# ---------------------------------------------------------------------------
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 4 technical indicators that make up X_t, via the `ta`
    library. See the module docstring above for why each one is trusted."""
    out = df.copy()

    out["SMA_20"] = SMAIndicator(close=out["Close"], window=SMA_WINDOW).sma_indicator()
    out["RSI_14"] = RSIIndicator(close=out["Close"], window=RSI_WINDOW).rsi()
    out["OBV"] = OnBalanceVolumeIndicator(close=out["Close"], volume=out["Volume"]).on_balance_volume()
    adx_raw = ADXIndicator(
        high=out["High"], low=out["Low"], close=out["Close"], window=ADX_WINDOW
    ).adx()
    out["ADX_14"] = _mask_adx_warmup(adx_raw)
    # MACD histogram = MACD line minus Signal line
    # Positive = short-term momentum above long-term (bullish pressure building)
    # Negative = short-term momentum below long-term (bearish pressure building)
    # Crossing zero = potential reversal signal
    out["MACD"] = MACD(close=out["Close"]).macd_diff()
    # VIX is already in the dataframe from extract() — no computation needed

    return out


# ---------------------------------------------------------------------------
# Stage 4 -- Clean
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Drop the warm-up rows where one or more indicators are still NaN.
    We pulled several years of data, so losing the first few weeks costs
    almost nothing."""
    print(f"Rows before dropping warm-up NaNs: {len(df)}")
    df_clean = df.dropna(subset=feature_cols).copy()
    print(f"Rows after: {len(df_clean)}")
    return df_clean


# ---------------------------------------------------------------------------
# Stage 5 -- Assemble
# ---------------------------------------------------------------------------
def assemble(df: pd.DataFrame, feature_cols: list, ticker: str) -> pd.DataFrame:
    df = df.copy()
    df["returns"]     = df["Close"].pct_change()
    df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))
    keep = ["Open", "High", "Low", "Close", "Volume"] + feature_cols + ["returns", "log_returns"]
    x_t = df[keep].copy()
    x_t.columns = [c.lower() for c in x_t.columns]
    x_t.index.name = "date"
    x_t["ticker"] = ticker
    return x_t


# ---------------------------------------------------------------------------
# Stage 6 -- Plot
# ---------------------------------------------------------------------------
def plot(df: pd.DataFrame, ticker: str, path: str) -> None:
    """Save a quick visual sanity check -- not displayed interactively, so
    this works the same whether the script is run from a terminal, VS Code,
    or anywhere else."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(df.index, df["Close"], label="Close")
    axes[0].plot(df.index, df["SMA_20"], label="SMA_20")
    axes[0].set_title(f"{ticker} -- Close vs SMA_20")
    axes[0].legend()

    axes[1].plot(df.index, df["RSI_14"], color="darkorange")
    axes[1].axhline(70, color="red", linestyle="--", linewidth=0.8)
    axes[1].axhline(30, color="green", linestyle="--", linewidth=0.8)
    axes[1].set_title("RSI_14 (70 = overbought, 30 = oversold)")

    axes[2].plot(df.index, df["ADX_14"], color="purple")
    axes[2].axhline(25, color="gray", linestyle="--", linewidth=0.8)
    axes[2].set_title("ADX_14 (above ~25 = trending market)")

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Sanity-check plot saved to {path }")


# ---------------------------------------------------------------------------
# Stage 7 -- Load
# ---------------------------------------------------------------------------
def load(x_t: pd.DataFrame, path: str) -> None:
    x_t.to_csv(path)
    print(f"X_t saved to {path}")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    for ticker in TICKERS:
        csv_path  = os.path.join(DATA_DIR, f"X_t_{ticker}.csv")
        plot_path = os.path.join(DATA_DIR, f"X_t_{ticker}_sanity_check.png")

        print(f"\n{'='*55}")
        print(f"  Processing {ticker}")
        print(f"{'='*55}")

        print("\n--- Stage 1: Extract ---")
        raw = extract(ticker, START, END)

        print("\n--- Stage 2: Validate ---")
        validate(raw)

        print("\n--- Stage 3: Transform ---")
        transformed = transform(raw)

        print("\n--- Stage 4: Clean ---")
        cleaned = clean(transformed, FEATURE_COLS)

        print("\n--- Stage 5: Assemble ---")
        x_t = assemble(cleaned, FEATURE_COLS, ticker)
        print(f"\nX_t shape: {x_t.shape}")
        print(x_t.tail(3))

        print("\n--- Stage 6: Plot ---")
        plot(cleaned, ticker, plot_path)

        print("\n--- Stage 7: Load ---")
        load(x_t, csv_path)

    print(f"\n{'='*55}")
    print(f"  All {len(TICKERS)} tickers processed.")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
