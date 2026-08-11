"""
Verification script -- NOT part of the production pipeline.

Purpose: prove that the `ta` library's indicator calculations produce the
same numbers as our hand-rolled versions in step1_price_features.py, BEFORE
we trust them enough to swap them into the real script.

Round 1 -- RSI: done, accepted. The small gap at the very start of the
series is a known, decaying seeding artifact (NaN-vs-zero handling on the
first row) -- it shrinks toward 0 and is not a real disagreement.

Round 2 -- OBV: done. ta's two-way (up-or-not) definition vs our three-way
(up/down/unchanged) definition creates a permanent, non-decaying gap that
only grows on days with an exact tie in closing price -- rare, and washes
out almost entirely once OBV gets turned into a rate-of-change feature
rather than used at its raw cumulative level. Decision on which version to
keep for OBV specifically is still open.

Round 3 -- ADX: done, accepted. Structurally different again -- ta seeds
its smoothing with a plain average of the first `window` raw values
(stacked across multiple layers), vs our continuous EWM-from-day-1. The
real run showed the gap DOES decay away, just over a longer span (~150-200
rows) than RSI's ~20 -- same kind of benign seeding artifact, just slower.

Round 4 -- SMA: below. Different shape of check entirely. SMA has no
recursive/exponential smoothing step at all -- it's a plain windowed
average, recomputed fresh at every row, with nothing carried over from
one row to the next. Checked ta's actual source (`_sma()` in ta/utils.py):
it's `series.rolling(window=periods, min_periods=periods).mean()` --
exactly what our hand-rolled version does. With no recursion, there's no
seed for the two versions to disagree about in the first place, so this
round is a sanity check, not a real test -- expect the gap to be 0 (or
floating-point noise, ~1e-10) everywhere, not just decaying toward 0.

Run it with:
    pip install ta
    python verify_indicators_ta.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.trend import ADXIndicator, SMAIndicator

TICKER = "AAPL"
START = "2018-01-01"
END = "2024-12-31"


# ---------------------------------------------------------------------------
# The hand-rolled version, copied as-is from step1_price_features.py.
# This is our "known good" reference -- we already verified by hand how
# every line of this works, so it's the thing we're checking the library
# AGAINST, not the other way around.
# ---------------------------------------------------------------------------
def compute_rsi_manual(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# Round 2 -- OBV. Same idea: this is the hand-rolled version from
# step1_price_features.py, copied as-is, treated as "known good."
#
# np.sign(close.diff()) gives THREE outcomes per day: +1 (price up),
# -1 (price down), 0 (price exactly unchanged). .fillna(0) handles day 1,
# where there's no prior close to compare against, by also treating it as
# "no signal" -- contributes 0, not a + or - move.
# ---------------------------------------------------------------------------
def compute_obv_manual(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


# ---------------------------------------------------------------------------
# Round 3 -- ADX. Hand-rolled version, copied as-is from the notebook.
#
# Three stages: (1) True Range + DM smoothed with continuous Wilder EWM,
# (2) +DI/-DI built from those smoothed values, (3) DX (the gap between
# +DI/-DI) smoothed AGAIN with the same continuous EWM to produce ADX.
# Every smoothing step here uses .ewm(adjust=False) started from day 1 --
# there is no separate "plain average" seed anywhere in this version.
# ---------------------------------------------------------------------------
def compute_adx_manual(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return adx


# ---------------------------------------------------------------------------
# Round 4 -- SMA. Hand-rolled version, copied as-is from the notebook.
#
# No smoothing recursion, no seed -- just "average the last N raw closes,"
# recomputed fresh at every row. pandas' .rolling(window=N).mean() does
# exactly that, with min_periods defaulting to N (so the first N-1 rows
# come out NaN, same warm-up shape as the other indicators).
# ---------------------------------------------------------------------------
def compute_sma_manual(close: pd.Series, period: int = 20) -> pd.Series:
    return close.rolling(window=period).mean()


def main():
    print(f"Pulling {TICKER} from Yahoo Finance ({START} to {END})...")
    raw = yf.download(TICKER, start=START, end=END, auto_adjust=True, progress=False)

    # Same MultiIndex flattening fix as step1_price_features.py -- needed
    # here too since we're calling yf.download() the same way.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
        raw.columns.name = None

    close = raw["Close"]

    # --- Version A: our hand-rolled formula (already trusted) ---
    rsi_manual = compute_rsi_manual(close, period=14)

    # --- Version B: the `ta` library, one call ---
    # RSIIndicator takes the closing-price Series plus the lookback window.
    # .rsi() returns the result as its own standalone Series -- nothing
    # gets attached to a DataFrame automatically; that's the explicit,
    # "you do it yourself" style we picked `ta` for over pandas-ta.
    rsi_ta = RSIIndicator(close=close, window=14).rsi()

    # --- Compare them ---
    # Put both side by side in one small table, plus a single "how far
    # apart are these" number, so any mismatch is impossible to miss.
    comparison = pd.DataFrame({
        "manual": rsi_manual,
        "ta_lib": rsi_ta,
    })
    comparison["abs_diff"] = (comparison["manual"] - comparison["ta_lib"]).abs()

    print("\nLast 10 rows, manual vs. ta library:")
    print(comparison.tail(10))

    # Both versions have warm-up NaNs at the start (for the same reason --
    # not enough history yet for a 14-day smoothed average). dropna() here
    # keeps only rows where BOTH have a real number, so we're comparing
    # apples to apples.
    real_rows = comparison.dropna()
    print(f"\nRows compared (both non-NaN): {len(real_rows)}")
    print(f"Max absolute difference across all of them: {real_rows['abs_diff'].max():.6f}")
    print(f"Mean absolute difference: {real_rows['abs_diff'].mean():.6f}")
    print("\nExpect both numbers to be at or extremely close to 0.000000 --")
    print("that's what it looks like when two implementations of the same")
    print("formula agree. Anything meaningfully bigger than that means one")
    print("of the two is doing something subtly different.")

    # --- Where exactly is the disagreement? ---
    # If the two implementations only differ in how they SEED the very
    # first smoothed average (a known, well-documented RSI gotcha), the
    # gap should be biggest right after warm-up and shrink toward 0 the
    # further we get from day 1 -- NOT scattered randomly throughout.
    worst_date = real_rows["abs_diff"].idxmax()
    print(f"\nDate with the single largest disagreement: {worst_date}")
    print(real_rows.loc[[worst_date]])

    print("\nFirst 20 valid (non-NaN) rows -- watch whether abs_diff shrinks")
    print("as we move further away from the start of the series:")
    print(real_rows.head(20))

    # -----------------------------------------------------------------
    # ROUND 2 -- OBV
    # -----------------------------------------------------------------
    # Different shape of test than RSI. RSI is a SMOOTHED AVERAGE -- old
    # data gets exponentially forgotten, so a seeding mismatch decays away.
    # OBV is a RAW RUNNING TOTAL (.cumsum()) -- it never forgets anything.
    # If the two formulas disagree on even one single day, that disagreement
    # gets baked into every value for the rest of the series, permanently.
    # So instead of "does the gap shrink over time," the real question here
    # is "does the gap stay exactly constant, or does it keep growing?"
    print("\n" + "=" * 70)
    print("ROUND 2 -- OBV (On-Balance Volume)")
    print("=" * 70)

    volume = raw["Volume"]

    # --- Version A: our hand-rolled formula (the textbook definition --
    #     up day = +volume, down day = -volume, unchanged day = +0) ---
    obv_manual = compute_obv_manual(close, volume)

    # --- Version B: the `ta` library, one call ---
    # ta's internal formula only checks "is today's close LESS than
    # yesterday's?" If not -- whether price went up OR stayed exactly
    # the same OR it's day 1 with no prior close to compare to -- it
    # falls through to the same "+volume" branch. There's no explicit
    # "unchanged" case the way our np.sign() version has.
    obv_ta = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
    obv_ta.index = close.index  # ta drops the DatetimeIndex name; restore it for alignment/printing

    obv_comparison = pd.DataFrame({
        "manual": obv_manual,
        "ta_lib": obv_ta,
    })
    obv_comparison["abs_diff"] = (obv_comparison["manual"] - obv_comparison["ta_lib"]).abs()

    # OBV has no warm-up period -- both versions produce a real number from
    # day 1 onward, so there's no dropna() step needed here like with RSI.
    print(f"\nFirst day's volume (the amount ta's version adds that ours doesn't,")
    print(f"if day 1 is the ONLY source of disagreement): {volume.iloc[0]:,.0f}")

    print("\nFirst 3 rows -- this is where the day-1 NaN-comparison effect shows up:")
    print(obv_comparison.head(3))

    print("\nLast 3 rows -- compare this gap to the first-day volume above:")
    print(obv_comparison.tail(3))

    flat_days = int((close == close.shift(1)).sum())
    print(f"\nNumber of days where close == previous close exactly: {flat_days}")
    print("(Each one of these, if it occurs after day 1, is a place where our")
    print("version adds 0 and ta's version adds +volume -- another permanent step.)")

    unique_diffs = obv_comparison["abs_diff"].nunique()
    print(f"\nNumber of distinct abs_diff values across the whole series: {unique_diffs}")
    print("If this is 1, the entire gap is just the day-1 effect -- one constant")
    print("offset, never growing. If it's more than 1, one or more flat days")
    print("happened later in the series too, each adding another step.")

    print(f"\nMax absolute difference: {obv_comparison['abs_diff'].max():,.0f}")
    print(f"Min absolute difference: {obv_comparison['abs_diff'].min():,.0f}")
    print("\nUnlike RSI, do NOT expect this to shrink toward 0 over time -- a")
    print("constant (non-shrinking) gap here would actually be the EXPECTED,")
    print("benign outcome, not a red flag. A gap that keeps growing in steps")
    print("would be the thing to look at more closely.")

    # -----------------------------------------------------------------
    # ROUND 3 -- ADX
    # -----------------------------------------------------------------
    # Different shape of test again. Our version smooths everything with
    # one continuous EWM started on day 1. ta's version seeds its running
    # totals with a literal average of the first `window` raw values, and
    # does this at multiple stacked layers (TR/DM smoothing, then DX-to-ADX
    # smoothing again on top of that). Unlike RSI's clean single-layer
    # seeding gap, there's no guarantee this one decays away quickly --
    # need to actually look at how the gap behaves much further into the
    # series, not just the first 20 rows.
    print("\n" + "=" * 70)
    print("ROUND 3 -- ADX (Average Directional Index)")
    print("=" * 70)

    high = raw["High"]
    low = raw["Low"]

    # --- Version A: our hand-rolled formula ---
    adx_manual = compute_adx_manual(high, low, close, period=14)

    # --- Version B: the `ta` library ---
    # ta's ADX fills its warm-up period with literal 0.0, not NaN (it builds
    # the warm-up block with np.zeros and concatenates it onto the front).
    # Our version has real NaN there from the chained diff/shift/ewm calls.
    # That 0-vs-NaN warm-up is a KNOWN, EXPECTED mismatch, not a real
    # disagreement -- we drop those rows using OUR side's NaN mask below,
    # the same way we already decided NaN-vs-0 is a non-issue back in RSI.
    adx_ta = ADXIndicator(high=high, low=low, close=close, window=14).adx()
    adx_ta.index = close.index

    adx_comparison = pd.DataFrame({
        "manual": adx_manual,
        "ta_lib": adx_ta,
    })
    adx_comparison["abs_diff"] = (adx_comparison["manual"] - adx_comparison["ta_lib"]).abs()

    # Only compare rows where OUR version has cleared its own warm-up --
    # ta's side is never NaN by construction, so dropna() on the pair only
    # removes rows based on the manual column, which is exactly what we want.
    adx_real_rows = adx_comparison.dropna()
    print(f"\nRows compared (manual non-NaN): {len(adx_real_rows)}")
    print(f"Max absolute difference: {adx_real_rows['abs_diff'].max():.6f}")
    print(f"Mean absolute difference: {adx_real_rows['abs_diff'].mean():.6f}")

    print("\nFirst 20 valid rows:")
    print(adx_real_rows.head(20))

    # RSI's gap was basically gone within ~20 rows. Check much further out
    # this time, at several checkpoints, to see whether ADX's gap follows
    # the same fast-decay pattern or behaves differently (stays flat,
    # decays much more slowly, or doesn't shrink at all).
    n = len(adx_real_rows)
    checkpoints = [0, min(49, n - 1), min(99, n - 1), min(199, n - 1), n - 1]
    print("\nabs_diff at several checkpoints spread across the whole series")
    print("(row index within the valid/compared rows, not the calendar date):")
    for cp in checkpoints:
        row = adx_real_rows.iloc[cp]
        print(f"  row {cp:>4}  date {adx_real_rows.index[cp].date()}  "
              f"manual={row['manual']:.4f}  ta_lib={row['ta_lib']:.4f}  "
              f"abs_diff={row['abs_diff']:.4f}")

    print("\nIf abs_diff keeps shrinking across these checkpoints -- same")
    print("decaying-seed pattern as RSI, just slower because of the stacked")
    print("smoothing layers. If it flattens out at some non-zero level or")
    print("keeps growing, that's a real, lasting disagreement worth digging")
    print("into further before trusting the ta version for ADX specifically.")

    # -----------------------------------------------------------------
    # ROUND 4 -- SMA
    # -----------------------------------------------------------------
    # No recursion, no seed, nothing to disagree about -- this round exists
    # to confirm that, not to hunt for a subtle mismatch. Expect max_diff to
    # be exactly 0 or down at floating-point noise (~1e-10), with no shrink-
    # over-time pattern needed because there's nothing to decay.
    print("\n" + "=" * 70)
    print("ROUND 4 -- SMA (Simple Moving Average)")
    print("=" * 70)

    sma_manual = compute_sma_manual(close, period=20)
    sma_ta = SMAIndicator(close=close, window=20).sma_indicator()
    sma_ta.index = close.index

    sma_comparison = pd.DataFrame({
        "manual": sma_manual,
        "ta_lib": sma_ta,
    })
    sma_comparison["abs_diff"] = (sma_comparison["manual"] - sma_comparison["ta_lib"]).abs()

    sma_real_rows = sma_comparison.dropna()
    print(f"\nRows compared (both non-NaN): {len(sma_real_rows)}")
    print(f"Max absolute difference: {sma_real_rows['abs_diff'].max():.10f}")
    print(f"Mean absolute difference: {sma_real_rows['abs_diff'].mean():.10f}")

    print("\nFirst 5 valid rows:")
    print(sma_real_rows.head(5))
    print("\nLast 5 rows:")
    print(sma_real_rows.tail(5))

    print("\nExpect max_diff to be 0.0000000000 (or floating-point noise --")
    print("anything like 1e-10 or smaller is just float rounding, not a real")
    print("disagreement). Anything bigger than that means something is")
    print("actually different about the two windows/formulas -- worth a")
    print("second look before accepting.")


if __name__ == "__main__":
    main()
