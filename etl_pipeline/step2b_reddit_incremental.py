"""
Step 2b -- Reddit Incremental Update
=====================================
Fetches only NEW Reddit posts from the last date in the existing CSV up to today.
Merges with existing data and saves. Run this whenever you want to update sentiment.

If no existing CSV is found, performs a full fetch from 2018 (same as step2a).

Usage:
  python step2b_reddit_incremental.py          # defaults to AAPL
  python step2b_reddit_incremental.py MSFT
  python step2b_reddit_incremental.py NVDA
"""

import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz

# ── Per-ticker config ─────────────────────────────────────────────────────────
TICKER_CONFIGS = {
    "AAPL": {
        "keywords"   : ["AAPL", "Apple"],
        "subreddits" : ["stocks", "investing", "wallstreetbets"],
    },
    "MSFT": {
        "keywords"   : ["MSFT", "Microsoft"],
        "subreddits" : ["stocks", "investing", "wallstreetbets", "microsoft"],
    },
    "NVDA": {
        "keywords"   : ["NVDA", "Nvidia", "NVIDIA"],
        "subreddits" : ["stocks", "investing", "wallstreetbets", "nvidia"],
    },
}

TICKER = sys.argv[1].upper() if len(sys.argv) > 1 else "AAPL"
if TICKER not in TICKER_CONFIGS:
    print(f"Unknown ticker '{TICKER}'. Supported: {list(TICKER_CONFIGS.keys())}")
    sys.exit(1)

KEYWORDS   = TICKER_CONFIGS[TICKER]["keywords"]
SUBREDDITS = TICKER_CONFIGS[TICKER]["subreddits"]

EASTERN           = pytz.timezone("US/Eastern")
MARKET_CLOSE_HOUR = 16

BASE_URL  = "https://arctic-shift.photon-reddit.com/api/posts/search"
DATA_DIR  = "data"
CSV_PATH  = os.path.join(DATA_DIR, f"S_t_raw_{TICKER}_reddit.csv")
FULL_FETCH_START = "2018-01-01"  # used when no existing CSV found


def date_to_ts(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.replace(tzinfo=pytz.utc).timestamp())


def assign_trading_day(created_utc: int) -> str:
    dt_utc = datetime.utcfromtimestamp(created_utc).replace(tzinfo=pytz.utc)
    dt_et  = dt_utc.astimezone(EASTERN)
    if dt_et.hour < MARKET_CLOSE_HOUR:
        trading_day = dt_et.date()
    else:
        trading_day = dt_et.date() + timedelta(days=1)
    return trading_day.strftime("%Y-%m-%d")


def is_relevant(post: dict) -> bool:
    title = (post.get("title", "") or "").lower()
    return any(kw.lower() in title for kw in KEYWORDS)


def fetch_posts(subreddit: str, after_ts: int, before_ts: int) -> list:
    params = {
        "subreddit": subreddit,
        "after":     after_ts,
        "before":    before_ts,
        "limit":     "auto",
        "sort":      "asc",
        "fields":    "id,title,selftext,created_utc,score,subreddit",
    }
    for attempt in range(4):
        try:
            response = requests.get(BASE_URL, params=params, timeout=60)
            if response.status_code in (422, 429, 500, 503):
                wait = 15 * (attempt + 1)
                print(f"  Server {response.status_code}, retrying in {wait}s...")
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json().get("data", [])
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout,
                requests.exceptions.ChunkedEncodingError):
            wait = 30 * (attempt + 1)
            print(f"  Connection dropped, retrying in {wait}s...")
            time.sleep(wait)
    print("  Failed after 4 retries, skipping batch.")
    return []


def collect_subreddit(subreddit: str, start_date: str, end_date: str) -> list:
    all_posts = []
    after_ts  = date_to_ts(start_date)
    end_ts    = date_to_ts(end_date)

    while after_ts < end_ts:
        batch = fetch_posts(subreddit, after_ts, end_ts)
        if not batch:
            break
        all_posts.extend(batch)
        after_ts = batch[-1]["created_utc"] + 1   # advance past last post
        print(f"  r/{subreddit}: {len(all_posts)} posts so far...")
        if len(batch) < 100:
            break   # reached the end of available data

    return all_posts


def main():
    print(f"\nRunning step2b for {TICKER}")
    print(f"  Keywords   : {KEYWORDS}")
    print(f"  Subreddits : {SUBREDDITS}")

    end_date = datetime.today().strftime("%Y-%m-%d")

    # ── Load existing CSV (may be partial from a previous interrupted run) ────
    if os.path.exists(CSV_PATH):
        existing = pd.read_csv(CSV_PATH)
        global_last_date   = pd.to_datetime(existing["trading_day"]).max()
        incremental_start  = (global_last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        already_fetched    = set(existing["subreddit"].unique()) if "subreddit" in existing.columns else set()
        print(f"\nExisting CSV: {len(existing)} posts, ends {global_last_date.date()}")
        if already_fetched:
            print(f"  Subreddits already in CSV: {sorted(already_fetched)}")
    else:
        existing           = pd.DataFrame()
        incremental_start  = None
        already_fetched    = set()
        print(f"\nNo existing CSV — full fetch from {FULL_FETCH_START}")

    # ── Fetch per subreddit ───────────────────────────────────────────────────
    # Each subreddit is saved to disk as soon as it completes.
    # On a re-run after a crash, subreddits already in the CSV are skipped
    # (for full-fetch mode) or only updated with new dates (incremental mode).
    any_new = False

    for subreddit in SUBREDDITS:
        if subreddit in already_fetched:
            # Incremental: only fetch dates newer than what we already have
            fetch_start = incremental_start
            if fetch_start >= end_date:
                print(f"\n  r/{subreddit}: already up to date, skipping")
                continue
            mode = "incremental"
        else:
            # Full fetch: this subreddit is missing from the CSV entirely
            fetch_start = FULL_FETCH_START
            mode = "full fetch"

        print(f"\n--- r/{subreddit} [{mode}] ({fetch_start} → {end_date}) ---")
        posts = collect_subreddit(subreddit, fetch_start, end_date)
        print(f"  Pulled {len(posts)} raw posts")

        subreddit_rows = []
        for post in posts:
            if not is_relevant(post):
                continue
            selftext = post.get("selftext", "") or ""
            if selftext == "[removed]":
                selftext = ""
            subreddit_rows.append({
                "trading_day": assign_trading_day(post["created_utc"]),
                "id":          post.get("id", ""),
                "title":       post.get("title", ""),
                "selftext":    selftext,
                "created_utc": post["created_utc"],
                "score":       post.get("score", 0),
                "subreddit":   subreddit,
                "source":      "reddit",
            })
        print(f"  Kept {len(subreddit_rows)} relevant posts")

        # ── Save incrementally after each subreddit ────────────────────────────
        # Even if a later subreddit crashes, this one's data is safe on disk.
        if subreddit_rows:
            new_df   = pd.DataFrame(subreddit_rows)
            combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
            combined = combined.drop_duplicates(subset=["id"]).sort_values("trading_day")
            combined.to_csv(CSV_PATH, index=False)
            # Update state for next iteration
            existing        = combined
            already_fetched = set(existing["subreddit"].unique())
            any_new         = True
            print(f"  ✓ Saved — {len(combined)} posts total in CSV")

    # ── Final summary ─────────────────────────────────────────────────────────
    if not any_new:
        print("\nData is already up to date. Nothing to fetch.")
        return

    final = pd.read_csv(CSV_PATH)
    print(f"\nDone. {TICKER} dataset: {len(final)} posts total")
    print(f"  Range: {pd.to_datetime(final['trading_day']).min().date()} → {pd.to_datetime(final['trading_day']).max().date()}")
    print(f"\nNext step: python step3_sentiment_finbert.py {TICKER}")


if __name__ == "__main__":
    main()
