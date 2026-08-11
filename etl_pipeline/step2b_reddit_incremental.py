"""
Step 2b -- Reddit Incremental Update
=====================================
Fetches only NEW Reddit posts from the last date in the existing CSV up to today.
Merges with existing data and saves. Run this whenever you want to update sentiment.

This avoids re-fetching 2018-2024 data that is already in the checkpoint.
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz
import json

TICKER     = "AAPL"
KEYWORDS   = ["AAPL", "Apple"]
SUBREDDITS = ["stocks", "investing", "wallstreetbets"]

EASTERN           = pytz.timezone("US/Eastern")
MARKET_CLOSE_HOUR = 16

BASE_URL    = "https://arctic-shift.photon-reddit.com/api/posts/search"
DATA_DIR    = "data"
CSV_PATH    = os.path.join(DATA_DIR, f"S_t_raw_{TICKER}_reddit.csv")


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
    combined = (post.get("title", "") + " " + post.get("selftext", "")).lower()
    return any(kw.lower() in combined for kw in KEYWORDS)


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
        except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
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
    # ── Find last date in existing CSV ────────────────────────────────────────
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found. Run step2a_reddit_raw.py first.")
        return

    existing = pd.read_csv(CSV_PATH)
    last_date = pd.to_datetime(existing["trading_day"]).max()
    # start fetching from the day after the last date we already have
    start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    end_date   = datetime.today().strftime("%Y-%m-%d")

    print(f"Existing data ends at: {last_date.date()}")
    print(f"Fetching new posts from {start_date} to {end_date}")
    print()

    if start_date >= end_date:
        print("Data is already up to date. Nothing to fetch.")
        return

    # ── Fetch new posts for each subreddit ────────────────────────────────────
    new_rows = []
    for subreddit in SUBREDDITS:
        print(f"--- Fetching r/{subreddit} ({start_date} → {end_date}) ---")
        posts = collect_subreddit(subreddit, start_date, end_date)
        print(f"  Pulled {len(posts)} raw posts")

        kept = 0
        for post in posts:
            if not is_relevant(post):
                continue
            selftext = post.get("selftext", "") or ""
            if selftext == "[removed]":
                selftext = ""
            new_rows.append({
                "trading_day": assign_trading_day(post["created_utc"]),
                "id":          post.get("id", ""),
                "title":       post.get("title", ""),
                "selftext":    selftext,
                "created_utc": post["created_utc"],
                "score":       post.get("score", 0),
                "subreddit":   subreddit,
                "source":      "reddit",
            })
            kept += 1
        print(f"  Kept {kept} relevant posts")

    # ── Merge with existing data and save ─────────────────────────────────────
    if not new_rows:
        print("\nNo new posts found — Arctic Shift may not have data past this date yet.")
        print("The system will use existing data up to", last_date.date())
        return

    new_df   = pd.DataFrame(new_rows)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["id"]).sort_values("trading_day")
    combined.to_csv(CSV_PATH, index=False)

    print(f"\nDone. Combined dataset: {len(combined)} posts")
    print(f"  Old range: 2018-01-01 → {last_date.date()}")
    print(f"  New range: 2018-01-01 → {pd.to_datetime(combined['trading_day']).max().date()}")
    print(f"\nNext step: run python step3_sentiment_finbert.py")


if __name__ == "__main__":
    main()
