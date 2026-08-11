"""
Step 2a -- Extract, Transform, Load: Reddit Posts (Raw Sentiment Input)

Pulls historical Reddit posts from the Arctic Shift public archive
(https://arctic-shift.photon-reddit.com) for three subreddits:
r/stocks, r/investing, r/wallstreetbets.

Why Arctic Shift and not the Reddit API directly?
  - Reddit's own API now requires OAuth and has strict rate limits
  - Arctic Shift is a free, no-credentials-needed archive of historical posts
  - It supports date-range filtering which is exactly what we need

Pipeline steps:
  1. For each subreddit, paginate through Arctic Shift in 100-post windows
     across the full date range (2018-2024)
  2. Filter locally -- keep only posts where title or selftext mentions
     "AAPL" or "Apple". We don't use the API's keyword filter because it
     times out on active subreddits when combined with a date range.
  3. Clean selftext -- replace "[removed]" with empty string so FinBERT
     doesn't process it in Step 3
  4. Assign each post to a trading day using a 4pm ET hard cutoff:
     posts before 4pm ET --> that calendar day
     posts at/after 4pm ET --> next calendar day

Output: data/S_t_raw_AAPL_reddit.csv
  Columns: trading_day, id, title, selftext, created_utc, score,
           subreddit, source
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz
import json


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TICKER     = "AAPL"
KEYWORDS   = ["AAPL", "Apple"]
SUBREDDITS = ["stocks", "investing", "wallstreetbets"]

START_DATE = "2018-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")   # always pulls up to today

EASTERN           = pytz.timezone("US/Eastern")
MARKET_CLOSE_HOUR = 16  # 4pm ET

BASE_URL = "https://arctic-shift.photon-reddit.com/api/posts/search"
DATA_DIR        = "data"
CSV_PATH        = os.path.join(DATA_DIR, f"S_t_raw_{TICKER}_reddit.csv")
CHECKPOINT_PATH = os.path.join(DATA_DIR, f"S_t_raw_{TICKER}_reddit_checkpoint.csv")
PROGRESS_PATH   = os.path.join(DATA_DIR, f"S_t_raw_{TICKER}_reddit_progress.json")

# ---------------------------------------------------------------------------
# Helper -- date string to Unix timestamp
# ---------------------------------------------------------------------------
def date_to_ts(date_str: str) -> int:
    """Convert a 'YYYY-MM-DD' string to a UTC Unix timestamp (midnight UTC).

    We attach tzinfo=pytz.utc before calling .timestamp() so Python doesn't
    silently assume your local machine's timezone and produce a wrong number."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.replace(tzinfo=pytz.utc).timestamp())


# ---------------------------------------------------------------------------
# Stage 1 -- Fetch (single API call)
# ---------------------------------------------------------------------------
def fetch_posts(subreddit: str, after_ts: int, before_ts: int,
                limit: str = "auto") -> list:
    """Call Arctic Shift once and return up to `limit` posts.

    Parameters
    ----------
    subreddit : str   e.g. "stocks"
    after_ts  : int   Unix timestamp -- return posts after this moment
    before_ts : int   Unix timestamp -- return posts before this moment
    limit     : int   max posts per call (Arctic Shift cap is 100)

    Returns a list of post dictionaries, or an empty list if none found."""
    params = {
        "subreddit": subreddit,
        "after":     after_ts,
        "before":    before_ts,
        "limit":     limit,
        "sort":      "asc",
        "fields":    "id,title,selftext,created_utc,score,subreddit",
    }

    for attempt in range(4):
        try:
            response = requests.get(BASE_URL, params=params, timeout=60) # if it crashes and has no response at all, moves to the except block

            if response.status_code in (422, 429, 500, 503):
                wait = 15 * (attempt + 1)
                print(f"  Server returned {response.status_code}, retrying in {wait}s...")
                time.sleep(wait)
                continue # jumps back to the top of the for loop 

            response.raise_for_status()
            data = response.json()
            return data.get("data", [])

        except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout): 
            wait = 30 * (attempt + 1)  # 30s, 60s, 90s, 120s
            print(f"  Connection dropped, retrying in {wait}s...")
            time.sleep(wait) # falls through to the next attempt naturally

    # loop ends without ever hitting a return  
    print("  Failed after 4 retries, skipping this batch.")
    return [] # returns an empty list so the script continues rather than crashing


# ---------------------------------------------------------------------------
# Stage 2 -- Collect (paginate through full date range)
# ---------------------------------------------------------------------------
def collect_subreddit(subreddit: str, start: str, end: str, resume_ts: int = None) -> list:
    """Pull ALL posts from a subreddit between start and end dates by
    paginating through 100-post windows.

    After each call we advance after_ts to (last post's created_utc + 1)
    so the next call picks up exactly where the previous one left off.
    We stop when the API returns fewer than 100 posts -- that means we've
    reached the end of the data for this subreddit/date range."""
    all_posts = []
    after_ts  = resume_ts if resume_ts else date_to_ts(start) # resume from crash point
    end_ts    = date_to_ts(end)

    while after_ts < end_ts:
        batch = fetch_posts(subreddit, after_ts, end_ts)

        if not batch:
            break

        batch.sort(key=lambda p: p["created_utc"])
        all_posts.extend(batch)

        # advance the window to just after the last post we received
        after_ts = batch[-1]["created_utc"] + 1

        with open(PROGRESS_PATH, "w") as f:
            json.dump({"subreddit": subreddit, "after_ts": after_ts}, f)

        print(f"  r/{subreddit}: {len(all_posts)} posts so far, "
              f"up to {datetime.utcfromtimestamp(after_ts).date()}")

        time.sleep(0.2)  

    return all_posts


# ---------------------------------------------------------------------------
# Stage 3 -- Filter (relevance check)
# ---------------------------------------------------------------------------
def is_relevant(post: dict) -> bool:
    """Return True if the post mentions AAPL or Apple in its title or body.

    We pull broadly (all posts by subreddit + date) and filter locally
    because using the API's keyword filter combined with a date range
    causes timeouts on active subreddits like r/wallstreetbets.

    [removed] selftext is treated as empty -- we don't want to match
    against the literal string "[removed]"."""
    title    = post.get("title", "") or ""
    selftext = post.get("selftext", "") or ""

    if selftext == "[removed]":
        selftext = ""

    combined = (title + " " + selftext).lower()
    return any(kw.lower() in combined for kw in KEYWORDS)


# ---------------------------------------------------------------------------
# Stage 4 -- Trading day assignment (4pm ET cutoff)
# ---------------------------------------------------------------------------
def assign_trading_day(created_utc: int) -> str:
    """Convert a Unix timestamp to a trading day string using a 4pm ET cutoff.

    Posts before 4pm ET --> assigned to that calendar date
    Posts at/after 4pm ET --> assigned to the next calendar date

    We use pytz for the UTC -> ET conversion because Eastern Time shifts
    between UTC-5 (winter/EST) and UTC-4 (summer/EDT) and a fixed offset
    would give wrong results for half the year."""
    dt_utc = datetime.utcfromtimestamp(created_utc).replace(tzinfo=pytz.utc)
    dt_et  = dt_utc.astimezone(EASTERN)

    if dt_et.hour < MARKET_CLOSE_HOUR:
        trading_day = dt_et.date()
    else:
        trading_day = dt_et.date() + timedelta(days=1)

    return trading_day.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # load checkpoint -- rows from fully completed subreddits
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint_df   = pd.read_csv(CHECKPOINT_PATH)
        all_rows        = checkpoint_df.to_dict("records")
        done_subreddits = set(checkpoint_df["subreddit"].unique())
        print(f"Checkpoint loaded: {len(all_rows)} rows, "
              f"completed subreddits: {done_subreddits}")
    else:
        all_rows        = []
        done_subreddits = set()

    # load mid-subreddit progress -- where we crashed last time
    resume_subreddit = None
    resume_ts        = None
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH) as f:
            prog = json.load(f)
        resume_subreddit = prog["subreddit"]
        resume_ts        = prog["after_ts"]
        done_subreddits.discard(resume_subreddit)  # treat as incomplete
        print(f"Mid-run progress found: r/{resume_subreddit}, resuming...")

    for subreddit in SUBREDDITS:
        if subreddit in done_subreddits:
            print(f"\n--- Skipping r/{subreddit} (already in checkpoint) ---")
            continue

        print(f"\n--- Collecting r/{subreddit} ---")
        start_ts = resume_ts if subreddit == resume_subreddit else None
        posts = collect_subreddit(subreddit, START_DATE, END_DATE, resume_ts = start_ts)
        print(f"  Total posts pulled: {len(posts)}")

        kept = 0
        for post in posts:
            if not is_relevant(post):
                continue

            selftext = post.get("selftext", "") or ""
            if selftext == "[removed]":
                selftext = ""

            all_rows.append({
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

        print(f"  Relevant posts kept: {kept}")

        # save checkpoint after each subreddit completes
        pd.DataFrame(all_rows).to_csv(CHECKPOINT_PATH, index=False)
        if os.path.exists(PROGRESS_PATH):
            os.remove(PROGRESS_PATH)
        print(f"  Checkpoint saved.")

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset="id")
    df = df.sort_values("trading_day").reset_index(drop=True)

    df.to_csv(CSV_PATH, index=False)
    print(f"\nSaved {len(df)} posts to {CSV_PATH}")
    print(df.head(10))


if __name__ == "__main__":
    main()