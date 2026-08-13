"""
Step 5b -- Load sentiment scores into PostgresSQL sentiment_scores table

Loads all available per-ticker sentiment CSVs and writes them as one table.
Currently supports: AAPL, MSFT, NVDA (any ticker that has a sentiment CSV).
"""
import os
import pandas as pd
from sqlalchemy import create_engine

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
DB_URL = f"postgresql://stockuser:stockpass@{POSTGRES_HOST}:5432/stockdb"

# Tickers we want sentiment for — only those with an existing CSV are loaded
SENTIMENT_TICKERS = ["AAPL", "MSFT", "NVDA"]
DATA_DIR = "data"

engine = create_engine(DB_URL)

def main():
    dfs = []
    for ticker in SENTIMENT_TICKERS:
        path = os.path.join(DATA_DIR, f"S_t_{ticker}_sentiment.csv")
        if not os.path.exists(path):
            print(f"  Skipping {ticker} — {path} not found")
            continue
        df = pd.read_csv(path)
        df = df.rename(columns={"trading_day": "date"})
        df["date"] = pd.to_datetime(df["date"])
        # ensure ticker column is set (step3 already sets it, but just in case)
        df["ticker"] = ticker
        dfs.append(df)
        print(f"  Loaded {ticker}: {len(df)} daily rows  ({df['date'].min().date()} → {df['date'].max().date()})")

    if not dfs:
        print("ERROR: No sentiment CSVs found. Run step3 first.")
        return

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined: {len(combined)} rows across {len(dfs)} ticker(s)")

    combined.to_sql(
        name     = "sentiment_scores",
        con      = engine,
        if_exists = "replace",
        index    = False,
    )

    print("Done. sentiment_scores table loaded.")
    print(combined.groupby("ticker")["date"].max().to_string())

if __name__ == "__main__":
    main()