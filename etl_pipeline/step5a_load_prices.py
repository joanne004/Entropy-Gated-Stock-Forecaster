"""
Step 5a -- Load price features into PostgreSQL price_features table

"""

import os
import pandas as pd
from sqlalchemy import create_engine

# When running locally, POSTGRES_HOST is not set so it defaults to localhost
# When running inside the Airflow container, POSTGRES_HOST=postgres (the Docker service name)
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
DB_URL = f"postgresql://stockuser:stockpass@{POSTGRES_HOST}:5432/stockdb"

TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"]

engine = create_engine(DB_URL) # creates the connection with PostgresSQL

def main():
    dfs = []
    for ticker in TICKERS:
        path = f"data/X_t_{ticker}.csv"
        print(f"Loading {path}...")
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        dfs.append(df)
        print(f"  {len(df)} rows  ({df['date'].min().date()} → {df['date'].max().date()})")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined: {len(combined)} rows across {len(TICKERS)} tickers")

    combined.to_sql( # is pandas doing the INSERT for us - translates the entire dataframe into SQL rows automatically
        name = "price_features",
        con = engine,
        if_exists = "replace", # wipe old data and reload fresh
        index = False, # don't write row numbers as columns
    )

    print("Done. price_features table loaded.")
    print(combined.groupby("ticker")["date"].max().to_string())

if __name__ == "__main__":
    main()