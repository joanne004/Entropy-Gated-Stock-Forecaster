"""
Step 6 -- Feature Fusion
Joins price features (X_t) and sentiment scores (S_t),
computes entropy gate alpha_t, writes to fused_features table

"""

import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
DB_URL = f"postgresql://stockuser:stockpass@{POSTGRES_HOST}:5432/stockdb"
engine = create_engine(DB_URL) # creates a connection

def main():
    # load both tables from PostgreSQL
    prices = pd.read_sql("SELECT * FROM price_features", engine)
    sentiment = pd.read_sql("SELECT * FROM sentiment_scores", engine)

    prices["date"]    = pd.to_datetime(prices["date"])
    sentiment["date"] = pd.to_datetime(sentiment["date"])

    # join on date and ticker
    df = pd.merge(prices, sentiment, on=["date", "ticker"], how="inner")
    print(f"Merged: {len(df)} rows")

    # compute entropy gate alpha_t
    # H_t / log(3) normalises entropy to 0-1 range 
    # alpha_t = 1 minus that, so high entropy = low entropy(trust sentiment less)
    df["alpha"] = 1 - (df["entropy"] / np.log(3))

    # apply alpha to sentiment to get gated sentiment
    df["sentiment"] = df["alpha"] * df["sentiment"] # recompute the sentiment based on entropy

    # keeps only the columns the fuse_feature table needs
    keep = ["date","ticker","open","high","low","close","volume",
            "sma_20","rsi_14","obv","adx_14","macd","returns","log_returns",
            "sentiment","entropy","alpha","post_count"]

    df = df[keep]

    print(df.head(5))
    print(f"Saving {len(df)} rows to fused_features....")

    df.to_sql(
        name = "fused_features",
        con = engine,
        if_exists = "replace",
        index = False
    )

    print("Done.")


if __name__ == "__main__":
    main()

