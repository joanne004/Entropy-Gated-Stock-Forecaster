"""
Step 5b -- Load sentiment scores into PostgresSQL sentiment_scores table

"""
import os
import pandas as pd
from sqlalchemy import create_engine

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
DB_URL = f"postgresql://stockuser:stockpass@{POSTGRES_HOST}:5432/stockdb"
CSV_PATH = "data/S_t_AAPL_sentiment.csv"

engine = create_engine(DB_URL) # make the connection to 

def main():
    print(f"Loading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    df = df.rename(columns = {"trading_day": "date"})
    df["date"] = pd.to_datetime(df["date"])

    print(f" {len(df)} rows to load...")
    df.to_sql(
        name = "sentiment_scores",
        con = engine,
        if_exists = "replace",
        index = False,
    )

    print("Done. sentiment_scores table loaded.")
    print(df.head(3))

if __name__ == "__main__":
    main()