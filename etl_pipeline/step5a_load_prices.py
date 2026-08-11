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
CSV_PATH = "data/X_t_AAPL.csv"

engine = create_engine(DB_URL) # creates the connection with PostgresSQL

def main():
    print(f"Loading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)

    # rename columns to match the database table
    df = df.rename(columns={
        "sma_20": "sma_20",
        "rsi_14": "rsi_14",
        "obv":    "obv",
        "adx_14": "adx_14",
    }) 

    df["date"] = pd.to_datetime(df["date"]) # converts the date column from a plain string into an actual date object that PostgreSQL understands

    print(f" {len(df)} rows to load...")

    df.to_sql( # is pandas doing the INSERT for us - translates the entire dataframe into SQL row automatically
        name = "price_features",
        con = engine,
        if_exists = "replace", # if there's data in the table, wipe it and reload fresh
        index = False, # don't write row numbers as columns
    )

    print("Done. price_features table loaded.")
    print(df.tail(3))

if __name__ == "__main__":
    main()