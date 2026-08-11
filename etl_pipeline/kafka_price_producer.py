"""
Kafka Price Producer
Reads price_features from PostgreSQL and publishes each row
as a JSON message to the 'aapl-prices' Kafka topic.
"""

import json
import time
import pandas as pd
from kafka import KafkaProducer # Kafka producer is the class from Kafka python that lets us publish messages
from sqlalchemy import create_engine

DB_URL   = "postgresql://stockuser:stockpass@localhost:5432/stockdb"
TOPIC    = "aapl-prices"
BROKER   = "localhost:29092"
DELAY_MS = 10  # milliseconds between messages (simulates a live stream)

engine = create_engine(DB_URL)


def make_producer() -> KafkaProducer:
    """Create a KafkaProducer that serialises messages to JSON bytes."""
    return KafkaProducer(
        bootstrap_servers = BROKER, # tells the producer where Kafka is running
        value_serializer = lambda v: json.dumps(v).encode("utf-8"), # this is what runs automatically on every message before it gets sent. It takes the python dict v, converts to JSON string with json.dumps(v), then converts that string to bytes with .encode("utf-8")
    )


def main():
    print(f"Loading price_features from PostgresSQL...")
    df = pd.read_sql("SELECT * FROM price_features ORDER BY date ASC", engine)
    print(f" {len(df)} rows loaded.")

    df["date"] = df["date"].astype(str) # JSON doesn't know what a Timestamp is, so we convert it to a plain string before we send it through Kafka

    producer = make_producer()

    for i, row in df.iterrows():
        message = row.to_dict() # converts one row of the dataframe to a plain python dict so that serializer can handle it
        producer.send(TOPIC, value = message)

        if(i + 1) % 100 == 0:
            print(f" Sent {i+1}/{len(df)} messages...")

        time.sleep(DELAY_MS / 1000)

    producer.flush() # this forces all buffered messages to actually go out. Without this, some might get dropped when the script exists
    print(f"\nDone. {len(df)} price messages published to '{TOPIC}'.")


if __name__ == "__main__":
    main()
