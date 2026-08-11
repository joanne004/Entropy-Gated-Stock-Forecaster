"""
Kafka Sentiment Producer
Reads sentiment_scores from PostgreSQL and publishes each row
as a JSON message to the 'aapl-sentiment' Kafka topic.
"""

import json
import time
import pandas as pd
from kafka import KafkaProducer
from sqlalchemy import create_engine

DB_URL   = "postgresql://stockuser:stockpass@localhost:5432/stockdb"
TOPIC    = "aapl-sentiment"
BROKER   = "localhost:29092"
DELAY_MS = 10  # milliseconds between messages

engine = create_engine(DB_URL)


def make_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=BROKER, # tells the produce where the Kafka server is running
        value_serializer=lambda v: json.dumps(v).encode("utf-8"), # is what takes the python dict converts to a JSON message and then convert to byte
    )


def main():
    print(f"Loading sentiment_scores from PostgreSQL...")
    df = pd.read_sql("SELECT * FROM sentiment_scores ORDER BY date ASC", engine)
    print(f"  {len(df)} rows loaded.")

    df["date"] = df["date"].astype(str)

    producer = make_producer()
    print(f"Connected to Kafka broker at {BROKER}.")
    print(f"Publishing {len(df)} messages to topic '{TOPIC}'...\n")

    for i, row in df.iterrows():
        message = row.to_dict()
        producer.send(TOPIC, value=message)

        if (i + 1) % 100 == 0:
            print(f"  Sent {i + 1} / {len(df)} messages...")

        time.sleep(DELAY_MS / 1000)

    producer.flush()
    print(f"\nDone. {len(df)} sentiment messages published to '{TOPIC}'.")


if __name__ == "__main__":
    main()
