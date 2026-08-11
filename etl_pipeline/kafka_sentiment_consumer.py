"""
Kafka Sentiment Consumer
Listens to the 'aapl-sentiment' topic and prints each message.
"""

import json
from kafka import KafkaConsumer

TOPIC    = "aapl-sentiment"
BROKER   = "localhost:29092"
GROUP_ID = "sentiment-consumer-group"


def make_consumer() -> KafkaConsumer:
    return KafkaConsumer(
        TOPIC,
        bootstrap_servers=BROKER,
        group_id=GROUP_ID,
        auto_offset_reset="earliest",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )


def handle_message(msg: dict) -> None:
    date       = msg.get("date", "?")
    sentiment  = msg.get("sentiment", "?")
    entropy    = msg.get("entropy", "?")
    post_count = msg.get("post_count", "?")
    ticker     = msg.get("ticker", "?")
    print(f"  [{ticker}] {date}  sentiment={sentiment:.4f}  entropy={entropy:.4f}  posts={post_count}")


def main():
    consumer = make_consumer()
    print(f"Listening on topic '{TOPIC}' (Ctrl+C to stop)...\n")

    try:
        for record in consumer:
            handle_message(record.value)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        consumer.close()
        print("Consumer closed.")


if __name__ == "__main__":
    main()
