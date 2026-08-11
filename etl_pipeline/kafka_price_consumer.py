"""
Kafka Price Consumer
Listens to the 'aapl-prices' topic and prints each message.
In production this would write records into a database or
trigger downstream model inference. 

"""

import json
from datetime import datetime
from kafka import KafkaConsumer

from prediction_model import load_model, save_prediction

TOPIC = "aapl-prices"
BROKER = "localhost:29092"
GROUP_ID = "price-consumer-group"
CONSUMER_TIMEOUT_MS = 10000  # stop after 10 seconds without new records


def make_consumer() -> KafkaConsumer:
    return KafkaConsumer(
        TOPIC,
        bootstrap_servers=BROKER,
        group_id=GROUP_ID,
        auto_offset_reset="earliest",
        consumer_timeout_ms=CONSUMER_TIMEOUT_MS,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )


def handle_message(msg: dict, model) -> None:
    """Process one price record and persist the prediction."""
    date = msg.get("date", "?")
    close = msg.get("close", "?")
    ticker = msg.get("ticker", "?")

    print(f"  [{ticker}] {date}  close={close}")

    prediction = model.predict(msg)
    prediction_row = {
        "ticker": ticker,
        "date": date,
        "close": float(close) if close != "?" else None,
        "predicted_close": prediction["predicted_close"],
        "predicted_return": prediction["predicted_return"],
        "model_name": model.name,
        "predicted_at": datetime.utcnow(),
    }
    save_prediction(prediction_row)
    print(f"    saved prediction={prediction_row['predicted_close']:.4f}")


def main():
    consumer = make_consumer()
    model = load_model()
    print(f"Reading pending messages from topic '{TOPIC}'...")

    try:
        for record in consumer:
            handle_message(record.value, model)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        consumer.close()
        print("Consumer closed.")


if __name__ == "__main__":
    main()
