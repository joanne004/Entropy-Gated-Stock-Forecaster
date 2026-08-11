"""
Prediction model helper for the price consumer.

This module is intentionally simple so the consumer can run after the
producer and still persist a prediction result that a FastAPI endpoint
can read later.
"""

from datetime import datetime

import pandas as pd
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Float,
    Date,
    DateTime,
)

DB_URL = "postgresql://stockuser:stockpass@localhost:5432/stockdb"
PREDICTIONS_TABLE = "predictions"
engine = create_engine(DB_URL)
metadata = MetaData()

predictions_table = Table(
    PREDICTIONS_TABLE,
    metadata,
    Column("ticker", String, primary_key=True),
    Column("date", Date, primary_key=True),
    Column("close", Float),
    Column("predicted_close", Float),
    Column("predicted_return", Float),
    Column("model_name", String),
    Column("predicted_at", DateTime),
)

metadata.create_all(engine)


class DummyPriceModel:
    """A placeholder model that turns a price message into a prediction."""

    def __init__(self, name: str = "dummy"):
        self.name = name

    def predict(self, features: dict) -> dict:
        close = float(features.get("close", 0.0) or 0.0)
        predicted_close = close * 1.001 if close else 0.0
        predicted_return = ((predicted_close - close) / close) if close else 0.0
        return {
            "predicted_close": float(predicted_close),
            "predicted_return": float(predicted_return),
        }


def load_model() -> DummyPriceModel:
    return DummyPriceModel()


def save_prediction(prediction: dict) -> None:
    df = pd.DataFrame([prediction])
    df.to_sql(PREDICTIONS_TABLE, engine, if_exists="append", index=False)


def get_latest_prediction(ticker: str | None = None) -> list[dict]:
    sql = "SELECT * FROM predictions"
    params = {}
    if ticker:
        sql += " WHERE ticker = :ticker"
        params = {"ticker": ticker}

    sql += " ORDER BY date DESC, predicted_at DESC LIMIT 1"
    df = pd.read_sql(sql, engine, params=params)
    if df.empty:
        return []

    df["predicted_at"] = pd.to_datetime(df["predicted_at"])
    return df.to_dict(orient="records")
