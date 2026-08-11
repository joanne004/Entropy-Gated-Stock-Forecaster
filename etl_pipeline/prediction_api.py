"""
FastAPI endpoint to read the latest saved prediction.

This API can be started any morning after the consumer has processed the
Kafka message and saved the prediction into the database.
"""

from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import create_engine

DB_URL = "postgresql://stockuser:stockpass@localhost:5432/stockdb"
engine = create_engine(DB_URL)
app = FastAPI(title="Stock Price Prediction API")


class PredictionResponse(BaseModel):
    ticker: str
    date: str
    close: float
    predicted_close: float
    predicted_return: float
    model_name: str
    predicted_at: datetime


def load_latest_prediction(ticker: Optional[str] = None) -> Optional[dict]:
    sql = "SELECT * FROM predictions"
    params = {}
    if ticker:
        sql += " WHERE ticker = :ticker"
        params = {"ticker": ticker}

    sql += " ORDER BY date DESC, predicted_at DESC LIMIT 1"
    df = pd.read_sql(sql, engine, params=params)
    if df.empty:
        return None

    record = df.iloc[0].to_dict()
    record["predicted_at"] = pd.to_datetime(record["predicted_at"]).to_pydatetime()
    record["date"] = str(record["date"])
    return record


@app.get("/predictions/latest", response_model=PredictionResponse)
def latest_prediction(ticker: Optional[str] = Query(None, description="Ticker symbol to filter")):
    result = load_latest_prediction(ticker)
    if not result:
        raise HTTPException(status_code=404, detail="No predictions found")
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
