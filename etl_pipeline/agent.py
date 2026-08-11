"""
Agentic Analyst — AAPL Stock Direction Agent
=============================================
A LangChain agent powered by Llama 3.3 70B via Groq API.

The agent has 4 tools it can call in any order:
  1. get_prediction        — calls the FastAPI inference server (/predict)
  2. get_latest_features   — fetches today's RSI, sentiment, entropy etc. from the DB
  3. get_entropy_history   — checks if today's entropy is unusually high (noisy signal)
  4. get_recent_performance — last 5 days of returns and sentiment trend

The LLM decides which tools to call and in what order depending on your question.
It loops — tool → result back to LLM → next tool → result → ... → final answer.

Run with:
  python agent.py
Make sure the inference server is running first:
  python -m uvicorn inference_server:app --port 8001
"""

import os
import requests
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()  # loads GROQ_API_KEY from .env file

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent


# ── Constants ─────────────────────────────────────────────────────────────────
POSTGRES_HOST    = os.environ.get("POSTGRES_HOST", "localhost")
DB_URL           = f"postgresql://stockuser:stockpass@{POSTGRES_HOST}:5432/stockdb"
INFERENCE_SERVER = "http://localhost:8001"
ENTROPY_THRESHOLD = 0.75
GROQ_API_KEY     = os.environ.get("GROQ_API_KEY", "")


# ── Tool 1: get_prediction ────────────────────────────────────────────────────
@tool
def get_prediction() -> dict:
    """
    Call the inference server and get the next-day AAPL direction prediction.
    Returns XGBoost probability, LSTM probability, combined probability, direction
    (UP, DOWN, or UNCERTAIN), and whether models agreed.
    Always call this first when the user asks about tomorrow's prediction.
    """
    try:
        response = requests.get(f"{INFERENCE_SERVER}/predict", timeout=10)
        return response.json()
    except Exception as e:
        return {"error": str(e), "message": "Inference server may not be running"}


# ── Tool 2: get_latest_features ───────────────────────────────────────────────
@tool
def get_latest_features() -> dict:
    """
    Get today's feature values from the database.
    Returns RSI-14, ADX-14, MACD, sentiment score, entropy, alpha, post_count, and daily returns.
    Call this when the user asks what the indicators look like or why the model made a prediction.
    """
    engine = create_engine(DB_URL)
    df = pd.read_sql(
        "SELECT * FROM fused_features ORDER BY date DESC LIMIT 1",
        engine,
    )
    row = df.iloc[0]
    return {
        "date"       : str(row["date"])[:10],
        "returns"    : round(float(row["returns"]),    4),
        "rsi_14"     : round(float(row["rsi_14"]),     2),
        "adx_14"     : round(float(row["adx_14"]),     2),
        "macd"       : round(float(row["macd"]),       4),
        "sentiment"  : round(float(row["sentiment"]),  4),
        "entropy"    : round(float(row["entropy"]),    4),
        "alpha"      : round(float(row["alpha"]),      4),
        "post_count" : int(row["post_count"]),
    }


# ── Tool 3: get_entropy_history ───────────────────────────────────────────────
@tool
def get_entropy_history() -> dict:
    """
    Get the last 30 days of entropy values to check if today's entropy is unusually high.
    High entropy means Reddit sentiment is divided — the signal is noisy and predictions
    should be treated with caution.
    Call this when assessing prediction confidence or when the user asks about uncertainty.
    """
    engine = create_engine(DB_URL)
    df = pd.read_sql(
        "SELECT date, entropy FROM fused_features ORDER BY date DESC LIMIT 30",
        engine,
    )
    df = df.iloc[::-1].reset_index(drop=True)

    today_entropy = float(df["entropy"].iloc[-1])
    p75           = float(df["entropy"].quantile(ENTROPY_THRESHOLD))

    return {
        "today_entropy"   : round(today_entropy, 4),
        "75th_percentile" : round(p75, 4),
        "is_elevated"     : today_entropy > p75,
        "last_30_values"  : df["entropy"].round(4).tolist(),
    }


# ── Tool 4: get_recent_performance ────────────────────────────────────────────
@tool
def get_recent_performance() -> dict:
    """
    Get the last 5 trading days of price returns, sentiment, and closing price.
    Useful for showing the recent trend context around today's prediction.
    Call this when the user asks about recent market behaviour or trend.
    """
    engine = create_engine(DB_URL)
    df = pd.read_sql(
        "SELECT date, returns, sentiment, close FROM fused_features ORDER BY date DESC LIMIT 5",
        engine,
    )
    df = df.iloc[::-1].reset_index(drop=True)

    records = []
    for _, row in df.iterrows():
        records.append({
            "date"      : str(row["date"])[:10],
            "returns"   : round(float(row["returns"]),   4),
            "sentiment" : round(float(row["sentiment"]), 4),
            "close"     : round(float(row["close"]),     2),
        })
    return {"last_5_days": records}


# ── LLM setup ─────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model       = "llama-3.3-70b-versatile",
    temperature = 0.1,
    api_key     = GROQ_API_KEY,
)


# ── System prompt — one-shot + chain-of-thought ───────────────────────────────
system_prompt = SystemMessage(content="""You are an expert financial analyst assistant for AAPL stock prediction.
You have access to live prediction data, technical indicators, and Reddit sentiment signals.

TOOL SELECTION — call only the tools relevant to the question:
  - Questions about tomorrow's prediction → get_prediction + get_entropy_history
  - Questions about indicators or why the model predicted X → get_latest_features
  - Questions about confidence or signal reliability → get_entropy_history
  - Questions about recent trend or this week → get_recent_performance
  - General analysis questions → use all tools as needed

RESPONSE FORMAT — match the depth and structure to the question asked.
For full prediction questions, use this format:

Example question: "What is the prediction for AAPL tomorrow?"

Example answer:
"**Prediction: UP** (models in agreement)

**Model confidence:** XGBoost 52.3% UP · LSTM 51.8% UP · Combined 52.1%

**Sentiment signal:** Entropy is LOW (0.42 vs 75th percentile 0.68) — Reddit sentiment is
consistent, so the signal is reliable. Sentiment score is mildly positive at +0.12.

**Technical picture:** RSI at 54 (neutral, no overbought/oversold warning). MACD histogram
positive at +0.31 — short-term momentum is above long-term, supporting the UP call.
ADX at 22 — market is trending but not strongly.

**Recent trend:** AAPL has been up 3 of the last 5 days with improving sentiment. The trend
supports the model's prediction.

**Bottom line:** Lean UP for tomorrow. Signal is reasonably reliable given low entropy and
agreeing technical indicators. As always, treat this as one input — not financial advice."

IMPORTANT RULES:
- If entropy is elevated (is_elevated = true), warn the user clearly that the signal is noisy
- If combined_direction is UNCERTAIN, tell the user the models disagree and avoid a directional call
- Always mention model agreement status
- Never give specific price targets — only direction
- End every response with: "Not financial advice."
""")


# ── Agent setup ───────────────────────────────────────────────────────────────
tools = [get_prediction, get_latest_features, get_entropy_history, get_recent_performance]
agent = create_react_agent(llm, tools, prompt=system_prompt)


# ── ask_agent() — callable by Streamlit ───────────────────────────────────────
def ask_agent(question: str) -> str:
    """
    Single-question interface for Streamlit.
    Returns the agent's final answer as a plain string.
    """
    try:
        response = agent.invoke({"messages": [("user", question)]})
        return response["messages"][-1].content
    except Exception as e:
        return f"Error: {e}"


# ── CLI conversation loop ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  AAPL Analyst Agent  (powered by Llama 3.3 via Groq)")
    print("  Type 'quit' to exit")
    print("=" * 55)

    while True:
        question = input("\nYou: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        print(f"\nAgent: {ask_agent(question)}")
