"""
Agentic Analyst — Multi-Stock Direction Agent
==============================================
A LangChain agent powered by Llama 3.3 70B via Groq API.

The agent has 4 tools it can call in any order:
  1. get_prediction        — calls the FastAPI inference server (/predict?ticker=...)
  2. get_latest_features   — fetches today's RSI, sentiment, entropy etc. from the DB
  3. get_entropy_history   — checks if today's entropy is unusually high (noisy signal)
  4. get_recent_performance — last 5 days of returns and sentiment trend

The LLM decides which tools to call and in what order depending on your question.
It loops — tool → result back to LLM → next tool → result → ... → final answer.

Supported tickers: AAPL, MSFT, TSLA, NVDA, GOOGL

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
# Cloud (Neon): set DATABASE_URL in the environment
# Local (Docker): set POSTGRES_HOST (defaults to localhost)
DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL:
    DB_URL = DATABASE_URL
else:
    POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
    DB_URL = f"postgresql://stockuser:stockpass@{POSTGRES_HOST}:5432/stockdb"

INFERENCE_SERVER  = os.environ.get("INFERENCE_SERVER_URL", "http://localhost:8001")
ENTROPY_THRESHOLD = 0.75
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY", "")

# Module-level ticker — set by ask_agent() before each call so all tools
# automatically use whichever stock the user has selected in the UI.
CURRENT_TICKER = "AAPL"


# ── Tool 1: get_prediction ────────────────────────────────────────────────────
@tool
def get_prediction() -> dict:
    """
    Call the inference server and get the next-day direction prediction for the
    currently selected stock ticker.
    Returns XGBoost probability, LSTM probability, combined probability, direction
    (UP, DOWN, or UNCERTAIN), and whether models agreed.
    Always call this first when the user asks about tomorrow's prediction.
    """
    try:
        response = requests.get(
            f"{INFERENCE_SERVER}/predict",
            params  = {"ticker": CURRENT_TICKER},
            timeout = 10,
        )
        return response.json()
    except Exception as e:
        return {"error": str(e), "message": "Inference server may not be running"}


# ── Tool 2: get_latest_features ───────────────────────────────────────────────
@tool
def get_latest_features() -> dict:
    """
    Get the most recent feature values from the database for the selected ticker.
    Returns RSI-14, ADX-14, MACD, VIX, sentiment score, entropy, alpha, post_count, and daily returns.
    Call this when the user asks what the indicators look like or why the model made a prediction.
    """
    engine = create_engine(DB_URL)
    df = pd.read_sql(
        f"SELECT * FROM fused_features WHERE ticker = '{CURRENT_TICKER}' ORDER BY date DESC LIMIT 1",
        engine,
    )
    row = df.iloc[0]
    return {
        "date"       : str(row["date"])[:10],
        "returns"    : round(float(row["returns"]),    4),
        "rsi_14"     : round(float(row["rsi_14"]),     2),
        "adx_14"     : round(float(row["adx_14"]),     2),
        "macd"       : round(float(row["macd"]),       4),
        "vix"        : round(float(row["vix"]),        2),
        "sentiment"  : round(float(row["sentiment"]),  4),
        "entropy"    : round(float(row["entropy"]),    4),
        "alpha"      : round(float(row["alpha"]),      4),
        "post_count" : int(row["post_count"]),
    }


# ── Tool 3: get_entropy_history ───────────────────────────────────────────────
@tool
def get_entropy_history() -> dict:
    """
    Get the last 30 days of entropy values for the selected ticker.
    High entropy means Reddit sentiment is divided — the signal is noisy and predictions
    should be treated with caution. For non-AAPL tickers entropy is always log(3) (max)
    because there is no Reddit sentiment data for them.
    Call this when assessing prediction confidence or when the user asks about uncertainty.
    """
    engine = create_engine(DB_URL)
    df = pd.read_sql(
        f"SELECT date, entropy FROM fused_features WHERE ticker = '{CURRENT_TICKER}' ORDER BY date DESC LIMIT 30",
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
    Get the last 5 trading days of price returns, sentiment, and closing price for the selected ticker.
    Useful for showing the recent trend context around today's prediction.
    Call this when the user asks about recent market behaviour or trend.
    """
    engine = create_engine(DB_URL)
    df = pd.read_sql(
        f"SELECT date, returns, sentiment, close FROM fused_features WHERE ticker = '{CURRENT_TICKER}' ORDER BY date DESC LIMIT 5",
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


# ── LLM setup (lazy — initialised on first ask_agent() call) ──────────────────
_llm   = None
_agent = None


# ── System prompt — one-shot + chain-of-thought ───────────────────────────────
system_prompt = SystemMessage(content="""You are an expert financial analyst assistant for stock direction prediction.
You support multiple stocks: AAPL, MSFT, TSLA, NVDA, GOOGL.
The user has selected a specific ticker — all your tool calls will automatically query that ticker.
Note: Reddit sentiment data is only available for AAPL. For other tickers, entropy will always be at maximum (log(3)) and sentiment will be 0 — make sure to mention this when discussing non-AAPL predictions.
You have access to live prediction data and technical indicators.

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
ADX at 22 — market is trending but not strongly. VIX at 14.2 — market is calm, trends
more likely to hold. (VIX above 25 = elevated fear, above 35 = panic/crisis)

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


# ── ask_agent() — callable by Streamlit ───────────────────────────────────────
def ask_agent(question: str, ticker: str = "AAPL") -> str:
    """
    Single-question interface for Streamlit.
    Sets CURRENT_TICKER before invoking so all tools query the right stock.
    Returns the agent's final answer as a plain string.
    Initialises the LLM and agent on first call so the module can be imported
    even before GROQ_API_KEY is available in the environment.
    """
    global CURRENT_TICKER, _llm, _agent
    CURRENT_TICKER = ticker.upper()

    # Lazy init — reads env var at call time, not at import time
    if _agent is None:
        key = os.environ.get("GROQ_API_KEY", "")
        _llm = ChatGroq(
            model       = "llama-3.1-8b-instant",
            temperature = 0.1,
            api_key     = key,
        )
        _tools = [get_prediction, get_latest_features, get_entropy_history, get_recent_performance]
        _agent = create_react_agent(_llm, _tools, prompt=system_prompt)

    try:
        response = _agent.invoke({"messages": [("user", question)]})
        return response["messages"][-1].content
    except Exception as e:
        return f"Error: {e}"


# ── CLI conversation loop ─────────────────────────────────────────────────────
def _run_verbose(question: str, ticker: str = "AAPL"):
    """Run agent and print every reasoning step (tool calls + outputs)."""
    global CURRENT_TICKER, _llm, _agent
    CURRENT_TICKER = ticker.upper()

    if _agent is None:
        key = os.environ.get("GROQ_API_KEY", "")
        _llm = ChatGroq(
            model       = "llama-3.1-8b-instant",
            temperature = 0.1,
            api_key     = key,
        )
        _tools = [get_prediction, get_latest_features, get_entropy_history, get_recent_performance]
        _agent = create_react_agent(_llm, _tools, prompt=system_prompt)

    response = _agent.invoke({"messages": [("user", question)]})

    for msg in response["messages"]:
        kind = type(msg).__name__
        if kind == "HumanMessage":
            continue  # already printed as "You:"
        elif kind == "AIMessage":
            if msg.tool_calls:
                print("\n[Agent → Tool calls]")
                for tc in msg.tool_calls:
                    print(f"  → {tc['name']}({tc['args']})")
            else:
                print(f"\n[Agent — Final Answer]\n{msg.content}")
        elif kind == "ToolMessage":
            print(f"\n[Tool result: {msg.name}]")
            print(f"  {msg.content[:500]}")  # cap at 500 chars to keep terminal readable


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  AAPL Analyst Agent  (powered by Llama 3.3 via Groq)")
    print("  Type 'quit' to exit | verbose ReAct steps shown")
    print("=" * 55)

    while True:
        question = input("\nYou: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        _run_verbose(question)
