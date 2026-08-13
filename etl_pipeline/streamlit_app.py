"""
Streamlit UI — Multi-Stock Direction Predictor
===============================================
Run with:
  streamlit run streamlit_app.py

Make sure the inference server is running first:
  python -m uvicorn inference_server:app --port 8001

Supported tickers: AAPL, MSFT, TSLA, NVDA, GOOGL
Note: Reddit sentiment is only available for AAPL.
"""

import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # load GROQ_API_KEY from .env before importing agent

from agent import ask_agent


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Stock Direction Predictor",
    page_icon  = "📈",
    layout     = "wide",
)


# ── Header ────────────────────────────────────────────────────────────────────
st.title("📈 Stock Direction Predictor")
st.caption("Entropy-Gated Adaptive Hybrid Forecasting System")

# ── Ticker selector ───────────────────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"]
ticker = st.selectbox(
    "Select stock",
    TICKERS,
    index = 0,
    help  = "Reddit sentiment data is only available for AAPL. Other tickers use price indicators only.",
)

st.divider()


# ── Fetch prediction from inference server ────────────────────────────────────
INFERENCE_SERVER_URL = os.environ.get("INFERENCE_SERVER_URL", "http://localhost:8001")


def fetch_prediction(ticker: str = "AAPL"):
    try:
        r = requests.get(
            f"{INFERENCE_SERVER_URL}/predict",
            params  = {"ticker": ticker},
            timeout = 60,  # Render free tier needs ~30-60s to wake up
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}


pred = fetch_prediction(ticker)


# ── Layout: left = prediction panel, right = chat ─────────────────────────────
left, right = st.columns([1, 1.6], gap="large")


# ── LEFT: Prediction panel ────────────────────────────────────────────────────
with left:
    st.subheader(f"{ticker} — Next-Day Prediction")

    if "error" in pred:
        st.error(f"Inference server not reachable: {pred['error']}")
        st.info("Start the server with:\n```\npython -m uvicorn inference_server:app --port 8001\n```")
    else:
        direction = pred.get("combined_direction", "UNCERTAIN")
        agreement = pred.get("model_agreement", False)
        as_of     = pred.get("as_of_date", "—")

        # Big direction indicator
        if direction == "UP":
            st.success(f"## ▲  {direction}")
        elif direction == "DOWN":
            st.error(f"## ▼  {direction}")
        else:
            st.warning(f"## —  {direction}")

        st.caption(f"Based on data as of **{as_of}** · Predicting next trading day")

        st.divider()

        # Model probabilities
        # Numeric delta = prob minus 50% baseline.
        # Positive → green up arrow (UP), negative → red down arrow (DOWN).
        st.markdown("**Model Probabilities**")
        col1, col2 = st.columns(2)
        xgb_prob = pred.get("xgb_probability_up", 0.5)
        lstm_prob = pred.get("lstm_probability_up", 0.5)
        xgb_dir  = pred.get("xgb_direction", "—")
        lstm_dir = pred.get("lstm_direction", "—")
        col1.metric("XGBoost", f"{xgb_prob*100:.1f}% · {xgb_dir}",
                    delta=round((xgb_prob - 0.5) * 100, 2),
                    delta_color="normal")
        col2.metric("LSTM",    f"{lstm_prob*100:.1f}% · {lstm_dir}",
                    delta=round((lstm_prob - 0.5) * 100, 2),
                    delta_color="normal")

        combined_pct = pred.get("combined_probability", 0.5) * 100
        st.metric("Combined (85% XGB + 15% LSTM)", f"{combined_pct:.1f}%")

        st.divider()

        # Agreement status
        if agreement:
            st.success("✅ Models in agreement — higher confidence")
        else:
            st.warning("⚠️ Models disagree — treat as UNCERTAIN")

        st.divider()

        # Refresh button
        if st.button("🔄 Refresh Prediction"):
            st.cache_data.clear()
            st.rerun()


# ── RIGHT: Agent chat ─────────────────────────────────────────────────────────
with right:
    st.subheader("Ask the Analyst Agent")
    st.caption(f"Powered by Llama 3.3 70B via Groq · Analysing {ticker}")

    # Reset chat history when ticker changes
    if st.session_state.get("last_ticker") != ticker:
        st.session_state.messages = []
        st.session_state["last_ticker"] = ticker

    # Initialise chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render chat history
    chat_container = st.container(height=480)
    with chat_container:
        if not st.session_state.messages:
            st.markdown(
                f"_Ask me anything about {ticker} — e.g. 'What is the prediction for tomorrow?' "
                f"or 'What do the technical indicators say?'_"
            )
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input(f"Ask about {ticker}..."):
        # Add user message to history and rerender
        st.session_state.messages.append({"role": "user", "content": prompt})

        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    answer = ask_agent(prompt, ticker=ticker)
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear chat"):
            st.session_state.messages = []
            st.rerun()
