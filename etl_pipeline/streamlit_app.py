"""
Streamlit UI — Multi-Stock Direction Predictor
===============================================
Run with:
  streamlit run streamlit_app.py

Supported tickers: AAPL, MSFT, TSLA, NVDA, GOOGL
Note: Reddit sentiment is only available for AAPL.
"""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # load GROQ_API_KEY from .env before importing agent

from agent import ask_agent
from predictor import _load_models, predict


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


# ── Load models once (cached) ─────────────────────────────────────────────────
models = _load_models()


def fetch_prediction(ticker: str = "AAPL"):
    try:
        return predict(ticker, models)
    except Exception as e:
        return {"error": str(e)}


# ── Layout: left = prediction panel, right = chat ─────────────────────────────
left, right = st.columns([1, 1.6], gap="large")


# ── LEFT: Prediction panel ────────────────────────────────────────────────────
with left:
    st.subheader(f"{ticker} — Next-Day Prediction")

    cache_key = f"pred_{ticker}"

    if st.button("🔄 Get Prediction", type="primary"):
        with st.spinner("Fetching prediction…"):
            st.session_state[cache_key] = fetch_prediction(ticker)

    pred = st.session_state.get(cache_key)

    if pred is None:
        st.info("Click **Get Prediction** to fetch the latest forecast.")
    elif "error" in pred:
        st.error(f"Prediction error: {pred['error']}")
    else:
        xgb_dir        = pred.get("xgb_direction", "UNCERTAIN")
        xgb_prob       = pred.get("xgb_probability_up", 0.5)
        lstm_available = pred.get("lstm_available", False)
        lstm_confirms  = pred.get("lstm_confirms", None)
        as_of          = pred.get("as_of_date", "—")

        # Primary prediction from XGBoost
        if xgb_dir == "UP":
            st.success(f"## ▲  {xgb_dir}")
        elif xgb_dir == "DOWN":
            st.error(f"## ▼  {xgb_dir}")
        else:
            st.warning(f"## —  {xgb_dir}")

        st.caption(f"Based on data as of **{as_of}** · Predicting next trading day")

        st.divider()

        st.markdown("**Model Signals**")
        col1, col2 = st.columns(2)
        col1.metric(
            "XGBoost (primary)",
            f"{xgb_prob*100:.1f}% UP",
            delta=round((xgb_prob - 0.5) * 100, 2),
            delta_color="normal",
            help="Primary predictor — gradient-boosted trees on technical + sentiment features",
        )

        if lstm_available:
            lstm_prob = pred.get("lstm_probability_up", 0.5)
            lstm_dir  = pred.get("lstm_direction", "—")
            col2.metric(
                "LSTM (confirmation gate)",
                f"{lstm_prob*100:.1f}% UP · {lstm_dir}",
                delta=round((lstm_prob - 0.5) * 100, 2),
                delta_color="normal",
                help="Secondary model — confirms or flags disagreement. Does not alter the primary XGBoost signal.",
            )
        else:
            col2.metric("LSTM (confirmation gate)", "N/A",
                        help="LSTM requires TensorFlow locally. Not available in cloud deployment.")

        st.divider()

        # Confidence indicator based on agreement gate
        if lstm_available:
            if lstm_confirms:
                st.success("✅ LSTM confirms XGBoost — high-confidence signal")
            else:
                st.warning("⚠️ LSTM disagrees — treat with caution")
        else:
            st.info("ℹ️ XGBoost-only mode — LSTM unavailable in this environment")


# ── RIGHT: Agent chat ─────────────────────────────────────────────────────────
with right:
    st.subheader("Ask the Analyst Agent")
    st.caption(f"Powered by Llama 3.3 70B via Groq · Analysing {ticker}")

    if st.session_state.get("last_ticker") != ticker:
        st.session_state.messages = []
        st.session_state["last_ticker"] = ticker

    if "messages" not in st.session_state:
        st.session_state.messages = []

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

    if prompt := st.chat_input(f"Ask about {ticker}..."):
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

    if st.session_state.messages:
        if st.button("🗑️ Clear chat"):
            st.session_state.messages = []
            st.rerun()
