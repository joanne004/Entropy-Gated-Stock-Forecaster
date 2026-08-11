"""
Streamlit UI — AAPL Stock Direction Predictor
=============================================
Run with:
  streamlit run streamlit_app.py

Make sure the inference server is running first:
  python -m uvicorn inference_server:app --port 8001
"""

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # load GROQ_API_KEY from .env before importing agent

from agent import ask_agent


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "AAPL Predictor",
    page_icon  = "📈",
    layout     = "wide",
)


# ── Header ────────────────────────────────────────────────────────────────────
st.title("📈 AAPL Stock Direction Predictor")
st.caption("Entropy-Gated Adaptive Hybrid Forecasting System · York University Capstone")
st.divider()


# ── Fetch prediction from inference server ────────────────────────────────────
@st.cache_data(ttl=300)   # cache for 5 minutes — avoids hammering the server on rerenders
def fetch_prediction():
    try:
        r = requests.get("http://localhost:8001/predict", timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


pred = fetch_prediction()


# ── Layout: left = prediction panel, right = chat ─────────────────────────────
left, right = st.columns([1, 1.6], gap="large")


# ── LEFT: Prediction panel ────────────────────────────────────────────────────
with left:
    st.subheader("Next-Day Prediction")

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
        st.markdown("**Model Probabilities**")
        col1, col2 = st.columns(2)
        col1.metric("XGBoost", f"{pred.get('xgb_probability_up', 0)*100:.1f}%", "UP" if pred.get('xgb_direction') == 'UP' else "DOWN")
        col2.metric("LSTM",    f"{pred.get('lstm_probability_up', 0)*100:.1f}%", "UP" if pred.get('lstm_direction') == 'UP' else "DOWN")

        combined_pct = pred.get("combined_probability", 0.5) * 100
        st.metric("Combined (70% XGB + 30% LSTM)", f"{combined_pct:.1f}%")

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
    st.caption("Powered by Llama 3.3 70B via Groq · Uses live prediction + DB data")

    # Initialise chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render chat history
    chat_container = st.container(height=480)
    with chat_container:
        if not st.session_state.messages:
            st.markdown(
                "_Ask me anything about AAPL — e.g. 'What is the prediction for tomorrow?' "
                "or 'What do the technical indicators say?'_"
            )
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask about AAPL..."):
        # Add user message to history and rerender
        st.session_state.messages.append({"role": "user", "content": prompt})

        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    answer = ask_agent(prompt)
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear chat"):
            st.session_state.messages = []
            st.rerun()
