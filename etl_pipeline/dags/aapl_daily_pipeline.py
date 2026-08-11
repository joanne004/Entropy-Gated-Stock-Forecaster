"""
AAPL Daily Pipeline DAG
Orchestrates the full ETL pipeline on a daily schedule:
  1. Fetch latest AAPL price data and compute indicators
  2. Scrape Reddit WSB posts
  3. Score sentiment with FinBERT
  4. Load prices to PostgreSQL
  5. Load sentiment to PostgreSQL
  6. Fuse features and apply entropy gate

Runs at 6pm on weekdays (after US markets close).
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator # BashOperator is a task type in Airflow that runs a shell command

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPTS_DIR = "/opt/airflow/scripts"  # mounted from ./etl_pipeline in docker-compose. Is the path in the docker container where the scripts are located

# Settings that apply to every task in DAG automatically
default_args = {
    "owner": "joanne",
    "retries": 1,                          # retry once if a task fails
    "retry_delay": timedelta(minutes=5),   # wait 5 minutes before retrying
}

# ── DAG definition ────────────────────────────────────────────────────────────
with DAG(
    dag_id="aapl_daily_pipeline", # is the unique name for this pipeline
    description="Daily AAPL ETL: prices + sentiment → fused features", 
    default_args=default_args,
    start_date=datetime(2024, 1, 1), # date Airflow considers the pipeline to have started from
    schedule="0 18 * * 1-5",   # its a Cron expression. Cron expression is a standard form for scheduling. 'minute, hour, day, month, day-of-week. " at minute 0, of hour 18, every day, every month, Monday through Friday"
    catchup=False,              # don't backfill missed runs, if Airflow starts today and the start date was months ago, it wont try to backfill and run the pipeline on the missed days
    tags=["aapl", "etl"], # labels so you can filter DAGs in the UI when you have many of them
) as dag:

    # ── Task 1: Fetch price data and compute indicators ────────────────────────
    fetch_prices = BashOperator(
        task_id="fetch_prices", # unique name for the task to be run,  shown in the Airflow UI 
        bash_command=f"cd {SCRIPTS_DIR} && python step1_price_features.py", # shell command it runs
    )

    # ── Task 2: Scrape Reddit WSB posts ───────────────────────────────────────
    scrape_reddit = BashOperator(
        task_id="scrape_reddit",
        bash_command=f"cd {SCRIPTS_DIR} && python step2a_reddit_raw.py",
    )

    # ── Task 3: Score sentiment with FinBERT ──────────────────────────────────
    score_sentiment = BashOperator(
        task_id="score_sentiment",
        bash_command=f"cd {SCRIPTS_DIR} && python step3_sentiment_finbert.py",
    )

    # ── Task 4: Load prices into PostgreSQL ───────────────────────────────────
    load_prices = BashOperator(
        task_id="load_prices",
        bash_command=f"cd {SCRIPTS_DIR} && python step5a_load_prices.py",
    )

    # ── Task 5: Load sentiment into PostgreSQL ────────────────────────────────
    load_sentiment = BashOperator(
        task_id="load_sentiment",
        bash_command=f"cd {SCRIPTS_DIR} && python step5b_load_sentiment.py",
    )

    # ── Task 6: Fuse features and apply entropy gate ──────────────────────────
    fuse_features = BashOperator(
        task_id="fuse_features",
        bash_command=f"cd {SCRIPTS_DIR} && python step6_fuse_features.py",
    )

    # ── Dependencies ──────────────────────────────────────────────────────────
    # Sentiment chain: scrape first, then score, then load
    scrape_reddit >> score_sentiment >> load_sentiment

    # Price chain: fetch prices, then load
    fetch_prices >> load_prices

    # Fusion waits for BOTH price and sentiment to be loaded
    [load_prices, load_sentiment] >> fuse_features
