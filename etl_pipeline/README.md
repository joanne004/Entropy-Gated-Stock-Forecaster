# ETL Pipeline — Local Setup Guide

This folder is where the ETL pipeline gets built, one verified piece at a time, on your own machine. We're deliberately starting with **logic first, infrastructure second**: prove the feature-computation code is correct in a plain Python script before wrapping it in Kafka, Airflow, Spark, MinIO, and Docker. The math doesn't change later — only the plumbing around it does.

Files in this step:
- `step1_price_features.py` — Extract OHLCV price data from Yahoo Finance, Transform it into the technical-indicator feature vector **X_t** (SMA, RSI, OBV, ADX), Load the result to a local file.
- `requirements.txt` — the Python packages this script needs.

## Step 0 — Check whether Python is already installed

Open **Command Prompt** (search "cmd" in the Start menu) and run:

```
python --version
```

If you see something like `Python 3.11.5`, you're set — skip to Step 1.

If you see `'python' is not recognized...`, try `py --version` instead — Windows sometimes installs the `py` launcher under a different name. If neither works, Python isn't installed yet.

## Step 1 — Install Python (only if Step 0 came back empty)

Download the installer from **python.org/downloads** and run it. On the first install screen, **check the box "Add python.exe to PATH"** before clicking Install — this is the single most common setup mistake; without it, `python` won't be recognized in Command Prompt no matter how many times you reinstall. Then close and reopen Command Prompt and re-run `python --version` to confirm.

## Step 2 — Create a virtual environment

A virtual environment is a self-contained copy of Python just for this project, so the packages it needs (a specific version of `pandas`, for example) can't clash with packages some other Python project on your computer needs. It's a one-time setup per project, not per session.

In Command Prompt, navigate into this folder and create the environment:

```
cd "C:\Users\joann\OneDrive\Documents\Claude\Projects\Stock Price Prediction\etl_pipeline"
python -m venv venv
venv\Scripts\activate
```

You'll know it worked when your prompt line starts showing `(venv)` in front of it. You'll need to run that `activate` line again every time you open a new Command Prompt window to work on this project — it doesn't stay active permanently.

## Step 3 — Install the dependencies

With `(venv)` showing in your prompt:

```
pip install -r requirements.txt
```

This installs `yfinance` (talks to Yahoo Finance, no API key needed), `pandas`/`numpy` (data handling and the indicator math), and `matplotlib` (the sanity-check plot).

## Step 4 — Run it

```
python step1_price_features.py
```

## What you should see

The script prints its progress through seven labeled stages — Extract, Validate, Transform, Clean, Assemble, Plot, Load — and finishes by creating a new `data/` subfolder containing:

- `X_t_AAPL_sanity_check.png` — open this and check it looks like a real stock chart: a price line with a smoother SMA line tracking it, RSI oscillating between roughly 20 and 80, and ADX moving up during clear trending stretches and down during choppy/sideways ones.
- `X_t_AAPL.csv` — the actual feature table: one row per trading day, columns `SMA_20`, `RSI_14`, `OBV`, `ADX_14`, `ticker`. This is X_t.

## Troubleshooting

- **`'python' is not recognized`** — try `py` instead of `python` in every command above.
- **`pip install` fails or hangs** — usually a network/proxy issue; check your internet connection, or if you're on a work/school network, ask whether it blocks package installs.
- **No plot window pops up** — that's expected, not a bug. The script saves the plot straight to a PNG file instead of popping up a window, so it works the same whether you run it from Command Prompt, VS Code, or anywhere else.

## What's next

Once you've confirmed the plot and CSV look right, let me know and we'll do Step 2 the same way: pulling headlines (NewsAPI) and posts (Reddit) — the first time we'll deal with API keys — then running FinBERT to get **S_t** (sentiment) and **H_t** (entropy). After both halves of the logic are proven, we start layering in Kafka, Airflow, Spark, MinIO, and PostgreSQL around them.
