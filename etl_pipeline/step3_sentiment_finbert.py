"""
Step3 -- FinBERT Sentiment Scoring
Loads the raw Reddit CSV, scores each post with FinBERT,
and aggregates to daily S_t(sentiment) and H_t(entropy)
"""


import os
import numpy as np
import pandas as pd
from transformers import pipeline # transformers is the HuggingFace library that gives us FinBERT

TICKER     = "AAPL"
DATA_DIR   = "data"
INPUT_CSV  = os.path.join(DATA_DIR, f"S_t_raw_{TICKER}_reddit.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, f"S_t_{TICKER}_sentiment.csv")

MODEL_NAME = "ProsusAI/finbert" # is the HuggingFace identifier for FinBERT

def load_model():
    return pipeline(
        "text-classification", # tells HuggingFace what type of task this is
        model=MODEL_NAME, 
        top_k=None, # return all three probabilities (positive, negative, neutral)
        truncation=True, # automatically cut text at 512 tokens (BERT's hard limit)
        max_length=512,
        device=-1, # means use CPU, 0 would mean GPU
    )

MAX_CHARS = 2000

def prepare_text(title: str, selftext: str) -> str:
    title    = title.strip()    if title    else "" # strip() removes the trailing spaces and if title else "" handles None( if the field is missing, treat it as an empty string)
    selftext = selftext.strip() if selftext else ""

    if selftext:
        combined = title + ". " + selftext
    else:
        combined = title

    return combined[:MAX_CHARS] # cuts it to 2000 characters maximum so we donn't exceed BERT's token limmit


def score_text(text: str, classifier) -> dict:
    if not text.strip(): # if text is empty, return neutral defaults
        return {"sentiment": 0.0, "entropy": 0.0,
                "p_pos": 0.0, "p_neg": 0.0, "p_neu": 1.0}

    results = classifier(text)[0] #runs FinBERT on text, [0] gets the first item. The results look like [{"label":"positive", "score":"0.8"}, {"label":"neutral", "score": "0.8"}]
    probs   = {r["label"].lower(): r["score"] for r in results}

    p_pos = probs.get("positive", 0.0)
    p_neg = probs.get("negative", 0.0)
    p_neu = probs.get("neutral",  0.0)

    sentiment = p_pos - p_neg # S_t

    probs_arr = np.array([p_pos, p_neg, p_neu])
    probs_arr = np.clip(probs_arr, 1e-9, 1.0) # prevents log(0) which would give negative infinity, so wwwe set any zero to a tiny number(0.0000001) instead.
    entropy   = -np.sum(probs_arr * np.log(probs_arr)) # Shannon entropy formula: H = -Σ p·log(p). High entropy (close to log(3) ≈ 1.099) means FinBERT is split evenly across all three labels — uncertain. Low entropy means it's confident in one label.

    return {"sentiment": sentiment, "entropy": entropy,
            "p_pos": p_pos, "p_neg": p_neg, "p_neu": p_neu}


def main():
    print(f"Loading {INPUT_CSV}.....")
    df = pd.read_csv(INPUT_CSV)
    print(f" {len(df)} posts loaded.")

    classifier = load_model()

    scores = []
    for i, row in df.iterrows():
        text = prepare_text(str(row["title"]), str(row["selftext"]))
        result = score_text(text, classifier)
        result["trading_day"] = row["trading_day"]
        result["score"] = row["score"]
        scores.append(result)

        if (i + 1) % 100 == 0:
            print(f" Scored {i + 1} / {len(df)} posts...")

    scores_df = pd.DataFrame(scores)

    daily = scores_df.groupby("trading_day").agg(
        sentiment = ("sentiment", "mean"),
        entropy = ("entropy", "mean"),
        post_count = ("sentiment", "count"),
        avg_score = ("score", "mean"),
    ).reset_index()

    daily["ticker"] = TICKER
    daily = daily.sort_values("trading_day").reset_index(drop = True)

    daily.to_csv(OUTPUT_CSV, index = False)
    print(f"\nSaved {len(daily)} daily rows to {OUTPUT_CSV}")
    print(daily.head(10))

if __name__ == "__main__":
    main()

    