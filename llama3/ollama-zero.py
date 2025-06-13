#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
zero_shot_ollama_eval_write_preds.py

Zero‐shot sentiment eval using Ollama’s `chat(...)` API. 
After evaluating on Test, append a new column "pred" (0=negative,1=neutral,2=positive) into test.csv.
"""
import argparse
import pathlib
import logging
import sys
import re
import string
import unicodedata

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from ollama import chat, ChatResponse

# ─────────── Logging ───────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("ollama-zero-gemma1b.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ─────────── Basic cleaner ───────────
URL, MENTION = re.compile(r"https?://\S+|www\.\S+"), re.compile(r"@\w+")
HASHTAG, WS   = re.compile(r"#(\w+)"), re.compile(r"\s+")

def clean(text: str) -> str:
    t = unicodedata.normalize("NFKD", str(text)).lower()
    t = URL.sub("<URL>", t)
    t = MENTION.sub("<USER>", t)
    # Note: We do not remove punctuation or hashtags here,
    # because model might rely on emojis / punctuation. Adjust if needed.
    # t = HASHTAG.sub(r"\1", t)
    return WS.sub(" ", t).strip() or "<EMPTY>"

LABEL_STR2ID = {"negative": 0, "neutral": 1, "positive": 2}

# ─────────── Load CSV into DataFrame ───────────
def load_df(path: pathlib.Path) -> pd.DataFrame:
    """
    Returns a DataFrame with columns ["text", "label"].
    - If 'sentiment' column exists, map it to 0/1/2 via LABEL_STR2ID.
    - Otherwise, look for an integer column 'label' or 'target'.
    - Then apply `clean(...)` to text column.
    """
    df = pd.read_csv(path)
    # detect text column:
    text_col = next(c for c in ("tweet", "text") if c in df.columns)
    # detect label column:
    if "sentiment" in df.columns:
        df["label"] = df["sentiment"].str.lower().map(LABEL_STR2ID)
    else:
        lbl_col = next(c for c in ("label", "target") if c in df.columns)
        df["label"] = df[lbl_col].astype(int)
    df["text"] = df[text_col].astype(str).apply(clean)
    return df[["text", "label"]]

# ─────────── Zero‐shot evaluation via Ollama ───────────
def predict_only_via_ollama(df: pd.DataFrame) -> np.ndarray:
    """
    For each row in df, call Ollama.chat(...) with:
      system: "You are a sentiment classifier. Your task is to classify..."
      user:   df["text"]
    Then parse single‐word response into {0,1,2}. Return np.array of preds.
    """
    all_preds = []
    system_msg = (
        "You are a sentiment classifier. "
        "Your task is to classify the given input into one of exactly three sentiment labels: "
        "\"negative\", \"neutral\", or \"positive\". "
        "Respond with only one of these words — no explanations, punctuation, or additional text."
    )

    for idx, row in df.iterrows():
        user_msg = row["text"]

        try:
            response: ChatResponse = chat(
                model="gemma3:12b",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_msg},
                ],
                stream=False,
            )
        except Exception as e:
            log.error("Ollama.chat failed on row %d: %s", idx, e)
            # If Ollama errors out, default to neutral:
            all_preds.append(LABEL_STR2ID["neutral"])
            continue

        # Extract the content string:
        reply = response.message.content.strip().lower()

        # Map to integer label:
        if "negative" in reply:
            pred_id = LABEL_STR2ID["negative"]
        elif "neutral" in reply:
            pred_id = LABEL_STR2ID["neutral"]
        elif "positive" in reply:
            pred_id = LABEL_STR2ID["positive"]
        else:
            # If Ollama returns something unexpected, default to neutral:
            log.warning(
                "Unexpected Ollama reply on row %d: '%s' → defaulting to 'neutral'", 
                idx, reply
            )
            pred_id = LABEL_STR2ID["neutral"]

        all_preds.append(pred_id)

    return np.array(all_preds, dtype=int)

# ─────────── Main ───────────
def main(args):
    # 1) Load valid/test into DataFrames
    log.info("▶ Loading CSVs …")
    valid_df = load_df(args.valid_csv)
    test_df_raw = pd.read_csv(args.test_csv)  # keep raw for writing later
    test_df = load_df(args.test_csv)

    # 2) Predict on Valid, log metrics
    log.info("▶ Running zero‐shot on Valid …")
    valid_preds = predict_only_via_ollama(valid_df)
    valid_labels = valid_df["label"].to_numpy()
    valid_f1 = f1_score(valid_labels, valid_preds, average="macro")
    valid_acc = (valid_preds == valid_labels).mean()
    log.info("► Valid → Macro‐F1: %.4f   Accuracy: %.4f", valid_f1, valid_acc)

    # 3) Predict on Test, log metrics and write preds back to CSV
    log.info("▶ Running zero‐shot on Test …")
    test_preds = predict_only_via_ollama(test_df)
    test_labels = test_df["label"].to_numpy()
    test_f1 = f1_score(test_labels, test_preds, average="macro")
    test_acc = (test_preds == test_labels).mean()
    log.info("► Test  → Macro‐F1: %.4f   Accuracy: %.4f", test_f1, test_acc)
    log.info(
        "\n%s",
        classification_report(
            test_labels, test_preds, target_names=["neg", "neu", "pos"]
        ),
    )
    log.info("Confusion matrix:\n%s", confusion_matrix(test_labels, test_preds))

    # 4) Append integer predictions into original test.csv
    log.info("▶ Writing integer predictions into %s …", args.test_csv)
    test_df_raw["pred"] = test_preds.tolist()
    test_df_raw.to_csv(args.test_csv, index=False)
    log.info("✓ Done.  Check ollama-zero.log and updated %s", args.test_csv)


# ─────────── CLI ───────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--valid_csv", type=pathlib.Path, default="data/valid.csv"
    )
    parser.add_argument(
        "--test_csv",  type=pathlib.Path, default="data/test.csv"
    )
    args = parser.parse_args()
    main(args)
