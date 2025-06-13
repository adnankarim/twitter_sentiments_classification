#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
finetune_tabularis_3class.py

Fine-tunes a three-class (“negative”, “neutral”, “positive”) DistilBERT-multilingual model
on your own train/valid/test CSV splits.

Each CSV is expected to have:
  • A text column named either “tweet” or “text”
  • A label column named either “sentiment” or “target”

By default we map string labels:
    "negative" → 0
    "neutral"  → 1
    "positive" → 2

If your CSV uses numeric labels 0–2 already, you can skip the string→int mapping.

Requirements:
    pip install transformers datasets scikit-learn torch pandas
"""

import argparse
import pathlib
import re
import string
import unicodedata
import logging
import sys

import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# ───────────────────────────────────────────────
# Logging setup (logs to stdout and to `training-tabularis.log`)
# ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("training-tabularis.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────
# LABEL ↔ ID MAPPING (3 classes)
# ───────────────────────────────────────────────
LABEL2ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ───────────────────────────────────────────────
# TEXT CLEANING (basic normalizations)
# ───────────────────────────────────────────────
URL_REGEX = re.compile(r"https?://\S+|www\.\S+")
MENTION_REGEX = re.compile(r"@\w+")
HASHTAG_REGEX = re.compile(r"#(\w+)")
MULTI_SPACE_REGEX = re.compile(r"\s+")

def clean_text(text: str) -> str:
    """
    Lowercase → unicode normalize → substitute URLs / @mentions → strip punctuation →
    collapse multiple spaces → return "<EMPTY>" if result is blank.
    """
    t = unicodedata.normalize("NFKD", str(text)).lower()
    t = URL_REGEX.sub("<URL>", t)
    t = MENTION_REGEX.sub("<USER>", t)
    t = t.translate(str.maketrans("", "", string.punctuation))
    t = HASHTAG_REGEX.sub(r"\1", t)  # remove leading '#'
    t = MULTI_SPACE_REGEX.sub(" ", t).strip()
    return t or "<EMPTY>"

# ───────────────────────────────────────────────
# LOAD CSV → HF Dataset
# ───────────────────────────────────────────────
def load_split(path: pathlib.Path) -> Dataset:
    """
    1) Read CSV into Pandas
    2) Detect text column (“tweet” or “text”)
    3) Detect label column (“sentiment” or “target”)
    4) Clean the text
    5) Map string labels → int (0–2). If CSV uses numeric 0–2 already, keep as is.
    6) Return a HuggingFace Dataset with fields: { "text": str, "label": int }
    """
    df = pd.read_csv(path)

    # 1) text column
    text_col = next(c for c in ("tweet", "text") if c in df.columns)

    # 2) label column
    if "sentiment" in df.columns:
        raw_labels = df["sentiment"].astype(str).str.lower()
        # If purely digits “0/1/2”, keep numeric; otherwise map via LABEL2ID
        if raw_labels.str.isdigit().all():
            labels = raw_labels.astype(int).to_numpy()
        else:
            labels = raw_labels.map(LABEL2ID).to_numpy()
    else:
        # assume Kaggle style with “target” = {0,2,4}, collapse: 0→0,2→1,4→2
        raw = df["target"].astype(int)
        mapper = {0: 0, 2: 1, 4: 2}
        labels = raw.map(mapper).to_numpy()

    # 3) Clean text
    df["text"] = df[text_col].astype(str).apply(clean_text)

    # 4) Build HF Dataset
    return Dataset.from_pandas(
        pd.DataFrame({"text": df["text"], "label": labels}),
        preserve_index=False
    )

# ───────────────────────────────────────────────
# TOKENIZE + ENCODE
# ───────────────────────────────────────────────
def tokenize_batch(batch, tokenizer, max_len):
    """
    Apply tokenizer(...) to a batch, producing input_ids & attention_mask.
    Keep “label” as is.
    """
    enc = tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=max_len,
    )
    enc["labels"] = batch["label"]
    return enc

# ───────────────────────────────────────────────
# COMPUTE METRICS (MACRO F1 + ACCURACY)
# ───────────────────────────────────────────────
def compute_metrics(eval_pred):
    """
    eval_pred: (logits, labels)
    Take argmax(logits, dim=-1) → preds, then compute macro-F1 + accuracy.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro = f1_score(labels, preds, average="macro")
    acc = (preds == labels).astype(int).mean()
    return {"macro_f1": macro, "accuracy": acc}

# ───────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────
def main(args):
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 1) Load splits
    logger.info("Loading train/valid/test CSV splits …")
    train_ds = load_split(pathlib.Path(args.train_csv))
    valid_ds = load_split(pathlib.Path(args.valid_csv))
    test_ds  = load_split(pathlib.Path(args.test_csv))

    # 2) Load tokenizer & model
    logger.info("Loading model & tokenizer from `tabularisai/multilingual-sentiment-analysis` …")
    tokenizer = AutoTokenizer.from_pretrained("tabularisai/multilingual-sentiment-analysis")
    model = AutoModelForSequenceClassification.from_pretrained(
        "tabularisai/multilingual-sentiment-analysis"
    )
    model.to(device)

    # 3) Tokenize each split
    logger.info("Tokenizing train dataset …")
    train_ds = train_ds.map(
        lambda batch: tokenize_batch(batch, tokenizer, args.max_len),
        batched=True,
        remove_columns=["text", "label"],
    )
    logger.info("Tokenizing valid dataset …")
    valid_ds = valid_ds.map(
        lambda batch: tokenize_batch(batch, tokenizer, args.max_len),
        batched=True,
        remove_columns=["text", "label"],
    )
    logger.info("Tokenizing test dataset …")
    test_ds = test_ds.map(
        lambda batch: tokenize_batch(batch, tokenizer, args.max_len),
        batched=True,
        remove_columns=["text", "label"],
    )

    # 4) Set format to PyTorch
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    valid_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_ds.set_format("torch",  columns=["input_ids", "attention_mask", "labels"])

    # 5) Data collator (pads at batch time)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 6) TrainingArguments + Trainer
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Disable intermediate checkpoint saving to avoid write errors
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="no",
        save_strategy="no",             # do NOT save during training
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir="logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 7) Train
    logger.info("Starting training …")
    trainer.train()
    logger.info("Training complete.")

    # 8) Manually save best model + tokenizer
    logger.info("Saving final model to %s …", str(output_dir / "best_model"))
    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))

    # 9) Evaluate on validation set
    logger.info("Evaluating on VALID set …")
    val_metrics = trainer.evaluate(valid_ds)
    logger.info("VALID metrics: %s", val_metrics)

    # 10) Evaluate on test set
    logger.info("Evaluating on TEST set …")
    test_metrics = trainer.evaluate(test_ds)
    logger.info("TEST metrics: %s", test_metrics)

    # Detailed classification report + confusion matrix on test set:
    logger.info("Generating detailed classification report on TEST set …")
    preds_output = trainer.predict(test_ds)
    test_preds = np.argmax(preds_output.predictions, axis=-1)
    test_labels = preds_output.label_ids

    cls_report = classification_report(
        test_labels,
        test_preds,
        target_names=["Negative", "Neutral", "Positive"],
    )
    conf_mat = confusion_matrix(test_labels, test_preds)

    logger.info("\n%s", cls_report)
    logger.info("Confusion matrix:\n%s", conf_mat)

    logger.info("All done. Logs are in training-tabularis.log")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_csv",
        type=str,
        default="data/train.csv",
        help="Path to train.csv",
    )
    parser.add_argument(
        "--valid_csv",
        type=str,
        default="data/valid.csv",
        help="Path to valid.csv",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="data/test.csv",
        help="Path to test.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out_multilingual_3class",
        help="Where to save checkpoints + logs",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Train batch size per device",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Eval batch size per device",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=256,
        help="Maximum tokenizer length (truncation/padding)",
    )
    args = parser.parse_args()
    main(args)
# python finetune_tabularis_3class.py \
#   --train_csv data/train.csv \
#   --valid_csv data/valid.csv \
#   --test_csv data/test.csv \
#   --output_dir out_multilingual_3class \
#   --epochs 3 \
#   --batch_size 16 \
#   --eval_batch_size 32 \
#   --lr 2e-5 \
#   --max_len 256
