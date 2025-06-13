#!/usr/bin/env python
# -------------------------------------------------------------
# train_transformer.py  – BERT 3-class sentiment fine-tuning
# -------------------------------------------------------------
import argparse, logging, sys, re, string, unicodedata, pathlib
import numpy as np, pandas as pd, torch
from datasets import Dataset
import transformers, evaluate
from sklearn.metrics import classification_report, confusion_matrix

# ---------- logging ---------------------------------------------------------#
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("transformer_bert.log", "w", "utf-8"),
              logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------- tweet cleaner ---------------------------------------------------#
URL, MENTION = re.compile(r"https?://\S+|www\.\S+"), re.compile(r"@\w+")
HASHTAG, MULTI = re.compile(r"#(\w+)"), re.compile(r"\s+")

def clean(txt: str) -> str:
    txt = unicodedata.normalize("NFKD", str(txt)).lower()
    txt = URL.sub("<URL>", txt)
    txt = MENTION.sub("<USER>", txt)
    txt = txt.translate(str.maketrans("", "", string.punctuation))
    txt = HASHTAG.sub(r"\1", txt)
    txt = MULTI.sub(" ", txt).strip()
    return txt or "<EMPTY>"

def load_split(csv_path: pathlib.Path) -> Dataset:
    df = pd.read_csv(csv_path)
    text_col = next(c for c in ("tweet", "text") if c in df.columns)
    if "sentiment" in df.columns:
        df["label"] = df["sentiment"].map({"negative": 0, "neutral": 1, "positive": 2})
    else:
        df["label"] = df["target"].map({0: 0, 2: 1, 4: 2})
    df["text"] = df[text_col].astype(str).apply(clean)
    return Dataset.from_pandas(df[["text", "label"]])

# ---------- metric ----------------------------------------------------------#
metric_f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    macro = metric_f1.compute(predictions=preds,
                              references=labels,
                              average="macro")["f1"]
    acc = (preds == labels).mean()
    return {"macro_f1": macro, "accuracy": acc}

# ---------- main ------------------------------------------------------------#
def main(args):
    transformers.set_seed(args.seed)

    # load datasets
    train_ds = load_split(args.train_csv)
    val_ds   = load_split(args.valid_csv)
    test_ds  = load_split(args.test_csv)

    # tokenizer
    tok = transformers.AutoTokenizer.from_pretrained(args.model_name)
    def tokenize(batch):
        return tok(batch["text"],
                   padding="max_length",
                   truncation=True,
                   max_length=args.max_len)
    train_ds = train_ds.map(tokenize, batched=True).remove_columns(["text"])
    val_ds   = val_ds  .map(tokenize, batched=True).remove_columns(["text"])
    test_ds  = test_ds .map(tokenize, batched=True).remove_columns(["text"])

    # model
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=3)

    # training arguments
    training_args = transformers.TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        weight_decay=1e-5,
          eval_strategy="epoch",   # instead of evaluation_strategy
         save_strategy="epoch",
    
        logging_dir="./logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="tensorboard",
        seed=args.seed,
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # evaluation
    logger.info("Best checkpoint loaded. Evaluating …")
    for split_name, data in [("Validation", val_ds), ("Test", test_ds)]:
        metrics = trainer.evaluate(eval_dataset=data)
        logger.info("%s macro-F1 %.4f  |  accuracy %.4f",
                    split_name, metrics["eval_macro_f1"], metrics["eval_accuracy"])

    # detailed test report
    preds = np.argmax(trainer.predict(test_ds).predictions, axis=1)
    logger.info("\n%s",
                classification_report(test_ds["label"], preds,
                                      target_names=["neg", "neu", "pos"]))
    logger.info("Confusion matrix:\n%s",
                confusion_matrix(test_ds["label"], preds))

# ---------- CLI -------------------------------------------------------------#
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv",  default="data/train.csv")
    p.add_argument("--valid_csv",  default="data/valid.csv")
    p.add_argument("--test_csv",  default="data/test.csv")
    p.add_argument("--model_name", type=str, default="bert-base-uncased")
    p.add_argument("--model_dir",  type=str, default="./results_bert")
    p.add_argument("--max_len",    type=int, default=512)
    p.add_argument("--batch",      type=int, default=16)
    p.add_argument("--epochs",     type=int, default=4)
    p.add_argument("--lr",         type=float, default=2e-5)
    p.add_argument("--seed",       type=int, default=42)
    args = p.parse_args()
    main(args)
