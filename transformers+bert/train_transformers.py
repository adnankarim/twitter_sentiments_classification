
# train_transformer.py

import argparse, logging, sys, re, string, unicodedata, pathlib, numpy as np, pandas as pd
from datasets import Dataset
import torch, transformers, evaluate
from transformers import TrainingArguments

# ---------- logging ---------------------------------------------------------#
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("transformer.log", "w", "utf-8"),
              logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------- tweet cleaner (same rules) -------------------------------------#
URL, MENTION = re.compile(r"https?://\S+|www\.\S+"), re.compile(r"@\w+")
HASHTAG, MULTI = re.compile(r"#(\w+)"), re.compile(r"\s+")
def clean(txt:str)->str:
    txt = unicodedata.normalize("NFKD", str(txt)).lower()
    txt = URL.sub("<URL>", txt)
    txt = MENTION.sub("<USER>", txt)
    txt = txt.translate(str.maketrans("", "", string.punctuation))
    txt = HASHTAG.sub(r"\1", txt)
    txt = MULTI.sub(" ", txt).strip()
    return txt or "<EMPTY>"

def load_split(csv:pathlib.Path)->Dataset:
    df = pd.read_csv(csv)
    text_col = next(c for c in ("tweet","text") if c in df.columns)
    if "sentiment" in df.columns:
        df["label"] = df["sentiment"].map({"negative":0,"neutral":1,"positive":2})
    else:
        df["label"] = df["target"].map({0:0,2:1,4:2})
    df["text"] = df[text_col].astype(str).apply(clean)
    return Dataset.from_pandas(df[["text","label"]])

# ---------- metric ----------------------------------------------------------#
metric_f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    macro = metric_f1.compute(predictions=preds, references=labels,
                              average="macro")["f1"]
    return {"macro_f1": macro,
            "accuracy": (preds==labels).mean()}

# ---------- main ------------------------------------------------------------#
def main(a):
    train_ds = load_split(a.train_csv)
    val_ds   = load_split(a.valid_csv)
    test_ds  = load_split(a.test_csv)

    tok = transformers.AutoTokenizer.from_pretrained(a.model_name)
    def tokenize(batch):
        return tok(batch["text"],
                   padding="max_length",
                   truncation=True,
                   max_length=a.max_len)
    train_ds = train_ds.map(tokenize, batched=True).remove_columns(["text"])
    val_ds   = val_ds  .map(tokenize, batched=True).remove_columns(["text"])
    test_ds  = test_ds .map(tokenize, batched=True).remove_columns(["text"])

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        a.model_name, num_labels=3)

    # training args
    training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=9,
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="epoch",  # Changed from evaluation_strategy
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to='tensorboard',
    metric_for_best_model="accuracy"
)

    trainer = transformers.Trainer(
        model,
        training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # final evaluation
    val_metrics  = trainer.evaluate(val_ds)
    test_metrics = trainer.evaluate(test_ds)
    logger.info("Validation  macro-F1 %.4f", val_metrics["eval_macro_f1"])
    logger.info("Test        macro-F1 %.4f", test_metrics["eval_macro_f1"])

    # detailed report on test set
    preds = np.argmax(trainer.predict(test_ds).predictions, axis=1)
    from sklearn.metrics import classification_report, confusion_matrix
    logger.info("\n%s", classification_report(test_ds["label"], preds,
                                              target_names=["neg","neu","pos"]))
    logger.info("\nConfusion\n%s", confusion_matrix(test_ds["label"], preds))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv",  default="data/train.csv")
    p.add_argument("--valid_csv",  default="data/valid.csv")
    p.add_argument("--test_csv",  default="data/test.csv")
    p.add_argument("--model_name", type=str, default="BAAI/bge-base-en-v1.5")
    p.add_argument("--model_dir",  type=str, default="./baseline_transformer")
    p.add_argument("--max_len",    type=int, default=64)
    p.add_argument("--batch",      type=int, default=16)
    p.add_argument("--epochs",     type=int, default=100)
    p.add_argument("--lr",         type=float, default=2e-5)
    p.add_argument("--seed",       type=int, default=42)
    args = p.parse_args()
    main(args)
