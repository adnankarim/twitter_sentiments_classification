
import argparse, logging, sys, re, string, unicodedata, pathlib, joblib, emoji, contractions
import numpy as np, pandas as pd, nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.utils import Bunch

# ---------- logging ---------------------------------------------------------#
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("tfidf.log", "w", "utf-8"),
              logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------- cleaning utils --------------------------------------------------#
URL_RE,    MENTION_RE   = re.compile(r"https?://\S+|www\.\S+"), re.compile(r"@\w+")
HASHTAG_RE, MULTISPACE_RE = re.compile(r"#(\w+)"), re.compile(r"\s+")
NEGATORS = {...}   # (same set as earlier)
try: nltk.data.find("corpora/stopwords")
except LookupError: nltk.download("stopwords")
STOP = set(nltk.corpus.stopwords.words("english"))

class TweetCleaner:
    def __init__(self, drop_stop=False):
        self.drop_stop = drop_stop
        self.punct = str.maketrans("", "", string.punctuation.replace("'", ""))

    def __call__(self, txt:str)->str:
        txt = unicodedata.normalize("NFKD", str(txt)).lower()
        txt = contractions.fix(txt)
        txt = URL_RE.sub("<URL>", txt)
        txt = MENTION_RE.sub("<USER>", txt)
        txt = emoji.demojize(txt, delimiters=("", ""))
        txt = re.sub(r"(.)\1{2,}", r"\1\1", txt)     # looove → loove
        txt = txt.translate(self.punct)
        txt = HASHTAG_RE.sub(r"\1", txt)
        txt = MULTISPACE_RE.sub(" ", txt).strip()

        toks, neg = [], False
        for tok in txt.split():
            if tok in NEGATORS:
                neg, toks = True, toks+[tok]; continue
            if re.fullmatch(r"[.!?,;]", tok): neg = False
            if neg: tok = "NOT_"+tok
            if not (self.drop_stop and tok in STOP):
                toks.append(tok)
        return " ".join(toks)

# ---------- data loader -----------------------------------------------------#
def load_csv(path: pathlib.Path)->Bunch:
    df = pd.read_csv(path)
    text_col = next(c for c in ("tweet","text") if c in df.columns)
    if "sentiment" in df.columns:
        y = df["sentiment"].map({"negative":0,"neutral":1,"positive":2})
    else:
        y = df["target"].map({0:0,2:1,4:2})
    cleaner = TweetCleaner()
    df["clean"] = df[text_col].astype(str).apply(cleaner)
    df = df[["clean"]].assign(label=y).dropna()
    logger.info("%s loaded: %d rows", path.name, len(df))
    return Bunch(text=df["clean"].tolist(), y=df["label"].values)

# ---------- model pipeline --------------------------------------------------#
def build_pipeline():
    tfidf = TfidfVectorizer(lowercase=False, min_df=2,
                            ngram_range=(1,1), sublinear_tf=True, norm="l2")
    clf   = LogisticRegression(multi_class="ovr", solver="liblinear",
                               max_iter=2000, n_jobs=-1)
    return Pipeline([("tfidf",tfidf), ("clf",clf)])

def grid_search(pipe,X,y,seed):
    grid = {
        "tfidf__ngram_range":[(1,1),(1,2)],
        "clf__C":[0.25,0.5,1,2],
        "clf__penalty":["l2","l1"],
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    gs = GridSearchCV(pipe, grid, scoring="f1_macro",
                      cv=cv, n_jobs=-1, verbose=2, refit=True)
    gs.fit(X,y)
    return gs

# ---------- feature inspection ---------------------------------------------#
def top_terms(model,n=15):
    vec, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    feats = np.array(vec.get_feature_names_out())
    for idx,label in enumerate(("NEG","NEU","POS")):
        coef = clf.coef_[idx]
        pos = feats[np.argsort(coef)[-n:][::-1]]
        neg = feats[np.argsort(coef)[:n]]
        logger.info("\n%s +weights: %s", label, ", ".join(pos))
        logger.info("%s -weights: %s", label, ", ".join(neg))

# ---------- main ------------------------------------------------------------#
def main(a):
    train = load_csv(a.train_csv)
    val   = load_csv(a.val_csv)
    test  = load_csv(a.test_csv)

    pipe  = build_pipeline()
    logger.info("Starting 5-fold grid search …")
    gs = grid_search(pipe, train.text, train.y, a.seed)
    logger.info("Best CV macro-F1 %.4f  |  params %s",
                gs.best_score_, gs.best_params_)

    # evaluation on val & test
    logger.info("\n=== Validation set ===")
    val_pred = gs.predict(val.text)
    logger.info("Macro-F1 = %.4f", f1_score(val.y, val_pred, average="macro"))
    logger.info("\n%s", classification_report(val.y, val_pred,
                                              target_names=["neg","neu","pos"]))

    logger.info("\n=== Test set ===")
    tst_pred = gs.predict(test.text)
    logger.info("Macro-F1 = %.4f", f1_score(test.y, tst_pred, average="macro"))
    logger.info("\n%s", classification_report(test.y, tst_pred,
                                              target_names=["neg","neu","pos"]))
    logger.info("Confusion matrix:\n%s", confusion_matrix(test.y, tst_pred))

    # interpretability
    top_terms(gs.best_estimator_)

    # save
    a.model_dir.mkdir(parents=True, exist_ok=True)
    out = a.model_dir / "logreg_tfidf.joblib"
    joblib.dump(gs.best_estimator_, out)
    logger.info("Model saved → %s", out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=pathlib.Path, required=True)
    p.add_argument("--val_csv",   type=pathlib.Path, required=True)
    p.add_argument("--test_csv",  type=pathlib.Path, required=True)
    p.add_argument("--model_dir", type=pathlib.Path,
                   default=pathlib.Path("./baseline_model"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)


# python tfidf.py --train_csv ./data/train.csv --val_csv   ./data/valid.csv  --test_csv  ./data/test.csv --model_dir ./baseline_model
