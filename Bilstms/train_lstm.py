#!/usr/bin/env python

# -------------------------------------------------------------
import argparse, pathlib, numpy as np, torch, torch.nn as nn, time, logging, sys
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# ---------- logging ---------------------------------------------------------#
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("bilstm.log", "w", "utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ---------- model -----------------------------------------------------------#
class SentenceBiLSTM(nn.Module):
    """
    Treat each 1024-D FlagEmbedding vector as a sequence length = 1.
    A Bi-LSTM with hidden//2 units in each direction yields a  ‘hidden’ vector,
    then dropout → linear classifier.
    """
    def __init__(self, in_dim: int, hidden: int = 384, n_cls: int = 3, dropout: float = 0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )  # output: (B, 1, hidden)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, n_cls)

    def forward(self, x):  # x: (B, 1024)
        # treat each vector as sequence length-1
        out, _ = self.lstm(x.unsqueeze(1))  # (B,1,hidden)
        out = out.squeeze(1)                # (B,hidden)
        return self.fc(self.drop(self.act(out)))

# ---------- I/O helpers -----------------------------------------------------#
def load_split(cache: pathlib.Path, split: str):
    X = np.load(cache / f"X_{split}.npy")
    y = np.load(cache / f"y_{split}.npy")
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def trains(model, loader, loss_fn, optimiser, device):
    train = optimiser is not None
    model.train(train)
    tot_loss = tot_f1 = n = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = loss_fn(logits, y)
        if train:
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        tot_loss += loss.item() * len(X)
        tot_f1 += f1_score(
            y.cpu().numpy(), logits.argmax(1).cpu().numpy(), average="macro"
        ) * len(X)
        n += len(X)
    return tot_loss / n, tot_f1 / n

# ---------- main ------------------------------------------------------------#
def main(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    Xtr, ytr = load_split(a.cache, "train")
    Xv, yv = load_split(a.cache, "valid")
    Xt, yt = load_split(a.cache, "test")

    train_ld = DataLoader(TensorDataset(Xtr, ytr), batch_size=a.batch, shuffle=True)
    val_ld = DataLoader(TensorDataset(Xv, yv), batch_size=a.batch)
    test_ld = DataLoader(TensorDataset(Xt, yt), batch_size=a.batch)

    model = SentenceBiLSTM(in_dim=Xtr.shape[1], hidden=384, dropout=0.5).to(device)
    logger.info(model)

    optimiser = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    best_F1, patience = 0.0, 0
    a.model_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        tr_loss, tr_f1 = trains(model, train_ld, loss_fn, optimiser, device)
        val_loss, val_f1 = trains(model, val_ld, loss_fn, None, device)
        logger.info(
            f"Epoch {epoch:02d} | "
            f"train_loss {tr_loss:.4f}  train_F1 {tr_f1:.4f} | "
            f"val_loss {val_loss:.4f}  val_F1 {val_f1:.4f} | "
            f"time {time.time() - t0:.1f}s"
        )

        if val_f1 > best_F1 + 1e-4:
            best_F1 = val_f1
            patience = 0
            torch.save(model.state_dict(), a.model_dir / "best_bilstm.pt")
        else:
            patience += 1
            if patience >= 20:
                logger.info("Early stopping.")
                break

    # ----- final evaluation -----
    def predict(loader):
        model.eval()
        out = []
        with torch.no_grad():
            for X, _ in loader:
                out.extend(model(X.to(device)).argmax(1).cpu().numpy())
        return np.array(out)

    val_pred = predict(val_ld)
    test_pred = predict(test_ld)

    logger.info(
        "\n=== Validation ===  Macro-F1 %.4f",
        f1_score(yv, val_pred, average="macro"),
    )
    logger.info(
        "\n=== Test ===  Macro-F1 %.4f", f1_score(yt, test_pred, average="macro")
    )
    logger.info(
        "\n%s",
        classification_report(yt, test_pred, target_names=["neg", "neu", "pos"]),
    )
    logger.info("Confusion matrix:\n%s", confusion_matrix(yt, test_pred))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=pathlib.Path, default=pathlib.Path("flag_cache3"))
    p.add_argument("--model_dir", type=pathlib.Path, default=pathlib.Path("./baseline_bilstm"))
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
