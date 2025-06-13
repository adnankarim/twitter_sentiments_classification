#!/usr/bin/env python
# -------------------------------------------------------------
# train_sentence_cnn.py
#
#  • Loads *.npy produced by embed_with_flag.py
#  • Simple 1-token Conv1d (equivalent to linear) → ReLU → Dropout
#  • Early stop on validation macro-F1
#
#   pip install torch numpy scikit-learn
# -------------------------------------------------------------
import argparse, pathlib, numpy as np, torch, torch.nn as nn, time, math, logging, sys
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("cnn.log","a","utf-8"), logging.StreamHandler(sys.stdout)]
)
logger=logging.getLogger(__name__)

class SentenceCNN(nn.Module):
    def __init__(self, in_dim:int, hidden:int=384, n_cls:int=3, dropout=0.5):
        super().__init__()
        self.conv = nn.Conv1d(1, hidden, kernel_size=1)  # (B,1,1024) -> (B,H,1024)
        self.act  = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)              # global max over dim
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, n_cls)
    def forward(self,x):                                 # x: (B,1024)
        x = x.unsqueeze(1)                               # (B,1,1024)
        x = self.pool(self.act(self.conv(x))).squeeze(-1) # (B,H)
        return self.fc(self.drop(x))

def load_split(dir: pathlib.Path, split:str):
    X=np.load(dir/f"X_{split}.npy"); y=np.load(dir/f"y_{split}.npy")
    return torch.tensor(X,dtype=torch.float32), torch.tensor(y,dtype=torch.long)

def trians(model,loader,loss_fn,opt,device):
    train = opt is not None; model.train(train)
    tot_loss, tot_f1, n=0.,0.,0
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        logits = model(X); loss = loss_fn(logits,y)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        tot_loss+=loss.item()*len(X)
        tot_f1+=f1_score(y.cpu(),logits.argmax(1).cpu(),average="macro")*len(X)
        n+=len(X)
    return tot_loss/n, tot_f1/n

def main(a):
    device="cuda" if torch.cuda.is_available() else "cpu"
    Xtr,ytr=load_split(a.cache,"train")
    Xv,yv  =load_split(a.cache,"valid")
    Xt,yt  =load_split(a.cache,"test")

    train_ld=DataLoader(TensorDataset(Xtr,ytr),batch_size=a.batch,shuffle=True)
    val_ld  =DataLoader(TensorDataset(Xv,yv), batch_size=a.batch)
    test_ld =DataLoader(TensorDataset(Xt,yt), batch_size=a.batch)

    model=SentenceCNN(in_dim=Xtr.shape[1], hidden=512, dropout=0.5).to(device)
    logger.info(model)
    opt=torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-5)
    loss_fn=nn.CrossEntropyLoss()

    best_F1, patience=0.,0
    a.model_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1,a.epochs+1):
        t0=time.time()
        tr_loss,tr_f1=trians(model,train_ld,loss_fn,opt,device)
        val_loss,val_f1=trians(model,val_ld,loss_fn,None,device)
        logger.info(f"Epoch {epoch:02d} | "
                    f"train_loss {tr_loss:.4f}  train_F1 {tr_f1:.4f} | "
                    f"val_loss {val_loss:.4f}  val_F1 {val_f1:.4f} | "
                    f"time {time.time()-t0:.1f}s")
        if val_f1>best_F1+1e-4:
            best_F1=val_f1; patience=0
            torch.save(model.state_dict(), a.model_dir/"best_cnn.pt")
        else:
            patience+=1
            if patience>=20:
                logger.info("Early stopping.")
                break

    # final evaluation
    def evaluate(loader):
        model.eval(); preds=[]
        with torch.no_grad():
            for X,y in loader:
                preds.extend(model(X.to(device)).argmax(1).cpu().numpy())
        return np.array(preds)
    val_pred=evaluate(val_ld); test_pred=evaluate(test_ld)

    logger.info("\n=== Validation ===  Macro-F1 %.4f", f1_score(yv,val_pred,average="macro"))
    logger.info("\n=== Test ===  Macro-F1 %.4f", f1_score(yt,test_pred,average="macro"))
    logger.info("\n%s", classification_report(yt,test_pred,target_names=["neg","neu","pos"]))
    logger.info("Confusion matrix:\n%s", confusion_matrix(yt,test_pred))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--cache", type=pathlib.Path, default=pathlib.Path("flag_cache"))
    p.add_argument("--model_dir", type=pathlib.Path, default=pathlib.Path("./baseline_cnn"))
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args=p.parse_args(); main(args)
