import os
import copy
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from tqdm import tqdm

logger = logging.getLogger(__name__)

class EarlyStopping:
    def __init__(self, patience: int = 5, delta: float = 1e-4):
        self.patience  = patience
        self.delta     = delta
        self.best_val  = -np.inf
        self.counter   = 0
        self.best_state = None

    def __call__(self, val_metric, model):
        if val_metric > self.best_val + self.delta:
            self.best_val   = val_metric
            self.counter    = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)

def train_classifier(config, model, X_train, y_train, X_val, y_val,
                     synth_X: np.ndarray, device: torch.device) -> dict:

    synth_y = np.ones(len(synth_X), dtype=np.int64)
    X_aug   = np.vstack([X_train, synth_X]).astype(np.float32)
    y_aug   = np.concatenate([y_train, synth_y]).astype(np.float32)
    logger.info(" Classifier Training | Augmented train size: %d", len(X_aug))

    n_pos = y_aug.sum()
    n_neg = len(y_aug) - n_pos
    pos_weight = torch.tensor([n_neg / (n_pos + 1e-8)], device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_ds = TensorDataset(
        torch.tensor(X_aug, dtype=torch.float32),
        torch.tensor(y_aug, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=config.CLF_BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=config.CLF_BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.CLF_LR,
        weight_decay=config.CLF_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    early_stop = EarlyStopping(patience=7)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "train_f1":   [], "val_f1":   [],
    }

    for epoch in range(1, config.CLF_EPOCHS + 1):

        model.train()
        tr_loss, tr_preds, tr_true = 0.0, [], []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.CLF_EPOCHS} [Train]")
        for Xb, yb in pbar:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss   = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_loss += loss.item() * len(Xb)
            preds    = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
            tr_preds.extend(preds); tr_true.extend(yb.long().cpu().numpy())

        tr_loss /= len(train_ds)
        tr_acc   = (np.array(tr_preds) == np.array(tr_true)).mean()
        tr_f1    = f1_score(tr_true, tr_preds, zero_division=0)

        model.eval()
        vl_loss, vl_preds, vl_true = 0.0, [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits  = model(Xb)
                loss    = criterion(logits, yb)
                vl_loss += loss.item() * len(Xb)
                preds    = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
                vl_preds.extend(preds); vl_true.extend(yb.long().cpu().numpy())

        vl_loss /= len(val_ds)
        vl_acc   = (np.array(vl_preds) == np.array(vl_true)).mean()
        vl_f1    = f1_score(vl_true, vl_preds, zero_division=0)

        history["train_loss"].append(tr_loss);  history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc);    history["val_acc"].append(vl_acc)
        history["train_f1"].append(tr_f1);      history["val_f1"].append(vl_f1)

        scheduler.step(vl_f1)

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "  [Epoch %2d/%d]  tr_loss=%.4f  vl_loss=%.4f  "
                "tr_acc=%.4f  vl_acc=%.4f  vl_f1=%.4f",
                epoch, config.CLF_EPOCHS,
                tr_loss, vl_loss, tr_acc, vl_acc, vl_f1
            )

        if early_stop(vl_f1, model):
            logger.info("   Early stopping at epoch %d (best val_f1=%.4f)",
                        epoch, early_stop.best_val)
            break

    early_stop.restore(model)

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(config.MODELS_DIR, "classifier.pt"))
    logger.info(" Classifier saved to: %s", config.MODELS_DIR)

    return history

def predict(model, X: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:

    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            Xb     = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
            logits = model(Xb)
            preds  = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
            all_preds.extend(preds)
    return np.array(all_preds)

def predict_proba(model, X: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:

    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            Xb    = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
            probs = torch.sigmoid(model(Xb)).cpu().numpy()
            all_probs.extend(probs)
    return np.array(all_probs)