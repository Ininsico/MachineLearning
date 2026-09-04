import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src import config
from src import data as data_module
from src.models import NEOMLP


def train_torch_model(splits, device=config.DEVICE, verbose=True):
    X_train = torch.tensor(splits["X_train"].values, dtype=torch.float32)
    y_train = torch.tensor(splits["y_train"], dtype=torch.float32)
    X_val = torch.tensor(splits["X_val"].values, dtype=torch.float32)
    y_val = torch.tensor(splits["y_val"], dtype=torch.float32)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_dl = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    model = NEOMLP(n_features=X_train.shape[1]).to(device)
    pos_weight = torch.tensor([data_module.get_pos_weight(splits["y_train"])], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)

    best_val_loss = float("inf")
    best_state = None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.shape[0]
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * xb.shape[0]
        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose:
            print(f"Epoch {epoch+1:02d}/{config.NUM_EPOCHS} | "
                  f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f}")

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), config.MODELS_DIR / "neo_mlp.pt")
    return model, history


@torch.no_grad()
def predict_torch(model, X, device=config.DEVICE):
    model.eval()
    Xt = torch.tensor(X.values, dtype=torch.float32).to(device)
    logits = model(Xt)
    probs = torch.sigmoid(logits).cpu().numpy()
    return probs
