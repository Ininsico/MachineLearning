import os
import pickle
import time

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from dataloader import ChatDataset
from model.gpt import GPTLanguageModel
from checkpoint import save_checkpoint, load_checkpoint


VOCAB_SIZE = 8000
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
EPOCHS = 5
CHECKPOINT_EVERY = 500
EVAL_EVERY = 250


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# =====================
# DATASETS
# =====================

train_dataset = ChatDataset(
    "data/train.pkl"
)

test_dataset = ChatDataset(
    "data/test.pkl"
)


# =====================
# DATALOADERS
# =====================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =====================
# MODEL
# =====================

model = GPTLanguageModel(
    vocab_size=VOCAB_SIZE
).to(device)


# =====================
# LOSS + OPTIMIZER
# =====================

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# =====================
# RESUME FROM CHECKPOINT
# =====================

start_epoch = 0
global_batch = 0
train_losses = []
val_losses = []

checkpoint_path = "models/checkpoint.pth"

if os.path.exists(checkpoint_path):

    loaded_epoch, global_batch, _, \
        train_losses, val_losses = load_checkpoint(
            model, optimizer, checkpoint_path
        )

    start_epoch = loaded_epoch


# =====================
# EVALUATION
# =====================

def evaluate():

    model.eval()

    total_loss = 0

    with torch.no_grad():

        for x, y in test_loader:

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            B, T, C = logits.shape

            loss = criterion(
                logits.view(B * T, C),
                y.view(B * T)
            )

            total_loss += loss.item()

    model.train()

    return total_loss / len(test_loader)


# =====================
# TRAINING LOOP
# =====================

total_batches = len(train_loader)

print(
    f"Starting training: {total_batches} batches, "
    f"{EPOCHS} epochs, "
    f"{total_batches * EPOCHS} total batches"
)

start_time = time.time()

for epoch in range(start_epoch, EPOCHS):

    for batch_idx, (x, y) in enumerate(train_loader):

        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        B, T, C = logits.shape

        loss = criterion(
            logits.view(B * T, C),
            y.view(B * T)
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        global_batch += 1

        if global_batch % EVAL_EVERY == 0:

            val_loss = evaluate()

            train_losses.append(
                loss.item()
            )

            val_losses.append(
                val_loss
            )

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch}/{EPOCHS - 1} | "
                f"Batch {global_batch} | "
                f"Train Loss: {loss.item():.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

        if global_batch % CHECKPOINT_EVERY == 0:

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                batch_idx=global_batch,
                loss=loss.item(),
                train_losses=train_losses,
                val_losses=val_losses,
                filename=checkpoint_path
            )


# =====================
# SAVE FINAL MODEL
# =====================

torch.save(
    model.state_dict(),
    "models/gpt_model.pth"
)

with open(
    "models/losses.pkl",
    "wb"
) as f:

    pickle.dump(
        (
            train_losses,
            val_losses
        ),
        f
    )

elapsed = time.time() - start_time

print(f"Training complete! Total time: {elapsed:.1f}s")
print("Train Losses:", train_losses)
print("Val Losses:", val_losses)
print("Model Saved!")
