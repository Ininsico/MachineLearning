# ============================================================
# WhatsAppLLM - Colab / Kaggle Training Script
# ============================================================
# Instructions:
#   1. Upload WhatsAppLLM_Full_Project.zip to your Google Drive
#   2. Open this in Colab: https://colab.research.google.com
#   3. Paste this whole file into a code cell and run it
# ============================================================

# ----------------------------------------
# 1. MOUNT DRIVE & EXTRACT
# ----------------------------------------
from google.colab import drive
import zipfile
import os

drive.mount("/content/drive")

ZIP_PATH = "/content/drive/MyDrive/WhatsAppLLM_Full_Project.zip"
EXTRACT_PATH = "/content/WhatsAppLLM_Full_Project"

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    z.extractall(EXTRACT_PATH)

os.chdir(f"{EXTRACT_PATH}/src")
print("Extracted!")

# ----------------------------------------
# 2. INSTALL DEPENDENCIES
# ----------------------------------------
!pip install tokenizers matplotlib

# ----------------------------------------
# 3. VERIFY GPU
# ----------------------------------------
import torch
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ----------------------------------------
# 4. TRAINING
# ----------------------------------------
import pickle
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import ChatDataset
from model.gpt import GPTLanguageModel
from checkpoint import save_checkpoint, load_checkpoint

VOCAB_SIZE = 8000
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
EPOCHS = 5
CHECKPOINT_EVERY = 500
EVAL_EVERY = 250

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

train_dataset = ChatDataset("../data/train.pkl")
test_dataset = ChatDataset("../data/test.pkl")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = GPTLanguageModel(vocab_size=VOCAB_SIZE).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

start_epoch = 0
global_batch = 0
train_losses = []
val_losses = []

checkpoint_path = "../models/checkpoint.pth"
if os.path.exists(checkpoint_path):
    loaded_epoch, global_batch, _, train_losses, val_losses = load_checkpoint(
        model, optimizer, checkpoint_path
    )
    start_epoch = loaded_epoch
    print(f"Resumed from epoch {start_epoch}, batch {global_batch}")

def evaluate():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            B, T, C = logits.shape
            loss = criterion(logits.view(B * T, C), y.view(B * T))
            total_loss += loss.item()
    model.train()
    return total_loss / len(test_loader)

total_batches = len(train_loader)
print(f"Training: {total_batches} batches/epoch x {EPOCHS} epochs = {total_batches * EPOCHS} total batches")
print(f"Batch size: {BATCH_SIZE}")

start_time = time.time()

for epoch in range(start_epoch, EPOCHS):
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        B, T, C = logits.shape
        loss = criterion(logits.view(B * T, C), y.view(B * T))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_batch += 1

        if global_batch % EVAL_EVERY == 0:
            val_loss = evaluate()
            train_losses.append(loss.item())
            val_losses.append(val_loss)
            elapsed = time.time() - start_time
            print(f"Epoch {epoch}/{EPOCHS-1} | Batch {global_batch} | Train: {loss.item():.4f} | Val: {val_loss:.4f} | Time: {elapsed:.1f}s")

        if global_batch % CHECKPOINT_EVERY == 0:
            save_checkpoint(model, optimizer, epoch, global_batch, loss.item(),
                            train_losses, val_losses, checkpoint_path)

    # Save checkpoint at end of each epoch
    save_checkpoint(model, optimizer, epoch, global_batch, loss.item(),
                    train_losses, val_losses, checkpoint_path)

# Final save
torch.save(model.state_dict(), "../models/gpt_model.pth")
with open("../models/losses.pkl", "wb") as f:
    pickle.dump((train_losses, val_losses), f)

elapsed = time.time() - start_time
print(f"\nDone! {elapsed:.1f}s ({elapsed/3600:.2f}h)")
print(f"Final train losses: {train_losses[-5:]}")
print(f"Final val losses:   {val_losses[-5:]}")
