import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import random
import math
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
import wandb
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import multiprocessing as mp

class MegaG2PDataset(Dataset):
    def __init__(self, metadata_path, max_samples=None):
        self.df = pd.read_csv(metadata_path, sep='|', header=None, names=['id','text','phonemes'], on_bad_lines='skip')
        if max_samples:
            self.df = self.df.sample(max_samples, random_state=42)
        self._build_vocabs()
        self._preprocess_data()

    def _build_vocabs(self):
        self.char2idx = {'<pad>':0, '<sos>':1, '<eos>':2, '<unk>':3}
        self.idx2char = {v:k for k,v in self.char2idx.items()}
        self.phon2idx = {'<pad>':0, '<sos>':1, '<eos>':2, '<unk>':3}
        self.idx2phon = {v:k for k,v in self.phon2idx.items()}
        char_counter = 4
        all_text = ' '.join(self.df['text'].astype(str)).lower()
        unique_chars = set(all_text)
        for char in unique_chars:
            if char not in self.char2idx:
                self.char2idx[char] = char_counter
                self.idx2char[char_counter] = char
                char_counter += 1
        phon_counter = 4
        all_phonemes = ' '.join(self.df['phonemes'].astype(str)).split()
        unique_phonemes = set(all_phonemes)
        for phon in unique_phonemes:
            if phon not in self.phon2idx:
                self.phon2idx[phon] = phon_counter
                self.idx2phon[phon_counter] = phon
                phon_counter += 1

    def _preprocess_data(self):
        self.samples = []
        for _, row in self.df.iterrows():
            text = str(row['text']).lower()
            chars = [self.char2idx.get(c, self.char2idx['<unk>']) for c in text]
            phonemes = str(row['phonemes']).strip().split()
            phons = [self.phon2idx.get(p, self.phon2idx['<unk>']) for p in phonemes]
            if len(chars) > 1 and len(phons) > 1:
                self.samples.append((
                    torch.tensor(chars, dtype=torch.long),
                    torch.tensor([self.phon2idx['<sos>']] + phons + [self.phon2idx['<eos>']], dtype=torch.long)
                ))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

IS_KAGGLE = 'KAGGLE_URL_BASE' in os.environ
BASE_PATH = "/kaggle/input/ljsvoicedata/LJSpeech-1.1" if IS_KAGGLE else r"C:\Users\arsla\Downloads\archive (4)\LJSpeech-1.1"
METADATA = "metadata.csv"
MODEL_SAVE_PATH = '/kaggle/working/g2p_megatron.pth' if IS_KAGGLE else './g2p_megatron.pth'
EMB_DIM = 512
HIDDEN_DIM = 512
CNN_CHANNELS = 512
NUM_LAYERS = 4
N_HEADS = 8
DROPOUT = 0.2
BATCH_SIZE = 64 if torch.cuda.is_available() else 64
LABEL_SMOOTHING = 0.1
MAX_LR = 5e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 30
GRAD_CLIP = 1.0

assert EMB_DIM % N_HEADS == 0
assert HIDDEN_DIM == EMB_DIM
assert CNN_CHANNELS == EMB_DIM

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

try:
    wandb.init(project="ultimate-g2p", config={
        "emb_dim": EMB_DIM,
        "hidden_dim": HIDDEN_DIM,
        "cnn_channels": CNN_CHANNELS,
        "num_layers": NUM_LAYERS,
        "n_heads": N_HEADS,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "label_smoothing": LABEL_SMOOTHING,
        "max_lr": MAX_LR,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS
    }, mode="online" if not IS_KAGGLE else "disabled")
except:
    wandb.init(mode="disabled")

def collate_batch(batch):
    src = [item[0] for item in batch]
    tgt = [item[1] for item in batch]
    src_lens = torch.tensor([len(s) for s in src], dtype=torch.long)
    tgt_lens = torch.tensor([len(t) for t in tgt], dtype=torch.long)
    src_pad = pad_sequence(src, batch_first=True, padding_value=0)
    tgt_pad = pad_sequence(tgt, batch_first=True, padding_value=0)
    return src_pad, src_lens, tgt_pad, tgt_lens

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class CNNEncoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMB_DIM, padding_idx=0)
        self.pos_encoder = PositionalEncoding(EMB_DIM, DROPOUT)
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(EMB_DIM, EMB_DIM, kernel_size=5, padding=2),
                nn.BatchNorm1d(EMB_DIM),
                nn.GELU(),
                nn.Dropout(DROPOUT)
            ) for _ in range(3)
        ])
        encoder_layers = TransformerEncoderLayer(
            d_model=EMB_DIM,
            nhead=N_HEADS,
            dim_feedforward=EMB_DIM*4,
            dropout=DROPOUT,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers, num_layers=2)

    def forward(self, src, src_lens):
        embedded = self.embedding(src) * math.sqrt(EMB_DIM)
        embedded = self.pos_encoder(embedded)
        conv_out = embedded.permute(0, 2, 1)
        for conv in self.conv_layers:
            conv_out = conv(conv_out) + conv_out
        conv_out = conv_out.permute(0, 2, 1)
        src_key_padding_mask = (src == 0)
        transformer_out = self.transformer_encoder(conv_out, src_key_padding_mask=src_key_padding_mask)
        return transformer_out

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMB_DIM, padding_idx=0)
        self.pos_encoder = PositionalEncoding(EMB_DIM, DROPOUT)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=EMB_DIM,
                nhead=N_HEADS,
                dim_feedforward=EMB_DIM*4,
                dropout=DROPOUT,
                batch_first=True
            ) for _ in range(NUM_LAYERS)
        ])
        self.output_proj = nn.Sequential(
            nn.Linear(EMB_DIM, EMB_DIM),
            nn.GELU(),
            nn.LayerNorm(EMB_DIM),
            nn.Dropout(DROPOUT),
            nn.Linear(EMB_DIM, vocab_size)
        )
        self.ctc_proj = nn.Linear(EMB_DIM, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, memory_key_padding_mask=None):
        embedded = self.embedding(tgt) * math.sqrt(EMB_DIM)
        embedded = self.pos_encoder(embedded)
        output = embedded
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
        logits = self.output_proj(output)
        ctc_logits = self.ctc_proj(output).log_softmax(dim=-1)
        return logits, ctc_logits

class MegaG2P(nn.Module):
    def __init__(self, char_vocab_size, phon_vocab_size):
        super().__init__()
        self.encoder = CNNEncoder(char_vocab_size)
        self.decoder = TransformerDecoder(phon_vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_lens, tgt):
        memory = self.encoder(src, src_lens)
        dec_input = tgt[:, :-1]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(dec_input.size(1)).to(DEVICE)
        logits, ctc_logits = self.decoder(
            dec_input,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=(src == 0)
        )
        return logits, ctc_logits

def compute_ctc_loss(ctc_logits, targets, target_lens):
    ctc_logits = ctc_logits.permute(1, 0, 2)
    input_lengths = torch.full((ctc_logits.size(1),), ctc_logits.size(0), dtype=torch.long, device=DEVICE)
    return F.ctc_loss(
        ctc_logits,
        targets[:, 1:],
        input_lengths,
        target_lens - 1,
        blank=0,
        zero_infinity=True
    )

def train_model():
    torch.cuda.empty_cache()
    metadata_path = os.path.join(BASE_PATH, METADATA)
    dataset = MegaG2PDataset(metadata_path)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch, num_workers=0 if IS_KAGGLE else 4, pin_memory=True, persistent_workers=not IS_KAGGLE)
    model = MegaG2P(len(dataset.char2idx), len(dataset.phon2idx)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY, eps=1e-8)
    scheduler = OneCycleLR(optimizer, max_lr=MAX_LR, steps_per_epoch=len(loader), epochs=EPOCHS, pct_start=0.1)
    cross_entropy = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=LABEL_SMOOTHING)
    scaler = GradScaler('cuda')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_ce_loss = 0
        total_ctc_loss = 0
        pbar = tqdm(loader)
        for batch_idx, (src, src_lens, tgt, tgt_lens) in enumerate(pbar):
            src = src.to(DEVICE)
            src_lens = src_lens.to(DEVICE)
            tgt = tgt.to(DEVICE)
            tgt_lens = tgt_lens.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                logits, ctc_logits = model(src, src_lens, tgt)
                ce_loss = cross_entropy(logits.view(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
                ctc_loss = compute_ctc_loss(ctc_logits, tgt, tgt_lens)
                loss = ce_loss + 0.3 * ctc_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_ctc_loss += ctc_loss.item()
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ce': f"{ce_loss.item():.4f}",
                'ctc': f"{ctc_loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
        avg_loss = total_loss / len(loader)
        avg_ce = total_ce_loss / len(loader)
        avg_ctc = total_ctc_loss / len(loader)
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            torch.save({
                'epoch': epoch+1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': avg_loss,
                'config': {
                    'char2idx': dataset.char2idx,
                    'phon2idx': dataset.phon2idx,
                    'idx2phon': dataset.idx2phon,
                    'emb_dim': EMB_DIM,
                    'hidden_dim': HIDDEN_DIM,
                    'num_layers': NUM_LAYERS,
                    'n_heads': N_HEADS
                }
            }, f'checkpoint_epoch_{epoch+1}.pth')
    torch.save({
        'model_state': model.state_dict(),
        'config': {
            'char2idx': dataset.char2idx,
            'phon2idx': dataset.phon2idx,
            'idx2phon': dataset.idx2phon,
            'emb_dim': EMB_DIM,
            'hidden_dim': HIDDEN_DIM,
            'num_layers': NUM_LAYERS,
            'n_heads': N_HEADS
        }
    }, MODEL_SAVE_PATH)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    train_model()
    
    
# As u can see i have had totally failed in my first attempt the phenome-recognition is nothing but absoltue garbaage it
# has acquired quite a large dataset yet the ambiguity is not one bit clear and instead the model keeps on failing and 
# produces even poor results then version 1.00 so now i ll have refine it and clear all the grabagae and train it on more 
# defined and better-expected results based data set
# Results produced for first attempt:
# LEARNING             → 
# ARTIFICIAL           → Political Activities
# INTELLIGENCE         → Political Activities
# TESTING              → 
# APPLE                → Automatic Court.
# BANANA               → Another point
# COMPUTER             → Breakfast Biscuit.
# ALGORITHM            → Automatic data 