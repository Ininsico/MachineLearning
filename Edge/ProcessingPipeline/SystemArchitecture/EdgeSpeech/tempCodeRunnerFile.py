# train_g2p_chain.py
import os, sys, time, random
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import nltk
from g2p_en import G2p
import math
import numpy as np
from tqdm import tqdm  # Added tqdm import

BASE_PATH = r"C:\Users\arsla\Downloads\archive (4)\LJSpeech-1.1"
METADATA = "metadata.csv"
SAMPLE_LIMIT = None
BATCH_SIZE = 64
EPOCHS_SEQ = 3
EPOCHS_TRANS = 6
EMB_DIM = 256
HID = 512
NUM_LAYERS = 3
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_SEQ = "g2p_student.pth"
SAVE_TRANS = "g2p_refiner.pth"
SEED = 42

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
try:
    nltk.data.find('corpora/cmudict')
except LookupError:
    nltk.download('cmudict', quiet=True)

meta_path = Path(BASE_PATH) / METADATA
if not meta_path.exists():
    print("metadata.csv not found at", meta_path, file=sys.stderr)
    sys.exit(1)

# Add progress bar for data loading
print("Loading and preprocessing data...")
with tqdm(total=4, desc="Data Preparation") as pbar:
    df = pd.read_csv(meta_path, sep="|", header=None, names=["id","text","norm"])
    if SAMPLE_LIMIT:
        df = df.sample(SAMPLE_LIMIT, random_state=SEED).reset_index(drop=True)
    else:
        df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    pbar.update(1)
    
    g2p = G2p()
    def text_to_phoneme_tokens(t):
        toks = [tok for tok in g2p(str(t)) if tok != " "]
        return toks

    # Add progress bar for phoneme tokenization
    tqdm.pandas(desc="Phoneme Tokenization")
    df["phon_tokens"] = df["norm"].astype(str).progress_apply(text_to_phoneme_tokens)
    pbar.update(1)
    
    # Add progress bar for character processing
    tqdm.pandas(desc="Character Processing")
    df["chars"] = df["norm"].astype(str).progress_apply(lambda s: list(s.lower()))
    pbar.update(1)
    
    all_chars = sorted({c for seq in df["chars"] for c in seq})
    char2idx = {c:i+3 for i,c in enumerate(all_chars)}
    char2idx["<pad>"]=0; char2idx["<sos>"]=1; char2idx["<eos>"]=2
    idx2char = {i:c for c,i in char2idx.items()}

    all_phons = sorted({p for seq in df["phon_tokens"] for p in seq})
    phon2idx = {p:i+3 for i,p in enumerate(all_phons)}
    phon2idx["<pad>"]=0; phon2idx["<sos>"]=1; phon2idx["<eos>"]=2
    idx2phon = {i:p for p,i in phon2idx.items()}
    pbar.update(1)

def encode_chars(chars):
    return [char2idx.get(c, char2idx["<pad>"]) for c in chars]

def encode_phons(phs):
    return [phon2idx.get(p, phon2idx["<pad>"]) for p in phs]

class G2PDataset(Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        chars = encode_chars(self.df.loc[idx, "chars"])
        phs = encode_phons(self.df.loc[idx, "phon_tokens"])
        phs = phs + [phon2idx["<eos>"]]
        return torch.tensor(chars, dtype=torch.long), torch.tensor(phs, dtype=torch.long)

def collate_fn(batch):
    inputs, targets = zip(*batch)
    in_lens = [len(x) for x in inputs]
    tgt_lens = [len(x) for x in targets]
    inputs_padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=char2idx["<pad>"])
    targets_padded = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=phon2idx["<pad>"])
    return inputs_padded, torch.tensor(in_lens), targets_padded, torch.tensor(tgt_lens)

dataset = G2PDataset(df)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)

class Encoder(nn.Module):
    def __init__(self, vocab_in, emb_dim, hid, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_in, emb_dim, padding_idx=char2idx["<pad>"])
        self.lstm = nn.LSTM(emb_dim, hid, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.bridge_h = nn.ModuleList([nn.Linear(hid*2, hid) for _ in range(num_layers)])
        self.bridge_c = nn.ModuleList([nn.Linear(hid*2, hid) for _ in range(num_layers)])
        self.num_layers = num_layers
    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        h = []
        c = []
        for layer in range(self.num_layers):
            hf = h_n[2*layer]; hb = h_n[2*layer+1]
            cf = c_n[2*layer]; cb = c_n[2*layer+1]
            h_cat = torch.cat([hf, hb], dim=1)
            c_cat = torch.cat([cf, cb], dim=1)
            h.append(torch.tanh(self.bridge_h[layer](h_cat)))
            c.append(torch.tanh(self.bridge_c[layer](c_cat)))
        h = torch.stack(h, dim=0)
        c = torch.stack(c, dim=0)
        return out, (h, c)

class Decoder(nn.Module):
    def __init__(self, vocab_out, emb_dim, hid, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_out, emb_dim, padding_idx=phon2idx["<pad>"])
        self.lstm = nn.LSTM(emb_dim, hid, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hid, vocab_out)
    def forward(self, tgt_inputs, hidden):
        emb = self.embedding(tgt_inputs)
        out, hidden = self.lstm(emb, hidden)
        logits = self.fc_out(out)
        return logits, hidden

class Seq2Seq(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec
    def forward(self, src, src_lens, tgt_inp):
        enc_out, enc_hidden = self.enc(src, src_lens)
        logits, _ = self.dec(tgt_inp, enc_hidden)
        return logits

enc = Encoder(vocab_in=len(char2idx), emb_dim=EMB_DIM, hid=HID, num_layers=NUM_LAYERS)
dec = Decoder(vocab_out=len(phon2idx), emb_dim=EMB_DIM, hid=HID, num_layers=NUM_LAYERS)
model_seq = Seq2Seq(enc, dec).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=phon2idx["<pad>"])
optimizer = torch.optim.Adam(model_seq.parameters(), lr=LEARNING_RATE)

def train_epoch_seq(epoch):
    model_seq.train()
    total_loss = 0.0
    t0 = time.time()
    
    # Add progress bar for batch processing
    batch_iter = tqdm(loader, desc=f"Seq2Seq Epoch {epoch}", leave=False)
    for i, (src, src_lens, tgt, tgt_lens) in enumerate(batch_iter):
        src, src_lens, tgt = src.to(DEVICE), src_lens.to(DEVICE), tgt.to(DEVICE)
        sos = torch.full((tgt.size(0), 1), phon2idx["<sos>"], dtype=torch.long, device=DEVICE)
        dec_inp = torch.cat([sos, tgt[:, :-1]], dim=1)
        logits = model_seq(src, src_lens, dec_inp)
        loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_seq.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        batch_iter.set_postfix(loss=loss.item())
    
    avg_loss = total_loss/len(loader)
    print(f"Seq Epoch {epoch} avg loss {avg_loss:.4f} time {time.time()-t0:.1f}s")
    return avg_loss

def greedy_decode_seq(model, src_seq):
    model.eval()
    with torch.no_grad():
        src = torch.tensor([encode_chars(list(src_seq.lower()))], dtype=torch.long, device=DEVICE)
        src_lens = torch.tensor([src.size(1)], dtype=torch.long, device=DEVICE)
        enc_out, enc_hidden = model.enc(src, src_lens)
        cur = torch.tensor([[phon2idx["<sos>"]]], dtype=torch.long, device=DEVICE)
        hidden = enc_hidden
        outputs = []
        for _ in range(120):
            logits, hidden = model.dec(cur, hidden)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            top = probs.argmax(dim=-1).item()
            if top == phon2idx["<eos>"] or top == phon2idx["<pad>"]:
                break
            outputs.append(top)
            cur = torch.tensor([[top]], dtype=torch.long, device=DEVICE)
    return outputs

def ids_to_phons(id_list):
    return [idx2phon[i] for i in id_list]

print("\nDevice:", DEVICE, "Starting Seq2Seq training")
seq_losses = []
for ep in range(1, EPOCHS_SEQ+1):
    loss = train_epoch_seq(ep)
    seq_losses.append(loss)
    sample_idx = random.randint(0, len(df)-1)
    sample_txt = df.loc[sample_idx, "norm"]
    teacher = df.loc[sample_idx, "phon_tokens"]
    pred_ids = greedy_decode_seq(model_seq, sample_txt)
    print("SAMPLE:", sample_txt)
    print("TEACHER:", " ".join(teacher[:40]))
    print("PRED  :", " ".join(ids_to_phons(pred_ids)[:40]))

torch.save({"model_state": model_seq.state_dict(), "char2idx": char2idx, "phon2idx": phon2idx, "idx2phon": idx2phon}, SAVE_SEQ)
print("Saved seq model to", SAVE_SEQ)

print("\nGenerating student predictions for the whole dataset (for refiner training)")
student_preds = []
targets = []

# Add progress bar for generating student predictions
with tqdm(total=len(df), desc="Generating Predictions") as pbar:
    for i in range(len(df)):
        txt = df.loc[i, "norm"]
        pred_ids = greedy_decode_seq(model_seq, txt)
        if len(pred_ids)==0:
            pred_ids = [phon2idx["<eos>"]]
        student_preds.append([idx2phon[x] for x in pred_ids])
        targets.append(df.loc[i, "phon_tokens"])
        pbar.update(1)

df["student_pred"] = student_preds
df["target_phons"] = targets

# Update phoneme vocabulary with any new predictions
all_pred_phons = sorted({p for seq in df["student_pred"] for p in seq})
for p in tqdm(all_pred_phons, desc="Updating Phoneme Vocabulary"):
    if p not in phon2idx:
        phon2idx[p] = len(phon2idx)
        idx2phon[phon2idx[p]] = p


class RefinerDataset(Dataset):
    def __init__(self, preds, targets):
        self.preds = preds
        self.targets = targets
    def __len__(self): return len(self.preds)
    def __getitem__(self, idx):
        src = [phon2idx.get(p, phon2idx["<pad>"]) for p in self.preds[idx]] + [phon2idx["<eos>"]]
        tgt = [phon2idx.get(p, phon2idx["<pad>"]) for p in self.targets[idx]] + [phon2idx["<eos>"]]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

def collate_refiner(batch):
    srcs, tgts = zip(*batch)
    src_lens = [len(x) for x in srcs]
    tgt_lens = [len(x) for x in tgts]
    src_pad = nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=phon2idx["<pad>"])
    tgt_pad = nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=phon2idx["<pad>"])
    return src_pad, torch.tensor(src_lens), tgt_pad, torch.tensor(tgt_lens)

ref_dataset = RefinerDataset(df["student_pred"].tolist(), df["target_phons"].tolist())
ref_loader = DataLoader(ref_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_refiner, num_workers=0)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerRefiner(nn.Module):
    def __init__(self, vocab_size, emb_dim, nhead=8, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.src_tok_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=phon2idx["<pad>"])
        self.tgt_tok_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=phon2idx["<pad>"])
        self.pos_enc = PositionalEncoding(emb_dim, dropout)
        self.transformer = nn.Transformer(d_model=emb_dim, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(emb_dim, vocab_size)
        self.vocab_size = vocab_size
    def forward(self, src, tgt):
        src_mask = (src == phon2idx["<pad>"]).to(DEVICE)
        tgt_mask = (tgt == phon2idx["<pad>"]).to(DEVICE)
        src_emb = self.pos_enc(self.src_tok_emb(src))
        tgt_emb = self.pos_enc(self.tgt_tok_emb(tgt))
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_mask)
        out = self.transformer.decoder(tgt_emb, memory, tgt_key_padding_mask=tgt_mask, memory_key_padding_mask=src_mask)
        logits = self.fc_out(out)
        return logits
vocab_size = len(phon2idx)
refiner = TransformerRefiner(vocab_size=vocab_size, emb_dim=EMB_DIM, nhead=8, 
                           num_encoder_layers=3, num_decoder_layers=3, 
                           dim_feedforward=2048).to(DEVICE)
opt_ref = torch.optim.Adam(refiner.parameters(), lr=1e-4)
crit_ref = nn.CrossEntropyLoss(ignore_index=phon2idx["<pad>"])

def per_from_seqs(pred_seq, tgt_seq):
    if len(tgt_seq)==0:
        return 0.0 if len(pred_seq)==0 else 1.0
    a = pred_seq
    b = tgt_seq
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1): dp[i][0]=i
    for j in range(len(b)+1): dp[0][j]=j
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            cost = 0 if a[i-1]==b[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[len(a)][len(b)]/max(1, len(b))
def train_refiner():
    refiner.train()
    ref_losses = []
    
    for epoch in range(1, EPOCHS_TRANS+1):
        total_loss = 0.0
        total_per = 0.0
        steps = 0
        t0 = time.time()
        
        # Add progress bar for refiner training
        batch_iter = tqdm(ref_loader, desc=f"Refiner Epoch {epoch}", leave=False)
        for i, (src, src_lens, tgt, tgt_lens) in enumerate(batch_iter):
            src, src_lens, tgt = src.to(DEVICE), src_lens.to(DEVICE), tgt.to(DEVICE)
            sos = torch.full((tgt.size(0), 1), phon2idx["<sos>"], dtype=torch.long, device=DEVICE)
            dec_inp = torch.cat([sos, tgt[:, :-1]], dim=1)
            logits = refiner(src, dec_inp)
            loss = crit_ref(logits.view(-1, logits.size(-1)), tgt.view(-1))
            opt_ref.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
            opt_ref.step()
            total_loss += loss.item()
            
            with torch.no_grad():
                preds = logits.argmax(dim=-1).cpu().numpy()
                tgts = tgt.cpu().numpy()
                batch_per = 0.0
                for b in range(preds.shape[0]):
                    pred_seq = [idx2phon[int(x)] for x in preds[b] if x!=phon2idx["<pad>"] and x!=phon2idx["<sos>"]]
                    tgt_seq = [idx2phon[int(x)] for x in tgts[b] if x!=phon2idx["<pad>"] and x!=phon2idx["<sos>"]]
                    batch_per += per_from_seqs(pred_seq, tgt_seq)
                total_per += batch_per
            steps += preds.shape[0]
            batch_iter.set_postfix(loss=loss.item())
        
        avg_loss = total_loss/len(ref_loader)
        avg_per = total_per/steps if steps>0 else 0.0
        ref_losses.append(avg_loss)
        print(f"Refiner Epoch {epoch} loss {avg_loss:.4f} avg PER {avg_per:.4f} time {time.time()-t0:.1f}s")
        torch.save({"model_state": refiner.state_dict(), "phon2idx": phon2idx, "idx2phon": idx2phon}, SAVE_TRANS)
    
    print("Saved refiner to", SAVE_TRANS)
    return ref_losses

def infer_refiner(src_phon_list, max_len=120):
    refiner.eval()
    src_ids = [phon2idx.get(p, phon2idx["<pad>"]) for p in src_phon_list] + [phon2idx["<eos>"]]
    src = torch.tensor([src_ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        tgt_cur = torch.tensor([[phon2idx["<sos>"]]], dtype=torch.long, device=DEVICE)
        outputs = []
        for _ in range(max_len):
            logits = refiner(src, tgt_cur)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            top = probs.argmax(dim=-1).item()
            if top==phon2idx["<eos>"] or top==phon2idx["<pad>"]:
                break
            outputs.append(idx2phon[top])
            tgt_cur = torch.cat([tgt_cur, torch.tensor([[top]], dtype=torch.long, device=DEVICE)], dim=1)
    return outputs

print("\nStarting refiner training")
ref_losses = train_refiner()

print("\nFinal test samples:")
test_samples = random.sample(range(len(df)), min(5, len(df)))
for i in tqdm(test_samples, desc="Testing Samples"):
    txt = df.loc[i, "norm"]
    teacher = df.loc[i, "phon_tokens"]
    seq_pred_ids = greedy_decode_seq(model_seq, txt)
    seq_pred = [idx2phon[x] for x in seq_pred_ids]
    refined = infer_refiner(seq_pred)
    print("\nTEXT:", txt)
    print("TEACHER:", " ".join(teacher[:80]))
    print("STUDENT:", " ".join(seq_pred[:80]))
    print("REFINED:", " ".join(refined[:80]))
    per_student = per_from_seqs(seq_pred, teacher)
    per_refined = per_from_seqs(refined, teacher)
    print(f"PER student {per_student:.4f} | PER refined {per_refined:.4f}")

print("\nTraining complete. Models saved to:")
print(f"- Seq2Seq model: {SAVE_SEQ}")
print(f"- Refiner model: {SAVE_TRANS}")