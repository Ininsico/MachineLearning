# ====== numpy+librosa AE (fine-tune capable, sharper recon) ======
import os, glob, time, math
import numpy as np
import librosa, librosa.display
import soundfile as sf
import matplotlib.pyplot as plt

# ---------------------------
# CONFIG
# ---------------------------
AUDIO_PATH = "your_audio.wav"       # single file (used if DATA_DIR is empty)
DATA_DIR   = "wavs"                 # optional folder of .wav files
OUTPUT_MODEL = "audio_autoencoder_model.npz"
RECON_WAV    = "reconstructed.wav"

SR      = 16000
N_MELS  = 256            # ↑ resolution helps
N_FFT   = 1024
HOP     = 256
PATCH_T = 128            # ↑ temporal context helps
H1, H2, LATENT = 1024, 512, 256   # deeper & wider
LR_INIT = 1e-3
LR_MIN  = 1e-5
BATCH   = 128
EPOCHS  = 300
WD      = 1e-5           # weight decay
EARLY_STOP = 20
REDUCE_PLATEAU = 6       # epochs w/o improvement -> half LR

# Loss weights
W_MSE = 1.0
W_LOGCOSH = 0.2
W_EDGE = 0.5             # time+freq gradient matching

SEED = 1337
np.random.seed(SEED)

# ---------------------------
# DATA
# ---------------------------
def load_wavs():
    files = []
    if os.path.isdir(DATA_DIR):
        files = sorted(glob.glob(os.path.join(DATA_DIR, "*.wav")))
    if not files and os.path.isfile(AUDIO_PATH):
        files = [AUDIO_PATH]
    if not files:
        raise FileNotFoundError("Provide AUDIO_PATH or put .wav files in DATA_DIR.")
    ys = []
    for f in files:
        y, _ = librosa.load(f, sr=SR, mono=True)
        ys.append(y)
    return ys

def to_mel(y):
    M = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0
    )
    M_db = librosa.power_to_db(M, ref=np.max)
    return M_db  # shape (n_mels, T)

def normalize_global(mel_list):
    lo = min(m.min() for m in mel_list)
    hi = max(m.max() for m in mel_list)
    out = [((m - lo) / (hi - lo + 1e-12)).T for m in mel_list]  # -> (T, F)
    return out, lo, hi

def make_random_patch_batch(X_list, patch_t, batch_size, augment=True):
    """Sample random time windows across multiple mel sequences."""
    F = X_list[0].shape[1]
    B = batch_size
    out = np.zeros((B, patch_t * F), dtype=np.float64)
    for b in range(B):
        Xi = X_list[np.random.randint(len(X_list))]
        T = Xi.shape[0]
        if T <= patch_t:  # pad if clip too short
            pad = np.tile(Xi[-1:], (patch_t - T + 1, 1))
            Xi_ext = np.vstack([Xi, pad])
            T = Xi_ext.shape[0]
            Xi = Xi_ext
        start = np.random.randint(0, T - patch_t + 1)
        patch = Xi[start:start + patch_t, :].copy()  # (patch_t, F)

        if augment:
            # SpecAugment-style (time/freq masking + small noise)
            if np.random.rand() < 0.7:
                w = np.random.randint(1, max(2, patch_t // 8))
                t0 = np.random.randint(0, patch_t - w + 1)
                patch[t0:t0 + w, :] *= 0.0  # time mask
            if np.random.rand() < 0.7:
                w = np.random.randint(1, max(2, F // 12))
                f0 = np.random.randint(0, F - w + 1)
                patch[:, f0:f0 + w] *= 0.0  # freq mask
            if np.random.rand() < 0.9:
                patch += np.random.normal(0, 0.02, size=patch.shape)
            patch = np.clip(patch, 0.0, 1.0)

        out[b] = patch.reshape(-1)
    return out

ys = load_wavs()
mels = [to_mel(y) for y in ys]
mel_norm_list, MEL_MIN, MEL_MAX = normalize_global(mels)  # list of (T,F)
FREQ_BINS = mel_norm_list[0].shape[1]
IN_DIM = PATCH_T * FREQ_BINS

# train/val split by clips (80/20)
perm = np.random.permutation(len(mel_norm_list))
split = max(1, int(0.8 * len(mel_norm_list)))
train_list = [mel_norm_list[i] for i in perm[:split]]
val_list   = [mel_norm_list[i] for i in perm[split:]] or mel_norm_list[:1]

# ---------------------------
# MODEL (Deep AE)
# ---------------------------
class AE:
    def __init__(self, in_dim, h1, h2, latent, params=None):
        if params is None:
            self.p = {
                "W1": np.random.randn(in_dim, h1) * np.sqrt(2.0 / in_dim),
                "b1": np.zeros((1, h1)),
                "W2": np.random.randn(h1, h2) * np.sqrt(2.0 / h1),
                "b2": np.zeros((1, h2)),
                "W3": np.random.randn(h2, latent) * np.sqrt(2.0 / h2),
                "b3": np.zeros((1, latent)),
                "W4": np.random.randn(latent, h2) * np.sqrt(2.0 / latent),
                "b4": np.zeros((1, h2)),
                "W5": np.random.randn(h2, h1) * np.sqrt(2.0 / h2),
                "b5": np.zeros((1, h1)),
                "W6": np.random.randn(h1, in_dim) * np.sqrt(2.0 / h1),
                "b6": np.zeros((1, in_dim)),
            }
        else:
            self.p = {k: v.copy() for k, v in params.items()}
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}  # Adam m
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}  # Adam v
        self.t = 0

    @staticmethod
    def relu(z): return np.maximum(0.0, z)
    @staticmethod
    def drelu(z): return (z > 0.0).astype(np.float64)
    @staticmethod
    def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

    def forward(self, X):
        c = {"A0": X}
        c["Z1"] = X @ self.p["W1"] + self.p["b1"]; c["A1"] = self.relu(c["Z1"])
        c["Z2"] = c["A1"] @ self.p["W2"] + self.p["b2"]; c["A2"] = self.relu(c["Z2"])
        c["Z3"] = c["A2"] @ self.p["W3"] + self.p["b3"]; c["A3"] = self.relu(c["Z3"])  # latent
        c["Z4"] = c["A3"] @ self.p["W4"] + self.p["b4"]; c["A4"] = self.relu(c["Z4"])
        c["Z5"] = c["A4"] @ self.p["W5"] + self.p["b5"]; c["A5"] = self.relu(c["Z5"])
        c["Z6"] = c["A5"] @ self.p["W6"] + self.p["b6"]; c["A6"] = self.sigmoid(c["Z6"])  # [0,1]
        return c

    # ---- Loss helpers ----
    @staticmethod
    def log_cosh(x):
        return np.log(np.cosh(np.clip(x, -10, 10)))

    @staticmethod
    def diff_time(Xbf):      # (B, T, F) -> (B, T-1, F)
        return Xbf[:, 1:, :] - Xbf[:, :-1, :]

    @staticmethod
    def diff_time_back(U):   # (B, T-1, F) -> (B, T, F)
        B, Tm1, F = U.shape
        G = np.zeros((B, Tm1 + 1, F), dtype=np.float64)
        G[:, 0, :]  -= U[:, 0, :]
        G[:, 1:-1, :] += U[:, :-1, :] - U[:, 1:, :]
        G[:, -1, :] += U[:, -1, :]
        return G

    @staticmethod
    def diff_freq(Xbf):      # (B, T, F) -> (B, T, F-1)
        return Xbf[:, :, 1:] - Xbf[:, :, :-1]

    @staticmethod
    def diff_freq_back(U):   # (B, T, F-1) -> (B, T, F)
        B, T, Fm1 = U.shape
        G = np.zeros((B, T, Fm1 + 1), dtype=np.float64)
        G[:, :, 0]  -= U[:, :, 0]
        G[:, :, 1:-1] += U[:, :, :-1] - U[:, :, 1:]
        G[:, :, -1] += U[:, :, -1]
        return G

    def loss_and_grad(self, cache, patch_t, F, wd=0.0):
        """
        Hybrid loss:
          MSE + log-cosh + edge loss (time+freq gradients)
        Returns (loss_scalar, grads_dict)
        """
        X0 = cache["A0"]; X6 = cache["A6"]
        B = X0.shape[0]
        # base terms
        diff = X6 - X0
        mse = np.mean(diff**2)
        logc = np.mean(self.log_cosh(diff))

        # edge losses (operate on reshaped [B, T, F])
        X0bf = X0.reshape(B, patch_t, F)
        X6bf = X6.reshape(B, patch_t, F)

        dt0 = self.diff_time(X0bf); dt6 = self.diff_time(X6bf)
        df0 = self.diff_freq(X0bf); df6 = self.diff_freq(X6bf)

        edge_t = np.mean((dt6 - dt0)**2)
        edge_f = np.mean((df6 - df0)**2)
        edge = edge_t + edge_f

        loss = W_MSE*mse + W_LOGCOSH*logc + W_EDGE*edge

        # L2 reg
        l2 = 0.0
        for k, v in self.p.items():
            if k[0] == "W":
                l2 += np.sum(v*v)
        loss += wd * l2

        # ---- gradient w.r.t A6 (before sigmoid backprop) ----
        dA6 = (2.0/B) * W_MSE * diff + (W_LOGCOSH/B) * np.tanh(np.clip(diff, -10, 10))

        # edge grads back to A6
        U_t = (2.0/B) * W_EDGE * (dt6 - dt0)          # (B, T-1, F)
        U_f = (2.0/B) * W_EDGE * (df6 - df0)          # (B, T, F-1)
        g_t = self.diff_time_back(U_t)                 # (B, T, F)
        g_f = self.diff_freq_back(U_f)                 # (B, T, F)
        dA6 += (g_t + g_f).reshape(B, -1)

        # backprop through sigmoid
        dZ6 = dA6 * (cache["A6"] * (1.0 - cache["A6"]))

        g = {}
        g["dW6"] = cache["A5"].T @ dZ6 + wd * 2.0 * self.p["W6"]
        g["db6"] = np.sum(dZ6, axis=0, keepdims=True)

        dA5 = dZ6 @ self.p["W6"].T
        dZ5 = dA5 * self.drelu(cache["Z5"])
        g["dW5"] = cache["A4"].T @ dZ5 + wd * 2.0 * self.p["W5"]
        g["db5"] = np.sum(dZ5, axis=0, keepdims=True)

        dA4 = dZ5 @ self.p["W5"].T
        dZ4 = dA4 * self.drelu(cache["Z4"])
        g["dW4"] = cache["A3"].T @ dZ4 + wd * 2.0 * self.p["W4"]
        g["db4"] = np.sum(dZ4, axis=0, keepdims=True)

        dA3 = dZ4 @ self.p["W4"].T
        dZ3 = dA3 * self.drelu(cache["Z3"])
        g["dW3"] = cache["A2"].T @ dZ3 + wd * 2.0 * self.p["W3"]
        g["db3"] = np.sum(dZ3, axis=0, keepdims=True)

        dA2 = dZ3 @ self.p["W3"].T
        dZ2 = dA2 * self.drelu(cache["Z2"])
        g["dW2"] = cache["A1"].T @ dZ2 + wd * 2.0 * self.p["W2"]
        g["db2"] = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.p["W2"].T
        dZ1 = dA1 * self.drelu(cache["Z1"])
        g["dW1"] = cache["A0"].T @ dZ1 + wd * 2.0 * self.p["W1"]
        g["db1"] = np.sum(dZ1, axis=0, keepdims=True)

        return loss, g

    def adam(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for k in self.p:
            dk = "d" + k
            self.m[k] = b1 * self.m[k] + (1 - b1) * grads[dk]
            self.v[k] = b2 * self.v[k] + (1 - b2) * (grads[dk]**2)
            mhat = self.m[k] / (1 - b1**self.t)
            vhat = self.v[k] / (1 - b2**self.t)
            self.p[k] -= lr * mhat / (np.sqrt(vhat) + eps)

# ---------------------------
# LOAD OR INIT MODEL
# ---------------------------
loaded = None
if os.path.isfile(OUTPUT_MODEL):
    data = np.load(OUTPUT_MODEL, allow_pickle=True)
    meta = data.get("meta")
    if meta is not None:
        meta = meta.tolist()
        (in_dim0, h10, h20, lat0, n_mels0, patch_t0, sr0, nfft0, hop0,
         mel_min0, mel_max0) = meta
        arch_match = (in_dim0 == IN_DIM and n_mels0 == N_MELS and patch_t0 == PATCH_T)
        if arch_match:
            print("Loading existing model for fine-tune:", OUTPUT_MODEL)
            loaded = {k: data[k] for k in data.files if k != "meta"}
        else:
            print("Found model, but arch mismatch. Starting fresh.")
    else:
        print("Found model without meta. Starting fresh.")

model = AE(IN_DIM, H1, H2, LATENT, params=loaded)

# ---------------------------
# TRAIN
# ---------------------------
def val_loss_epoch():
    # fixed val set (no aug)
    V = min(512, 4 * BATCH)
    xb = make_random_patch_batch(val_list, PATCH_T, V, augment=False)
    c = model.forward(xb)
    loss, _ = model.loss_and_grad(c, PATCH_T, FREQ_BINS, wd=0.0)
    return loss

lr = LR_INIT if loaded is None else max(LR_INIT * 0.2, 1e-4)  # smaller LR when fine-tuning
best, best_params = np.inf, {k: v.copy() for k, v in model.p.items()}
no_imp, since_reduce = 0, 0

for ep in range(1, EPOCHS + 1):
    # train one epoch
    steps = max(200, 1000 * len(train_list) // BATCH)  # more steps if only 1 file
    tr_running = 0.0
    for _ in range(steps):
        xb = make_random_patch_batch(train_list, PATCH_T, BATCH, augment=True)
        c = model.forward(xb)
        loss, grads = model.loss_and_grad(c, PATCH_T, FREQ_BINS, wd=WD)
        model.adam(grads, lr)
        tr_running += loss
    tr_loss = tr_running / steps

    va_loss = val_loss_epoch()
    print(f"Epoch {ep:03d} | lr {lr:.1e} | train {tr_loss:.6f} | val {va_loss:.6f}")

    improved = va_loss + 1e-6 < best
    if improved:
        best, no_imp, since_reduce = va_loss, 0, 0
        best_params = {k: v.copy() for k, v in model.p.items()}
    else:
        no_imp += 1; since_reduce += 1
        if since_reduce >= REDUCE_PLATEAU and lr > LR_MIN:
            lr = max(LR_MIN, lr * 0.5); since_reduce = 0
            print(f"  ↳ Reduce LR -> {lr:.1e}")
        if no_imp >= EARLY_STOP:
            print("  ↳ Early stopping.")
            break

# restore best
model.p = best_params

# save model
meta = np.array([IN_DIM, H1, H2, LATENT, N_MELS, PATCH_T, SR, N_FFT, HOP, MEL_MIN, MEL_MAX], dtype=object)
np.savez(OUTPUT_MODEL, **model.p, meta=meta)
print("Saved:", OUTPUT_MODEL)

# ---------------------------
# RECONSTRUCT FULL MEL
# ---------------------------
def reconstruct_full(model, X_full, patch_t, F):
    T = X_full.shape[0]
    n = T - patch_t + 1
    out = np.zeros((T, F)); cnt = np.zeros((T, F))
    for i in range(n):
        patch = X_full[i:i+patch_t, :].reshape(1, patch_t * F)
        rec = model.forward(patch)["A6"].reshape(patch_t, F)
        out[i:i+patch_t] += rec
        cnt[i:i+patch_t] += 1
    out /= np.maximum(cnt, 1e-9)
    return out

# pick the longest clip for demo recon
X_demo = max(mel_norm_list, key=lambda a: a.shape[0])
M_rec_norm = reconstruct_full(model, X_demo, PATCH_T, FREQ_BINS)
M_rec_db = M_rec_norm * (MEL_MAX - MEL_MIN) + MEL_MIN
M_rec_power = librosa.db_to_power(M_rec_db, ref=1.0)

# invert mel -> audio
y_rec = librosa.feature.inverse.mel_to_audio(
    M_rec_power.T, sr=SR, n_fft=N_FFT, hop_length=HOP, power=2.0, n_iter=96
)
sf.write(RECON_WAV, y_rec, SR)
print("Reconstructed audio:", RECON_WAV)

# ---------------------------
# PLOTS
# ---------------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1); plt.title("Original (mel dB)")
librosa.display.specshow(to_mel(ys[0]), sr=SR, hop_length=HOP, x_axis="time", y_axis="mel")
plt.subplot(1,2,2); plt.title("Reconstructed (mel dB)")
librosa.display.specshow(M_rec_db, sr=SR, hop_length=HOP, x_axis="time", y_axis="mel")
plt.tight_layout(); plt.show()
