"""
HR Imputation with a Bidirectional LSTM
=======================================
Trains a model that fills missing heart-rate segments in a run using the
surrounding pace/elevation context.

Data:   FitRec / Endomondo raw file (endomondoHR.json, ~6.6GB)
        Cite: Ni, Muhlstein, McAuley — "Modeling heart rate and activity data
        for personalized fitness recommendation." WWW 2019.

This file is written as a Colab-compatible script with # %% cell markers.
You can:
  - Open it as-is in Colab (File > Upload notebook, or paste cells)
  - Run it as a script: `py hr_imputation.py`
  - Convert to .ipynb via jupytext if you prefer

Doesn't touch the app: separate folder, separate deps, saves its own model.
"""

# %% [markdown]
# ## 1. Setup

# %%
import gzip
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__}, device={device}")


# %% [markdown]
# ## 2. Config — start SMALL, expand once it works
#
# 5,000 runs is plenty to prove the pipeline. Bump to 30k+ for a real training run.

# %%
FITREC_PATH   = "fitrec_diverse.json.gz"  # compact preprocessed file (run preprocess_fitrec.py first)
MAX_RUNS      = 15000                   # cap for THIS training run; the preprocessed file is already diverse
MAX_PER_USER  = 6                       # not enforced here anymore — done by preprocess_fitrec.py
MIN_LEN       = 200                     # skip runs shorter than this many samples
SEQ_LEN       = 300                     # crop / pad each run to this length
GAP_LEN       = 30                      # simulate a ~5min HR gap (samples in the mask)
BATCH_SIZE    = 64
EPOCHS        = 6
HIDDEN        = 128
LR            = 1e-3
VAL_FRACTION  = 0.15
SEED          = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


# %% [markdown]
# ## 3. Load FitRec — stream the file line by line
#
# Each line is a Python dict literal (NOT JSON). We keep only runs with usable
# HR/altitude/GPS and at least MIN_LEN samples.

# %%
def load_runs(path, max_runs, min_len):
    """Load from the compact preprocessed file — already diverse across users."""
    runs = []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            try:
                d = eval(line)
            except Exception:
                continue
            if d.get("sport") != "run":
                continue
            hr, alt = d.get("heart_rate"), d.get("altitude")
            lat, lon, ts = d.get("latitude"), d.get("longitude"), d.get("timestamp")
            if not (hr and alt and lat and lon and ts):
                continue
            n = min(len(hr), len(alt), len(lat), len(lon), len(ts))
            if n < min_len:
                continue
            runs.append({
                "userId": d.get("userId"),
                "hr": hr[:n], "alt": alt[:n],
                "lat": lat[:n], "lon": lon[:n], "ts": ts[:n],
            })
            if len(runs) >= max_runs:
                break
    return runs

print(f"Loading up to {MAX_RUNS} runs from {FITREC_PATH}...")
runs = load_runs(FITREC_PATH, MAX_RUNS, MIN_LEN)
print(f"Loaded {len(runs)} runs, {len(set(r['userId'] for r in runs))} unique users")


# %% [markdown]
# ## 4. Feature engineering — pace from GPS via haversine
#
# The raw file has no pace field. We derive it: distance-between-consecutive-points
# divided by the time-between-them, at each sample.

# %%
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat = p2 - p1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def run_to_features(r):
    """Return (T, 3) array: [pace_min_per_km, altitude_m, hr_bpm], length T."""
    lat = np.asarray(r["lat"], dtype=np.float32)
    lon = np.asarray(r["lon"], dtype=np.float32)
    alt = np.asarray(r["alt"], dtype=np.float32)
    hr = np.asarray(r["hr"], dtype=np.float32)
    ts = np.asarray(r["ts"], dtype=np.float64)

    # per-step distance (m) and dt (s); prepend 0/1 to keep length
    d = np.zeros_like(lat)
    d[1:] = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    dt = np.ones_like(ts, dtype=np.float32)
    dt[1:] = np.maximum(ts[1:] - ts[:-1], 1.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        speed_mps = np.where(dt > 0, d / dt, 0.0)
        pace = np.where(speed_mps > 0.5, 1000.0 / (speed_mps * 60.0), 15.0)  # min/km
    pace = np.clip(pace, 2.5, 15.0)

    return np.stack([pace, alt, hr], axis=1)    # (T, 3)


feature_runs = []
for r in tqdm(runs, desc="feature engineering"):
    x = run_to_features(r)
    if len(x) >= MIN_LEN and np.all(np.isfinite(x)):
        feature_runs.append((r["userId"], x))
print(f"After feature step: {len(feature_runs)} runs")


# %% [markdown]
# ## 5. Split by USER (never mix a user's runs across train/val — leakage rule)

# %%
users = list({u for u, _ in feature_runs})
np.random.shuffle(users)
n_val = int(len(users) * VAL_FRACTION)
val_users = set(users[:n_val])
train_runs = [(u, x) for u, x in feature_runs if u not in val_users]
val_runs   = [(u, x) for u, x in feature_runs if u in val_users]
print(f"train: {len(train_runs)} runs / {len(users) - n_val} users")
print(f"val:   {len(val_runs)} runs / {n_val} users")


# %% [markdown]
# ## 6. Normalize using TRAIN stats only (leakage rule again)

# %%
train_stack = np.concatenate([x for _, x in train_runs], axis=0)
mean = train_stack.mean(axis=0)
std  = train_stack.std(axis=0) + 1e-6
print(f"feature means (pace, alt, hr): {mean.round(2)}")
print(f"feature stds  (pace, alt, hr): {std.round(2)}")

def normalize(x):
    return ((x - mean) / std).astype(np.float32)


# %% [markdown]
# ## 7. Dataset — random-mask a GAP_LEN chunk each epoch (never see the same mask twice)
#
# Model INPUT: [pace_norm, alt_norm, hr_norm, mask_flag]  where hr_norm=0 in the gap
# Model TARGET: real hr_norm across the entire sequence

# %%
class HrGapDataset(Dataset):
    def __init__(self, runs, seq_len, gap_len):
        self.runs = runs
        self.seq_len = seq_len
        self.gap_len = gap_len

    def __len__(self):
        return len(self.runs)

    def __getitem__(self, i):
        _, x = self.runs[i]
        T = len(x)
        start = np.random.randint(0, max(1, T - self.seq_len + 1))
        seg = x[start:start + self.seq_len]
        if len(seg) < self.seq_len:
            pad = np.tile(seg[-1:], (self.seq_len - len(seg), 1))
            seg = np.concatenate([seg, pad], axis=0)
        seg = normalize(seg)

        # PREDICT-THE-FUTURE framing: always mask the LAST gap_len timesteps.
        # Model sees pace/alt for the whole sequence + HR for [0, gap_start], predicts HR for [gap_start, end].
        gap_start = self.seq_len - self.gap_len
        gap_end = self.seq_len

        mask = np.zeros(self.seq_len, dtype=np.float32)  # 1 = inside the gap
        mask[gap_start:gap_end] = 1.0

        hr = seg[:, 2].copy()
        hr_target = hr.copy()
        hr[gap_start:gap_end] = 0.0          # HR "missing" in the gap

        # DERIVATIVES of the observed signals (NOT hr — hr is masked, so its
        # derivative would spike at the gap edges and confuse the model).
        dpace = np.gradient(seg[:, 0])
        dalt  = np.gradient(seg[:, 1])

        # input features: pace, alt, masked-hr, mask-flag, dpace, dalt  (6 dims)
        x_in = np.stack([seg[:, 0], seg[:, 1], hr, mask, dpace, dalt], axis=1)
        return torch.from_numpy(x_in), torch.from_numpy(hr_target), torch.from_numpy(mask)


train_ds = HrGapDataset(train_runs, SEQ_LEN, GAP_LEN)
val_ds   = HrGapDataset(val_runs,   SEQ_LEN, GAP_LEN)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# %% [markdown]
# ## 8. Bidirectional LSTM

# %%
class BiLstmHr(nn.Module):
    def __init__(self, in_dim=6, hidden=HIDDEN, layers=2, dropout=0.2):
        super().__init__()
        # Fix B: unidirectional for forecast-the-future framing. The backward
        # pass in a biLSTM would read through the zeroed HR gap and add noise.
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True,
                            bidirectional=False, dropout=dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h).squeeze(-1)


model = BiLstmHr().to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)
print(model)


# %% [markdown]
# ## 9. Train — loss only on the GAP timesteps (that's what we're predicting)

# %%
def masked_mse(pred, target, mask):
    return ((pred - target) ** 2 * mask).sum() / mask.sum().clamp(min=1)

# Fix A: reconstruct HR on the WHOLE sequence, not just the gap. Gap timesteps
# get higher weight so we still optimize for the actual forecast task, but the
# 270 observed steps contribute gradient too — forcing the LSTM to actually
# learn the pace/alt -> HR mapping instead of collapsing to "extrapolate last HR".
GAP_WEIGHT = 5.0
def weighted_mse(pred, target, mask):
    w = mask * GAP_WEIGHT + (1.0 - mask)
    return ((pred - target) ** 2 * w).sum() / w.sum()

hr_std = std[2]  # for converting normalized-RMSE back to bpm

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for x, y, m in tqdm(train_dl, desc=f"epoch {epoch}"):
        x, y, m = x.to(device), y.to(device), m.to(device)
        pred = model(x)
        loss = weighted_mse(pred, y, m)
        opt.zero_grad(); loss.backward(); opt.step()
        train_loss += loss.item() * x.size(0)
    train_loss /= len(train_ds)

    model.eval()
    val_sse, val_n = 0.0, 0
    with torch.no_grad():
        for x, y, m in val_dl:
            x, y, m = x.to(device), y.to(device), m.to(device)
            pred = model(x)
            val_sse += (((pred - y) ** 2) * m).sum().item()
            val_n   += m.sum().item()
    val_rmse_norm = (val_sse / max(val_n, 1)) ** 0.5
    val_rmse_bpm  = val_rmse_norm * hr_std
    print(f"epoch {epoch}: train_loss={train_loss:.4f}  val_RMSE={val_rmse_bpm:.2f} bpm")


# %% [markdown]
# ## 10. Baselines — for PREDICT-THE-FUTURE framing (no right-endpoint to interp to)
#
# Two honest baselines:
#   (a) "last value held constant" — dumbest possible predictor
#   (b) "linear extrapolation from last 20 known points" — a smarter naive
# LSTM must beat both to earn its keep.

# %%
def last_value_pred(hr_seq, gap_start, gap_end):
    return np.full(gap_end - gap_start, hr_seq[gap_start - 1])

def linear_extrap_pred(hr_seq, gap_start, gap_end, k=20):
    k = min(k, gap_start)
    xs = np.arange(gap_start - k, gap_start)
    ys = hr_seq[gap_start - k:gap_start]
    slope, intercept = np.polyfit(xs, ys, 1)
    future_x = np.arange(gap_start, gap_end)
    return slope * future_x + intercept

lv_sse, ex_sse, model_sse, n_all = 0.0, 0.0, 0.0, 0
model.eval()
with torch.no_grad():
    for x, y, m in val_dl:
        pred = model(x.to(device)).cpu().numpy()
        y_np, m_np, x_np = y.numpy(), m.numpy(), x.numpy()
        for i in range(len(y_np)):
            hr_real = y_np[i]
            gap_idx = np.where(m_np[i] > 0)[0]
            if len(gap_idx) == 0: continue
            gs, ge = gap_idx[0], gap_idx[-1] + 1
            hr_seen = x_np[i, :, 2].copy()
            lv     = last_value_pred(hr_seen, gs, ge)
            ex     = linear_extrap_pred(hr_seen, gs, ge)
            truth  = hr_real[gs:ge]
            lv_sse    += ((lv    - truth) ** 2).sum()
            ex_sse    += ((ex    - truth) ** 2).sum()
            model_sse += ((pred[i, gs:ge] - truth) ** 2).sum()
            n_all += len(truth)

lv_bpm    = ((lv_sse    / n_all) ** 0.5) * hr_std
ex_bpm    = ((ex_sse    / n_all) ** 0.5) * hr_std
model_bpm = ((model_sse / n_all) ** 0.5) * hr_std
print(f"\n{'RESULTS (predict next 5 min from past)':=^50}")
print(f"last-value baseline:    {lv_bpm:.2f} bpm RMSE")
print(f"linear-extrap baseline: {ex_bpm:.2f} bpm RMSE")
print(f"BiLSTM model:           {model_bpm:.2f} bpm RMSE")
print(f"model vs best baseline: {min(lv_bpm, ex_bpm) - model_bpm:+.2f} bpm")


# %% [markdown]
# ## 11. Save the trained model + normalization stats

# %%
os.makedirs("out", exist_ok=True)
torch.save({
    "model_state": model.state_dict(),
    "mean": mean.tolist(),
    "std": std.tolist(),
    "config": {
        "in_dim": 4, "hidden": HIDDEN, "seq_len": SEQ_LEN, "gap_len": GAP_LEN,
    },
}, "out/hr_lstm.pt")
print("saved: out/hr_lstm.pt")


# %% [markdown]
# ## 12. Plot an example — a real gap being filled

# %%
model.eval()
x, y, m = next(iter(val_dl))
with torch.no_grad():
    pred = model(x.to(device)).cpu().numpy()

i = 0
gap = np.where(m[i].numpy() > 0)[0]
gs, ge = gap[0], gap[-1] + 1
hr_real = y[i].numpy() * std[2] + mean[2]     # denormalize -> bpm
hr_pred = pred[i] * std[2] + mean[2]
hr_seen = x[i, :, 2].numpy() * std[2] + mean[2]
hr_seen[gs:ge] = np.nan                       # visualize the gap

plt.figure(figsize=(11, 4))
plt.plot(hr_real, label="true HR (measured)", alpha=0.85)
plt.plot(hr_seen, label="observed (gap = missing)", alpha=0.7)
plt.plot(range(gs, ge), hr_pred[gs:ge], label="LSTM prediction (in gap)", linewidth=2)
plt.axvspan(gs, ge, alpha=0.1, color="orange")
plt.xlabel("timestep"); plt.ylabel("HR (bpm)")
plt.title("HR gap imputation — sample run")
plt.legend(); plt.tight_layout()
plt.savefig("out/example_fill.png", dpi=110)
plt.close()
print("saved: out/example_fill.png")
