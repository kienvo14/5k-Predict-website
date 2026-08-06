"""
Predict an athlete's fastest_5k from a SEQUENCE of their recent runs.

This is the "fair fight" version. Instead of hand-crafting athlete-level
aggregates (like the sklearn baseline does), we give the model a sequence
of per-run summary vectors and let it learn its own aggregation.

Each run -> compact summary vector (~10 numbers).
Each athlete -> variable-length sequence of runs -> single 5K label.
Model: LSTM over the run sequence + attention pool -> MLP head -> scalar.

Split by athlete. Augmentation: per epoch, randomly sample MAX_RUNS from
each athlete's history so the model sees many "views" of the same fitness.

Baseline to beat: sklearn GBoost ~80s CV MAE, using hand-crafted athlete
features (longest_run, max_hr, easy_pace, easy_hr, aerobic, avg_weekly_
distance_m, active_weeks, consistency_std, gender).
"""

import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
PARQUET      = os.path.join(_HERE, "fitrec_runs.parquet")
FEATURES_CSV = os.path.join(_HERE, "..", "features.csv")
MAX_RUNS     = 30          # sample this many recent-ish runs per athlete each epoch
MIN_RUNS     = 3           # skip athletes with fewer runs than this
BATCH_SIZE   = 32
EPOCHS       = 40
PATIENCE     = 6
LR           = 1e-3
HIDDEN       = 128
DROPOUT      = 0.3
VAL_FRAC     = 0.20
SEED         = 42
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

np.random.seed(SEED); torch.manual_seed(SEED)
print(f"torch {torch.__version__}, device={DEVICE}")


# ---------------------- per-run summary features ----------------------
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat = p2 - p1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

FEAT_NAMES = [
    "duration_s",       # log
    "distance_km",      # log
    "avg_pace",         # min/km
    "pace_std",
    "avg_hr",
    "hr_std",
    "max_hr",
    "elev_gain_m",      # log(1+x)
    "elev_loss_m",      # log(1+x)
    "decouple",         # (avg_pace_last25%) / (avg_pace_first75%) - 1
]
N_FEAT = len(FEAT_NAMES)

def summarize(row):
    hr  = np.asarray(row["heart_rate"], dtype=np.float32)
    alt = np.asarray(row["altitude"],   dtype=np.float32)
    lat = np.asarray(row["latitude"],   dtype=np.float32)
    lon = np.asarray(row["longitude"],  dtype=np.float32)
    ts  = np.asarray(row["timestamp"],  dtype=np.float64)
    n = min(len(hr), len(alt), len(lat), len(lon), len(ts))
    if n < 30:
        return None
    hr, alt, lat, lon, ts = hr[:n], alt[:n], lat[:n], lon[:n], ts[:n]

    d = np.zeros(n, dtype=np.float32)
    d[1:] = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    dt = np.zeros(n, dtype=np.float32)
    dt[1:] = np.maximum(ts[1:] - ts[:-1], 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, d / dt, 0.0)
        pace = np.where(speed > 0.5, 1000.0 / (speed * 60.0), 15.0)
    pace = np.clip(pace, 2.5, 15.0)

    duration = float(ts[-1] - ts[0])
    distance = float(d.sum()) / 1000.0
    if duration < 60 or distance < 0.3:
        return None

    dalt = np.diff(alt)
    elev_gain = float(np.clip(dalt, 0, 20).sum())
    elev_loss = float(np.clip(-dalt, 0, 20).sum())

    cut = int(n * 0.75)
    if cut < 5 or n - cut < 5:
        decouple = 0.0
    else:
        p1 = pace[:cut].mean(); p2 = pace[cut:].mean()
        decouple = float(p2 / p1 - 1.0) if p1 > 0 else 0.0

    feat = np.array([
        np.log(duration),
        np.log(max(distance, 0.1)),
        float(pace.mean()),
        float(pace.std()),
        float(hr.mean()),
        float(hr.std()),
        float(hr.max()),
        np.log1p(elev_gain),
        np.log1p(elev_loss),
        decouple,
    ], dtype=np.float32)

    if not np.all(np.isfinite(feat)):
        return None
    return feat


# ---------------------- load + build per-athlete ----------------------
def build_athlete_data():
    print(f"loading {PARQUET}...")
    df = pq.read_table(PARQUET).to_pandas()
    print(f"  {len(df)} runs")

    feats = pd.read_csv(FEATURES_CSV).dropna(subset=["fastest_5k"])[["userId", "fastest_5k"]]
    feats["userId"] = pd.to_numeric(feats["userId"], errors="coerce").astype("Int64").dropna().astype("int64")
    df = df.merge(feats, on="userId", how="inner")
    print(f"  after label join: {len(df)} runs from {df['userId'].nunique()} labelled athletes")

    per_athlete = {}   # userId -> (M, N_FEAT) matrix of run summaries, sorted by ts
    labels = {}        # userId -> fastest_5k (seconds)

    for uid, group in tqdm(df.groupby("userId"), desc="summarizing"):
        summaries = []
        for _, row in group.iterrows():
            s = summarize(row)
            if s is not None:
                summaries.append(s)
        if len(summaries) < MIN_RUNS:
            continue
        per_athlete[int(uid)] = np.stack(summaries, axis=0).astype(np.float32)
        labels[int(uid)] = float(group["fastest_5k"].iloc[0])

    print(f"  usable athletes (>= {MIN_RUNS} valid runs): {len(per_athlete)}")
    return per_athlete, labels


# ------------------------------- dataset ------------------------------
class AthleteDataset(Dataset):
    def __init__(self, uids, per_athlete, labels, max_runs, feat_mean, feat_std, training=True):
        self.uids = list(uids)
        self.per_athlete = per_athlete
        self.labels = labels
        self.max_runs = max_runs
        self.mean = feat_mean
        self.std = feat_std
        self.training = training

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        uid = self.uids[i]
        runs = self.per_athlete[uid]  # (M, N_FEAT)
        M = runs.shape[0]
        if self.training and M > self.max_runs:
            idx = np.random.choice(M, self.max_runs, replace=False)
            idx.sort()
            picked = runs[idx]
        else:
            picked = runs[-self.max_runs:] if M > self.max_runs else runs
        picked = (picked - self.mean) / self.std
        # pad to max_runs with zeros, build mask
        pad_n = self.max_runs - picked.shape[0]
        if pad_n > 0:
            picked = np.concatenate([picked, np.zeros((pad_n, N_FEAT), dtype=np.float32)], axis=0)
        mask = np.zeros(self.max_runs, dtype=np.float32)
        mask[:self.max_runs - pad_n] = 1.0
        return (
            torch.from_numpy(picked),                     # (T, F)
            torch.from_numpy(mask),                       # (T,)
            torch.tensor(self.labels[uid], dtype=torch.float32),
        )


# ------------------------------- model --------------------------------
class HistoryModel(nn.Module):
    """Per-run MLP encoder -> masked mean-pool over runs -> MLP head.

    We deliberately DROP the LSTM: runs are given in random order (see
    AthleteDataset), so temporal encoding was wasted capacity. This is a
    'learned Ridge' — same input as the Ridge baseline, but with a
    nonlinear per-run encoder and a nonlinear head.
    """
    def __init__(self, in_dim=N_FEAT, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.per_run = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, mask):
        h = self.per_run(x)                                 # (B, T, H)
        w = mask.unsqueeze(-1)                              # (B, T, 1)
        pooled = (h * w).sum(dim=1) / w.sum(dim=1).clamp(min=1)  # masked mean
        return self.head(pooled).squeeze(-1)


# ------------------------------- main ---------------------------------
def main():
    per_athlete, labels = build_athlete_data()

    uids = np.array(sorted(per_athlete.keys()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(uids)
    n_val = int(len(uids) * VAL_FRAC)
    val_uids = uids[:n_val]
    train_uids = uids[n_val:]
    print(f"split by athlete: {len(train_uids)} train / {len(val_uids)} val")

    # normalize using train stats (across all runs of train athletes)
    train_stack = np.concatenate([per_athlete[u] for u in train_uids], axis=0)
    feat_mean = train_stack.mean(axis=0)
    feat_std  = train_stack.std(axis=0) + 1e-6
    print("feature means:", dict(zip(FEAT_NAMES, feat_mean.round(2))))
    print("feature stds :", dict(zip(FEAT_NAMES, feat_std.round(2))))

    train_ds = AthleteDataset(train_uids, per_athlete, labels, MAX_RUNS, feat_mean, feat_std, training=True)
    val_ds   = AthleteDataset(val_uids,   per_athlete, labels, MAX_RUNS, feat_mean, feat_std, training=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # baselines on val
    y_train = np.array([labels[u] for u in train_uids])
    y_val   = np.array([labels[u] for u in val_uids])
    naive_pred = np.full_like(y_val, y_train.mean())
    print(f"\nBASELINES (val athletes):")
    print(f"  predict train mean:  MAE = {np.abs(naive_pred - y_val).mean():.1f} s")
    # simple linear-on-per-athlete-means baseline
    train_means = np.stack([per_athlete[u].mean(axis=0) for u in train_uids])
    val_means   = np.stack([per_athlete[u].mean(axis=0) for u in val_uids])
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import GradientBoostingRegressor
    ridge = Ridge().fit(train_means, y_train)
    gb = GradientBoostingRegressor(random_state=SEED).fit(train_means, y_train)
    print(f"  Ridge on run-mean features:      MAE = {np.abs(ridge.predict(val_means) - y_val).mean():.1f} s")
    print(f"  GBoost on run-mean features:     MAE = {np.abs(gb.predict(val_means) - y_val).mean():.1f} s")

    # normalize the target too — MAE loss on raw seconds gave huge initial
    # gradients (labels 600-3600s, initial pred ~0), the model spent 12 epochs
    # just crawling toward the mean. In normalized units the mean is 0, so
    # the model finds it in epoch 1 and can focus on the actual signal.
    y_mean = float(y_train.mean())
    y_std  = float(y_train.std()) + 1e-6
    print(f"target: mean={y_mean:.1f}s  std={y_std:.1f}s")

    model = HistoryModel().to(DEVICE)
    print(model)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.L1Loss()

    best_val = float("inf"); best_state = None; best_epoch = 0; stale = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss, tr_n = 0.0, 0
        for x, m, y in train_dl:
            x, m, y = x.to(DEVICE), m.to(DEVICE), y.to(DEVICE)
            y_norm = (y - y_mean) / y_std
            pred = model(x, m)
            loss = loss_fn(pred, y_norm)
            opt.zero_grad(); loss.backward(); opt.step()
            # report in raw seconds for readability
            tr_loss += loss.item() * y_std * x.size(0); tr_n += x.size(0)
        tr_mae = tr_loss / tr_n

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, m, y in val_dl:
                p_norm = model(x.to(DEVICE), m.to(DEVICE)).cpu().numpy()
                preds.append(p_norm * y_std + y_mean)     # denorm to seconds
                trues.append(y.numpy())
        preds = np.concatenate(preds); trues = np.concatenate(trues)
        val_mae = np.abs(preds - trues).mean()

        marker = ""
        if val_mae < best_val:
            best_val = val_mae
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0; marker = "  *best"
        else:
            stale += 1
        print(f"epoch {epoch:2d}: train_MAE={tr_mae:.1f}s  val_MAE={val_mae:.1f}s{marker}")
        if stale >= PATIENCE:
            print(f"early stop @ epoch {epoch}, best={best_val:.1f}s @ epoch {best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs("out", exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "feat_mean": feat_mean.tolist(),
        "feat_std":  feat_std.tolist(),
        "config":    {"hidden": HIDDEN, "max_runs": MAX_RUNS, "n_feat": N_FEAT},
        "best_val_MAE_seconds": float(best_val),
    }, "out/history_lstm.pt")
    print(f"\nsaved: out/history_lstm.pt")
    print(f"\nRESULT: PyTorch history-LSTM val MAE = {best_val:.1f} s")
    print(f"        (baselines above; sklearn on hand-crafted features ~= 80 s)")


if __name__ == "__main__":
    main()
