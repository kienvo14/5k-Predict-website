"""
Predict an athlete's fastest_5k time from a SINGLE run's time-series.

Data flow:
    fitrec_runs.parquet   (~117k runs, per-run time-series)
    features.csv          (776 athletes, has fastest_5k label)
    -> join on userId
    -> split by athlete (never mix an athlete across train/val)
    -> resample every run to fixed length -> (pace, hr, elev_delta) per step
    -> 1D-CNN -> global mean pool -> MLP head -> scalar (seconds)

Baseline to beat: current sklearn GBoost on 9 aggregate athlete-level
features scores ~80s CV MAE.  Our model has to hit similar-or-better on
the RUN-level task (harder framing: one run in -> full 5K prediction out).
"""

import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ------------------------------- CONFIG -------------------------------
PARQUET       = "fitrec_runs.parquet"
FEATURES_CSV  = "../features.csv"
SEQ_LEN       = 200       # every run resampled to this many timesteps
BATCH_SIZE    = 128
EPOCHS        = 15
PATIENCE      = 3
LR            = 1e-3
HIDDEN        = 128
DROPOUT       = 0.2
VAL_FRAC      = 0.20      # fraction of ATHLETES (not runs) held out
MIN_LEN       = 150       # skip any run shorter than this after loading
SEED          = 42
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

np.random.seed(SEED); torch.manual_seed(SEED)
print(f"torch {torch.__version__}, device={DEVICE}")


# --------------------------- DATA LOADING ----------------------------
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat = p2 - p1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def run_to_features(row, seq_len):
    """One run -> (seq_len, 3) array of [pace_min_per_km, hr_bpm, delta_elev_m]."""
    hr  = np.asarray(row["heart_rate"], dtype=np.float32)
    alt = np.asarray(row["altitude"],   dtype=np.float32)
    lat = np.asarray(row["latitude"],   dtype=np.float32)
    lon = np.asarray(row["longitude"],  dtype=np.float32)
    ts  = np.asarray(row["timestamp"],  dtype=np.float64)
    n = min(len(hr), len(alt), len(lat), len(lon), len(ts))
    if n < MIN_LEN:
        return None
    hr, alt, lat, lon, ts = hr[:n], alt[:n], lat[:n], lon[:n], ts[:n]

    # per-step pace (min/km) from GPS + timestamps
    d = np.zeros(n, dtype=np.float32)
    d[1:] = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    dt = np.ones(n, dtype=np.float32)
    dt[1:] = np.maximum(ts[1:] - ts[:-1], 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, d / dt, 0.0)
        pace = np.where(speed > 0.5, 1000.0 / (speed * 60.0), 15.0)
    pace = np.clip(pace, 2.5, 15.0).astype(np.float32)

    # elevation delta (per-step change, not raw altitude — removes location bias)
    delev = np.zeros(n, dtype=np.float32)
    delev[1:] = np.clip(alt[1:] - alt[:-1], -20.0, 20.0)

    # resample all three signals onto a fixed grid of seq_len points
    src_x = np.linspace(0, 1, n, dtype=np.float32)
    tgt_x = np.linspace(0, 1, seq_len, dtype=np.float32)
    out = np.stack([
        np.interp(tgt_x, src_x, pace),
        np.interp(tgt_x, src_x, hr).astype(np.float32),
        np.interp(tgt_x, src_x, delev),
    ], axis=1)  # (seq_len, 3)

    if not np.all(np.isfinite(out)):
        return None
    return out


def load_all():
    print(f"loading {PARQUET}...")
    table = pq.read_table(PARQUET)
    df = table.to_pandas()
    print(f"  {len(df)} runs, {df['userId'].nunique()} unique users in parquet")

    print(f"loading {FEATURES_CSV}...")
    feats = pd.read_csv(FEATURES_CSV)
    feats = feats.dropna(subset=["fastest_5k"])[["userId", "fastest_5k"]]
    feats["userId"] = pd.to_numeric(feats["userId"], errors="coerce").astype("Int64").dropna().astype("int64")
    print(f"  {len(feats)} athletes with fastest_5k labels")

    # attach label to every run
    df = df.merge(feats, on="userId", how="inner")
    print(f"  after label join: {len(df)} runs from {df['userId'].nunique()} labelled users")
    return df


def build_arrays(df, seq_len):
    """Convert every run row -> fixed-shape feature tensor. Returns X, y, uids."""
    xs, ys, uids = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="featurizing"):
        feat = run_to_features(row, seq_len)
        if feat is None:
            continue
        xs.append(feat)
        ys.append(float(row["fastest_5k"]))
        uids.append(int(row["userId"]))
    X = np.stack(xs, axis=0)                          # (N, seq_len, 3)
    y = np.asarray(ys, dtype=np.float32)              # (N,)
    uids = np.asarray(uids, dtype=np.int64)           # (N,)
    return X, y, uids


class RunDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        # channels-first for 1D-CNN: (C, T)
        return self.X[i].transpose(0, 1), self.y[i]


# ------------------------------ MODEL --------------------------------
class Run1DCNN(nn.Module):
    """Small 1D-CNN encoder + global mean pool + MLP -> scalar seconds."""
    def __init__(self, in_ch=3, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(in_ch,   hidden//2, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(hidden//2, hidden, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden,   hidden, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden,   hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        # x: (B, C, T)
        h = self.enc(x)                # (B, hidden, T')
        h = h.mean(dim=-1)             # global mean pool -> (B, hidden)
        return self.head(h).squeeze(-1)  # (B,)


# ------------------------------ MAIN ---------------------------------
def main():
    df = load_all()

    # athlete-level split BEFORE we do any heavy work
    users = df["userId"].unique()
    rng = np.random.default_rng(SEED)
    rng.shuffle(users)
    n_val = int(len(users) * VAL_FRAC)
    val_users = set(users[:n_val])
    train_users = set(users[n_val:])
    print(f"split by athlete: {len(train_users)} train / {len(val_users)} val")

    df_train = df[df["userId"].isin(train_users)].reset_index(drop=True)
    df_val   = df[df["userId"].isin(val_users)].reset_index(drop=True)
    print(f"runs: {len(df_train)} train / {len(df_val)} val")

    print("\n-- featurizing train --")
    Xtr, ytr, _ = build_arrays(df_train, SEQ_LEN)
    print(f"  Xtr={Xtr.shape}  ytr={ytr.shape}")

    print("\n-- featurizing val --")
    Xva, yva, uva = build_arrays(df_val, SEQ_LEN)
    print(f"  Xva={Xva.shape}  yva={yva.shape}")

    # normalize features using TRAIN stats only (leakage rule)
    mean = Xtr.reshape(-1, 3).mean(axis=0)
    std  = Xtr.reshape(-1, 3).std(axis=0) + 1e-6
    print(f"feature mean (pace, hr, delev): {mean.round(2)}")
    print(f"feature std  (pace, hr, delev): {std.round(2)}")
    Xtr = ((Xtr - mean) / std).astype(np.float32)
    Xva = ((Xva - mean) / std).astype(np.float32)

    # baselines
    y_mean_pred = np.full_like(yva, ytr.mean())
    print(f"\nBASELINES on val runs:")
    print(f"  predict-train-mean  MAE: {np.abs(y_mean_pred - yva).mean():.1f} s")

    # per-athlete median from TRAIN — but for val athletes we don't have their runs
    # in train, so global mean is the honest baseline. (An "average their features"
    # baseline would require re-implementing the sklearn model — skip for now.)

    train_dl = DataLoader(RunDataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(RunDataset(Xva, yva), batch_size=BATCH_SIZE, shuffle=False)

    model = Run1DCNN().to(DEVICE)
    print(model)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.L1Loss()  # MAE — matches the reporting metric

    best_val = float("inf"); best_state = None; best_epoch = 0; stale = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss, tr_n = 0.0, 0
        for x, y in tqdm(train_dl, desc=f"epoch {epoch}", mininterval=1.0):
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * x.size(0); tr_n += x.size(0)
        tr_mae = tr_loss / tr_n

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in val_dl:
                preds.append(model(x.to(DEVICE)).cpu().numpy())
                trues.append(y.numpy())
        preds = np.concatenate(preds); trues = np.concatenate(trues)
        run_mae = np.abs(preds - trues).mean()

        # ALSO compute per-athlete MAE — average the model's per-run predictions
        # for each val athlete, then compare to their (single) fastest_5k label.
        val_df = pd.DataFrame({"userId": uva, "pred": preds, "true": trues})
        ath = val_df.groupby("userId").agg(pred=("pred", "mean"), true=("true", "first"))
        ath_mae = (ath["pred"] - ath["true"]).abs().mean()

        marker = ""
        if ath_mae < best_val:
            best_val = ath_mae
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
            marker = "  *best"
        else:
            stale += 1
        print(f"epoch {epoch}: train_MAE={tr_mae:.1f}s  val_run_MAE={run_mae:.1f}s  val_athlete_MAE={ath_mae:.1f}s{marker}")
        if stale >= PATIENCE:
            print(f"early stop @ epoch {epoch}, best athlete_MAE={best_val:.1f}s @ epoch {best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs("out", exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "mean": mean.tolist(),
        "std":  std.tolist(),
        "config": {"seq_len": SEQ_LEN, "hidden": HIDDEN},
        "best_val_athlete_MAE_seconds": float(best_val),
    }, "out/run_to_5k.pt")
    print(f"\nsaved: out/run_to_5k.pt")
    print(f"BEST VAL athlete MAE: {best_val:.1f} s   (baseline to beat: ~80 s)")


if __name__ == "__main__":
    main()
