"""
Combined PyTorch model: run-sequence branch + hand-crafted athlete-features
branch, fused into one prediction head.

Goal: beat the deployed sklearn baseline (82.1s CV MAE on 9 hand-crafted
features) by giving the neural model the SAME hand-crafted features PLUS
the run-sequence info it already has.

Baselines in the same folds:
    - sklearn Ridge/GBoost on hand-crafted 9 features   (matches trainModel.py)
    - PyTorch mean-pool MLP on run-summaries only       (the 88s result)
    - PyTorch COMBINED (run-seq + hand-crafted)         (the new attempt)
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor

from train_5k_from_history import (
    build_athlete_data,
    MAX_RUNS, BATCH_SIZE, EPOCHS, PATIENCE, LR, SEED, DEVICE,
    N_FEAT, HIDDEN, DROPOUT,
)

import os as _os
FEATURES_CSV = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "features.csv")
N_FOLDS = 5

HAND_FEATS = ["longest_run", "max_hr", "easy_pace", "easy_hr", "aerobic",
              "avg_weekly_distance_m", "active_weeks", "consistency_std", "gender"]


def load_handcrafted():
    df = pd.read_csv(FEATURES_CSV)
    df["gender"] = df["gender"].map({"male": 0, "female": 1})
    df["userId"] = pd.to_numeric(df["userId"], errors="coerce").astype("Int64").dropna().astype("int64")
    df = df.dropna(subset=HAND_FEATS + ["fastest_5k"])
    hand = df.set_index("userId")[HAND_FEATS].astype(np.float32)
    return hand   # DataFrame indexed by userId


class CombinedDataset(Dataset):
    def __init__(self, uids, per_athlete, labels, hand, max_runs,
                 feat_mean, feat_std, hand_mean, hand_std, training=True):
        self.uids = list(uids)
        self.per_athlete = per_athlete
        self.labels = labels
        self.hand = hand
        self.max_runs = max_runs
        self.mean = feat_mean; self.std = feat_std
        self.hand_mean = hand_mean; self.hand_std = hand_std
        self.training = training

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        uid = self.uids[i]
        runs = self.per_athlete[uid]
        M = runs.shape[0]
        if self.training and M > self.max_runs:
            idx = np.sort(np.random.choice(M, self.max_runs, replace=False))
            picked = runs[idx]
        else:
            picked = runs[-self.max_runs:] if M > self.max_runs else runs
        picked = (picked - self.mean) / self.std
        pad_n = self.max_runs - picked.shape[0]
        if pad_n > 0:
            picked = np.concatenate([picked, np.zeros((pad_n, N_FEAT), dtype=np.float32)], axis=0)
        mask = np.zeros(self.max_runs, dtype=np.float32); mask[:self.max_runs - pad_n] = 1.0

        hand_vec = self.hand.loc[uid].values.astype(np.float32)
        hand_vec = (hand_vec - self.hand_mean) / self.hand_std

        return (
            torch.from_numpy(picked),
            torch.from_numpy(mask),
            torch.from_numpy(hand_vec),
            torch.tensor(self.labels[uid], dtype=torch.float32),
        )


class CombinedModel(nn.Module):
    def __init__(self, in_dim=N_FEAT, hand_dim=len(HAND_FEATS), hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.per_run = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.hand_enc = nn.Sequential(
            nn.Linear(hand_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_seq, mask, x_hand):
        h = self.per_run(x_seq)
        w = mask.unsqueeze(-1)
        pooled = (h * w).sum(dim=1) / w.sum(dim=1).clamp(min=1)
        h_hand = self.hand_enc(x_hand)
        return self.head(torch.cat([pooled, h_hand], dim=-1)).squeeze(-1)


def train_fold(train_uids, val_uids, per_athlete, labels, hand, fold_idx):
    train_stack = np.concatenate([per_athlete[u] for u in train_uids], axis=0)
    feat_mean = train_stack.mean(axis=0); feat_std = train_stack.std(axis=0) + 1e-6

    Xhand_tr = hand.loc[train_uids].values.astype(np.float32)
    Xhand_va = hand.loc[val_uids].values.astype(np.float32)
    hand_mean = Xhand_tr.mean(axis=0); hand_std = Xhand_tr.std(axis=0) + 1e-6

    y_train = np.array([labels[u] for u in train_uids])
    y_val   = np.array([labels[u] for u in val_uids])
    y_mean, y_std = float(y_train.mean()), float(y_train.std()) + 1e-6

    # sklearn baselines on hand-crafted features (matches trainModel.py setup)
    ridge_hand  = float(np.abs(Ridge().fit(Xhand_tr, y_train).predict(Xhand_va) - y_val).mean())
    gb_hand     = float(np.abs(GradientBoostingRegressor(random_state=SEED)
                          .fit(Xhand_tr, y_train).predict(Xhand_va) - y_val).mean())

    train_ds = CombinedDataset(train_uids, per_athlete, labels, hand, MAX_RUNS,
                                feat_mean, feat_std, hand_mean, hand_std, training=True)
    val_ds   = CombinedDataset(val_uids,   per_athlete, labels, hand, MAX_RUNS,
                                feat_mean, feat_std, hand_mean, hand_std, training=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    torch.manual_seed(SEED + fold_idx)
    model = CombinedModel().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.L1Loss()

    best_val = float("inf"); best_state = None; stale = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, m, h, y in train_dl:
            x, m, h, y = x.to(DEVICE), m.to(DEVICE), h.to(DEVICE), y.to(DEVICE)
            loss = loss_fn(model(x, m, h), (y - y_mean) / y_std)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, m, h, y in val_dl:
                p = model(x.to(DEVICE), m.to(DEVICE), h.to(DEVICE)).cpu().numpy() * y_std + y_mean
                preds.append(p); trues.append(y.numpy())
        val_mae = float(np.abs(np.concatenate(preds) - np.concatenate(trues)).mean())
        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break

    return {"ridge_hand": ridge_hand, "gboost_hand": gb_hand, "pytorch_combined": best_val}


def main():
    per_athlete, labels = build_athlete_data()
    hand = load_handcrafted()
    # keep only athletes present in BOTH the parquet-derived data AND the hand-crafted features
    common_uids = np.array(sorted(set(per_athlete.keys()) & set(hand.index)))
    print(f"\ncommon athletes (in both parquet & features.csv): {len(common_uids)}")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    results = []
    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(common_uids), 1):
        train_uids = common_uids[tr_idx]; val_uids = common_uids[va_idx]
        print(f"\n=== fold {fold_idx}/{N_FOLDS}  ({len(train_uids)} train / {len(val_uids)} val) ===")
        r = train_fold(train_uids, val_uids, per_athlete, labels, hand, fold_idx)
        print(f"  ridge_hand={r['ridge_hand']:.1f}s  gboost_hand={r['gboost_hand']:.1f}s  pytorch_combined={r['pytorch_combined']:.1f}s")
        results.append(r)

    print("\n" + "=" * 70)
    print(f"5-fold CV results (split by athlete, n={len(common_uids)}):")
    print("=" * 70)
    for name in ["ridge_hand", "gboost_hand", "pytorch_combined"]:
        vals = np.array([r[name] for r in results])
        print(f"  {name:20s}  MAE = {vals.mean():6.2f} s   (± {vals.std():.2f})   per-fold: {vals.round(1).tolist()}")


if __name__ == "__main__":
    main()
