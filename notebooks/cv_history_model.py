"""
5-fold cross-validation of the mean-pool history model, apples-to-apples
against the sklearn baselines on the SAME athlete folds.

Reuses everything from train_5k_from_history.py; only the training/eval
loop is wrapped in a KFold split.

Reports:
    mean ± std of val_MAE across 5 folds for
        - naive (predict train mean)
        - Ridge on per-athlete run-means
        - GBoost on per-athlete run-means
        - PyTorch mean-pool MLP
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor

from train_5k_from_history import (
    build_athlete_data, AthleteDataset, HistoryModel,
    MAX_RUNS, BATCH_SIZE, EPOCHS, PATIENCE, LR, SEED, DEVICE, N_FEAT,
)

N_FOLDS = 5

def train_fold(train_uids, val_uids, per_athlete, labels, fold_idx):
    train_stack = np.concatenate([per_athlete[u] for u in train_uids], axis=0)
    feat_mean = train_stack.mean(axis=0)
    feat_std  = train_stack.std(axis=0) + 1e-6

    train_ds = AthleteDataset(train_uids, per_athlete, labels, MAX_RUNS, feat_mean, feat_std, training=True)
    val_ds   = AthleteDataset(val_uids,   per_athlete, labels, MAX_RUNS, feat_mean, feat_std, training=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    y_train = np.array([labels[u] for u in train_uids])
    y_val   = np.array([labels[u] for u in val_uids])
    y_mean, y_std = float(y_train.mean()), float(y_train.std()) + 1e-6

    # baselines on this same fold
    train_means = np.stack([per_athlete[u].mean(axis=0) for u in train_uids])
    val_means   = np.stack([per_athlete[u].mean(axis=0) for u in val_uids])
    naive_mae  = float(np.abs(y_val - y_train.mean()).mean())
    ridge_mae  = float(np.abs(Ridge().fit(train_means, y_train).predict(val_means) - y_val).mean())
    gb_mae     = float(np.abs(GradientBoostingRegressor(random_state=SEED).fit(train_means, y_train).predict(val_means) - y_val).mean())

    # PyTorch model
    torch.manual_seed(SEED + fold_idx)
    model = HistoryModel().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.L1Loss()

    best_val = float("inf"); best_state = None; stale = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, m, y in train_dl:
            x, m, y = x.to(DEVICE), m.to(DEVICE), y.to(DEVICE)
            loss = loss_fn(model(x, m), (y - y_mean) / y_std)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, m, y in val_dl:
                p = model(x.to(DEVICE), m.to(DEVICE)).cpu().numpy() * y_std + y_mean
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

    return {"naive": naive_mae, "ridge": ridge_mae, "gboost": gb_mae, "pytorch": best_val}


def main():
    per_athlete, labels = build_athlete_data()
    uids = np.array(sorted(per_athlete.keys()))
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    results = []
    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(uids), 1):
        train_uids = uids[tr_idx]; val_uids = uids[va_idx]
        print(f"\n=== fold {fold_idx}/{N_FOLDS}  ({len(train_uids)} train / {len(val_uids)} val) ===")
        r = train_fold(train_uids, val_uids, per_athlete, labels, fold_idx)
        print(f"  naive={r['naive']:.1f}s  ridge={r['ridge']:.1f}s  gboost={r['gboost']:.1f}s  pytorch={r['pytorch']:.1f}s")
        results.append(r)

    print("\n" + "=" * 60)
    print(f"5-fold CV results ({N_FOLDS} folds, split by athlete):")
    print("=" * 60)
    for name in ["naive", "ridge", "gboost", "pytorch"]:
        vals = np.array([r[name] for r in results])
        print(f"  {name:8s}  MAE = {vals.mean():6.2f} s   (± {vals.std():.2f})   per-fold: {vals.round(1).tolist()}")


if __name__ == "__main__":
    main()
