"""
ONE SCRIPT to reproduce the full comparison. Runs 5-fold CV, split by
athlete, on the exact same folds for every model. Prints a clean summary.

Models compared:
  1. Naive               -> predict the training-set mean
  2. Ridge (hand)        -> sklearn Ridge on 9 hand-crafted athlete features
  3. GBoost (hand)       -> sklearn Gradient Boosting on same 9 features
                            (this is the DEPLOYED baseline from trainModel.py)
  4. PyTorch (runs only) -> mean-pool MLP on 10 per-run summary features
  5. PyTorch (combined)  -> run-sequence branch + hand-crafted branch fused

Run:
    py compare_all.py

Expected: PyTorch (combined) wins by ~14s over deployed GBoost.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

from train_5k_from_history import (
    build_athlete_data, AthleteDataset, HistoryModel,
    MAX_RUNS, BATCH_SIZE, EPOCHS, PATIENCE, LR, SEED, DEVICE, N_FEAT,
)
from cv_history_plus_handcrafted import (
    load_handcrafted, CombinedDataset, CombinedModel, HAND_FEATS,
)

N_FOLDS = 5


def train_pytorch_runs_only(train_uids, val_uids, per_athlete, labels, fold_idx):
    train_stack = np.concatenate([per_athlete[u] for u in train_uids], axis=0)
    feat_mean = train_stack.mean(axis=0); feat_std = train_stack.std(axis=0) + 1e-6
    y_train = np.array([labels[u] for u in train_uids])
    y_mean, y_std = float(y_train.mean()), float(y_train.std()) + 1e-6

    train_ds = AthleteDataset(train_uids, per_athlete, labels, MAX_RUNS, feat_mean, feat_std, training=True)
    val_ds   = AthleteDataset(val_uids,   per_athlete, labels, MAX_RUNS, feat_mean, feat_std, training=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    torch.manual_seed(SEED + fold_idx)
    model = HistoryModel().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.L1Loss()
    best = float("inf"); stale = 0
    for epoch in range(EPOCHS):
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
        if val_mae < best:
            best = val_mae; stale = 0
        else:
            stale += 1
            if stale >= PATIENCE: break
    return best


def train_pytorch_combined(train_uids, val_uids, per_athlete, labels, hand, fold_idx):
    train_stack = np.concatenate([per_athlete[u] for u in train_uids], axis=0)
    feat_mean = train_stack.mean(axis=0); feat_std = train_stack.std(axis=0) + 1e-6
    Xhand_tr = hand.loc[train_uids].values.astype(np.float32)
    hand_mean = Xhand_tr.mean(axis=0); hand_std = Xhand_tr.std(axis=0) + 1e-6
    y_train = np.array([labels[u] for u in train_uids])
    y_mean, y_std = float(y_train.mean()), float(y_train.std()) + 1e-6

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
    best = float("inf"); stale = 0
    for epoch in range(EPOCHS):
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
        if val_mae < best:
            best = val_mae; stale = 0
        else:
            stale += 1
            if stale >= PATIENCE: break
    return best


def main():
    per_athlete, labels = build_athlete_data()
    hand = load_handcrafted()
    common_uids = np.array(sorted(set(per_athlete.keys()) & set(hand.index)))
    print(f"\nAthletes used (in both parquet & features.csv): {len(common_uids)}")
    print(f"Total runs across those athletes: {sum(per_athlete[u].shape[0] for u in common_uids)}")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    per_fold = []
    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(common_uids), 1):
        train_uids = common_uids[tr_idx]; val_uids = common_uids[va_idx]
        y_tr = np.array([labels[u] for u in train_uids])
        y_va = np.array([labels[u] for u in val_uids])
        Xhand_tr = hand.loc[train_uids].values.astype(np.float32)
        Xhand_va = hand.loc[val_uids].values.astype(np.float32)

        naive = float(np.abs(y_va - y_tr.mean()).mean())
        linear = float(np.abs(LinearRegression().fit(Xhand_tr, y_tr).predict(Xhand_va) - y_va).mean())
        ridge = float(np.abs(Ridge().fit(Xhand_tr, y_tr).predict(Xhand_va) - y_va).mean())
        gboost = float(np.abs(GradientBoostingRegressor(random_state=SEED)
                              .fit(Xhand_tr, y_tr).predict(Xhand_va) - y_va).mean())
        pt_runs = train_pytorch_runs_only(train_uids, val_uids, per_athlete, labels, fold_idx)
        pt_comb = train_pytorch_combined(train_uids, val_uids, per_athlete, labels, hand, fold_idx)

        row = {"fold": fold_idx, "naive": naive, "linear_hand": linear, "ridge_hand": ridge,
               "gboost_hand": gboost, "pytorch_runs": pt_runs, "pytorch_combined": pt_comb}
        per_fold.append(row)
        print(f"fold {fold_idx}:  naive={naive:5.1f}   linear={linear:5.1f}   ridge={ridge:5.1f}   "
              f"GBoost={gboost:5.1f}   PT_runs={pt_runs:5.1f}   PT_combined={pt_comb:5.1f}")

    print("\n" + "=" * 78)
    print(f"5-fold CV MAE (seconds), n={len(common_uids)} athletes, split by athlete")
    print("=" * 78)
    print(f"{'Model':30s}  {'Mean':>7s}  {'Std':>6s}  {'per fold':<30s}")
    print("-" * 78)
    for name, label in [
        ("naive",            "Naive (predict mean)"),
        ("linear_hand",      "Linear  (hand-crafted) DEPLOYED"),
        ("ridge_hand",       "Ridge   (hand-crafted)"),
        ("gboost_hand",      "GBoost  (hand-crafted)"),
        ("pytorch_runs",     "PyTorch (run-summaries only)"),
        ("pytorch_combined", "PyTorch (COMBINED)"),
    ]:
        vals = np.array([r[name] for r in per_fold])
        per_fold_str = "[" + ", ".join(f"{v:5.1f}" for v in vals) + "]"
        print(f"{label:30s}  {vals.mean():7.2f}  {vals.std():6.2f}  {per_fold_str}")
    print("=" * 78)


if __name__ == "__main__":
    main()
