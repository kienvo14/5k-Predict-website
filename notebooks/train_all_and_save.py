"""
Train Ridge, Gradient Boosting, and the PyTorch mean-pool MLP on the SAME
data split, then save each to its own folder under notebooks/models/.

Each folder contains everything needed to load the model and make
predictions on new athletes later — model weights, normalization stats,
feature names, and a metadata.json with the held-out MAE.
"""
import json
import os
import pickle

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

from train_5k_from_history import (
    build_athlete_data, AthleteDataset, HistoryModel,
    MAX_RUNS, BATCH_SIZE, EPOCHS, PATIENCE, LR, SEED, DEVICE,
    N_FEAT, FEAT_NAMES,
)
from torch.utils.data import DataLoader
import torch.nn as nn

MODELS_DIR = "models"
VAL_FRAC   = 0.20

def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def train_pytorch(train_uids, val_uids, per_athlete, labels, feat_mean, feat_std):
    train_ds = AthleteDataset(train_uids, per_athlete, labels, MAX_RUNS, feat_mean, feat_std, training=True)
    val_ds   = AthleteDataset(val_uids,   per_athlete, labels, MAX_RUNS, feat_mean, feat_std, training=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    y_train = np.array([labels[u] for u in train_uids])
    y_mean, y_std = float(y_train.mean()), float(y_train.std()) + 1e-6

    torch.manual_seed(SEED)
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

    model.load_state_dict(best_state)
    return model, best_val, y_mean, y_std


def main():
    per_athlete, labels = build_athlete_data()

    # single train/val split (same as train_5k_from_history default)
    uids = np.array(sorted(per_athlete.keys()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(uids)
    n_val = int(len(uids) * VAL_FRAC)
    val_uids   = uids[:n_val]
    train_uids = uids[n_val:]
    print(f"split: {len(train_uids)} train / {len(val_uids)} val athletes")

    # normalize features using TRAIN stats
    train_stack = np.concatenate([per_athlete[u] for u in train_uids], axis=0)
    feat_mean = train_stack.mean(axis=0)
    feat_std  = train_stack.std(axis=0) + 1e-6

    # per-athlete mean features (input for Ridge / GBoost)
    Xtr = np.stack([per_athlete[u].mean(axis=0) for u in train_uids])
    Xva = np.stack([per_athlete[u].mean(axis=0) for u in val_uids])
    ytr = np.array([labels[u] for u in train_uids])
    yva = np.array([labels[u] for u in val_uids])

    # ---------- Ridge ----------
    print("\n[Ridge] training...")
    ridge = Ridge().fit(Xtr, ytr)
    ridge_mae = float(np.abs(ridge.predict(Xva) - yva).mean())
    print(f"  val MAE: {ridge_mae:.1f} s")

    ridge_dir = os.path.join(MODELS_DIR, "ridge")
    os.makedirs(ridge_dir, exist_ok=True)
    with open(os.path.join(ridge_dir, "model.pkl"), "wb") as f:
        pickle.dump(ridge, f)
    save_json(os.path.join(ridge_dir, "metadata.json"), {
        "model": "sklearn.linear_model.Ridge",
        "input": "per-athlete mean of 10 run-summary features",
        "feature_names": FEAT_NAMES,
        "n_train_athletes": len(train_uids),
        "n_val_athletes": len(val_uids),
        "val_MAE_seconds": round(ridge_mae, 2),
        "seed": SEED,
    })
    print(f"  saved: {ridge_dir}/")

    # ---------- Gradient Boosting ----------
    print("\n[GBoost] training...")
    gb = GradientBoostingRegressor(random_state=SEED).fit(Xtr, ytr)
    gb_mae = float(np.abs(gb.predict(Xva) - yva).mean())
    print(f"  val MAE: {gb_mae:.1f} s")

    gb_dir = os.path.join(MODELS_DIR, "gboost")
    os.makedirs(gb_dir, exist_ok=True)
    with open(os.path.join(gb_dir, "model.pkl"), "wb") as f:
        pickle.dump(gb, f)
    save_json(os.path.join(gb_dir, "metadata.json"), {
        "model": "sklearn.ensemble.GradientBoostingRegressor",
        "input": "per-athlete mean of 10 run-summary features",
        "feature_names": FEAT_NAMES,
        "n_train_athletes": len(train_uids),
        "n_val_athletes": len(val_uids),
        "val_MAE_seconds": round(gb_mae, 2),
        "seed": SEED,
    })
    print(f"  saved: {gb_dir}/")

    # ---------- PyTorch mean-pool MLP ----------
    print("\n[PyTorch] training...")
    model, pt_mae, y_mean, y_std = train_pytorch(
        train_uids, val_uids, per_athlete, labels, feat_mean, feat_std)
    print(f"  val MAE: {pt_mae:.1f} s")

    pt_dir = os.path.join(MODELS_DIR, "pytorch")
    os.makedirs(pt_dir, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "feat_mean":   feat_mean.tolist(),
        "feat_std":    feat_std.tolist(),
        "target_mean": y_mean,
        "target_std":  y_std,
        "config":      {"n_feat": N_FEAT, "max_runs": MAX_RUNS},
    }, os.path.join(pt_dir, "model.pt"))
    save_json(os.path.join(pt_dir, "metadata.json"), {
        "model": "HistoryModel (per-run MLP + masked mean-pool + MLP head)",
        "input": "sequence of up to 30 per-athlete run-summary vectors (10 features each)",
        "feature_names": FEAT_NAMES,
        "n_train_athletes": len(train_uids),
        "n_val_athletes": len(val_uids),
        "val_MAE_seconds": round(pt_mae, 2),
        "seed": SEED,
    })
    print(f"  saved: {pt_dir}/")

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  Ridge      val MAE: {ridge_mae:6.2f} s   ->  {ridge_dir}/")
    print(f"  GBoost     val MAE: {gb_mae:6.2f} s   ->  {gb_dir}/")
    print(f"  PyTorch    val MAE: {pt_mae:6.2f} s   ->  {pt_dir}/")


if __name__ == "__main__":
    main()
