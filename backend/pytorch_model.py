"""
Lazy-loaded PyTorch predictor for the "better model" path.

IMPORTANT: `torch` is imported lazily, inside functions — never at module import
time. So the main API process pays ZERO torch cost (import time, ~hundreds of MB
of RAM) until a request actually asks for the PyTorch model. The default
LinearRegression path never touches this file.

The model is `HistoryModel` (see notebooks/train_5k_from_history.py):
a per-run MLP encoder -> masked mean-pool over the runs -> MLP head.
Input: a sequence of up to 30 per-run vectors, each 10 features, in this order:
    duration_s(log), distance_km(log), avg_pace, pace_std, avg_hr, hr_std,
    max_hr, elev_gain(log1p), elev_loss(log1p), decouple
Features the website can't measure per run (pace_std, hr_std, elev_*, decouple)
are filled with the model's own training mean so they normalise to a neutral 0.
"""
import os
import math

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "ml", "pytorch", "model.pt")
MAX_RUNS = 30
FEAT_NAMES = [
    "duration_s", "distance_km", "avg_pace", "pace_std", "avg_hr",
    "hr_std", "max_hr", "elev_gain_m", "elev_loss_m", "decouple",
]
N_FEAT = len(FEAT_NAMES)

# Cached across requests once the first PyTorch prediction warms it up.
_state = {"loaded": False, "model": None, "feat_mean": None, "feat_std": None,
          "y_mean": None, "y_std": None, "torch": None}


def is_installed() -> bool:
    """True if torch is available — WITHOUT importing it (find_spec only inspects
    metadata). This keeps /models cheap: checking availability must not trigger the
    heavy torch import; that happens only in _ensure_loaded() on a real prediction."""
    import importlib.util
    return importlib.util.find_spec("torch") is not None


def _build_model(torch):
    """Recreate the HistoryModel architecture (must match the training script)."""
    import torch.nn as nn

    HIDDEN, DROPOUT = 128, 0.3

    class HistoryModel(nn.Module):
        def __init__(self, in_dim=N_FEAT, hidden=HIDDEN, dropout=DROPOUT):
            super().__init__()
            self.per_run = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.ReLU(),
            )
            self.head = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden, 1),
            )

        def forward(self, x, mask):
            h = self.per_run(x)
            w = mask.unsqueeze(-1)
            pooled = (h * w).sum(dim=1) / w.sum(dim=1).clamp(min=1)
            return self.head(pooled).squeeze(-1)

    return HistoryModel()


def _ensure_loaded():
    """Import torch + load weights once; cache for later calls. Raises if torch missing."""
    if _state["loaded"]:
        return
    import torch  # lazy — this is the ~hundreds-of-MB cost we defer until now
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = _build_model(torch)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    _state.update({
        "loaded": True, "torch": torch, "model": model,
        "feat_mean": ckpt["feat_mean"], "feat_std": ckpt["feat_std"],
        "y_mean": ckpt["target_mean"], "y_std": ckpt["target_std"],
    })


def run_features(distance_km: float, duration_s: float, avg_hr, max_hr) -> list:
    """Turn one run's available summary into the 10-feature vector.

    Only 5 features are truly measurable from a Strava row; the other 5 are left
    as None here and filled with the training mean (neutral) in predict().
    """
    avg_pace = (duration_s / 60.0) / distance_km if distance_km > 0 else None
    return [
        math.log(max(duration_s, 1.0)),          # duration_s (log)
        math.log(max(distance_km, 0.1)),         # distance_km (log)
        avg_pace,                                 # avg_pace (min/km)
        None,                                     # pace_std   -> fill mean
        float(avg_hr) if avg_hr else None,        # avg_hr
        None,                                     # hr_std     -> fill mean
        float(max_hr) if max_hr else None,        # max_hr
        None,                                     # elev_gain  -> fill mean
        None,                                     # elev_loss  -> fill mean
        None,                                     # decouple   -> fill mean
    ]


def predict(runs: list) -> float:
    """runs: list of 10-length feature rows (Nones allowed). Returns 5K seconds."""
    _ensure_loaded()
    torch = _state["torch"]
    fmean, fstd = _state["feat_mean"], _state["feat_std"]

    rows = runs[-MAX_RUNS:]
    T = len(rows)
    x = torch.zeros(1, MAX_RUNS, N_FEAT, dtype=torch.float32)
    mask = torch.zeros(1, MAX_RUNS, dtype=torch.float32)
    for i, row in enumerate(rows):
        for j in range(N_FEAT):
            raw = row[j]
            if raw is None:            # unmeasured feature -> training mean (normalises to 0)
                raw = fmean[j]
            x[0, i, j] = (raw - fmean[j]) / fstd[j]
        mask[0, i] = 1.0

    with torch.no_grad():
        norm = model_forward(torch, x, mask)
    return float(norm * _state["y_std"] + _state["y_mean"])


def model_forward(torch, x, mask):
    return _state["model"](x, mask).item()
