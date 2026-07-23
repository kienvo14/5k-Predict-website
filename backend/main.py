"""
5K Predictor API (FastAPI)

On startup it trains the SAME linear model you validated in trainModel.py,
reading features.csv. Then POST /predict takes a runner's inputs, rebuilds the
9 model features (converting km -> meters to match training units), and returns
the predicted 5K time plus a +/- range based on the cross-validated error.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

# The exact 9 columns the model was trained on, in order.
FEATURE_COLS = ["longest_run", "max_hr", "easy_pace", "easy_hr", "aerobic",
                "avg_weekly_distance_m", "active_weeks", "consistency_std", "gender"]

# Cross-validated MAE (seconds) -> used as the +/- prediction range.
CV_MAE = 82

app = FastAPI(title="5K Predictor API")

# Allow the React dev server (localhost:5173) to call this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def train_model() -> LinearRegression:
    """Train the linear model once, exactly like trainModel.py did."""
    df = pd.read_csv("features.csv")
    df["gender"] = df["gender"].map({"male": 0, "female": 1})
    X = df[FEATURE_COLS]
    y = df["fastest_5k"]
    mask = X.notna().all(axis=1) & y.notna()
    return LinearRegression().fit(X[mask], y[mask])


model = train_model()


def fmt(seconds: float) -> str:
    """Seconds -> 'M:SS'."""
    s = max(0, int(round(seconds)))
    return f"{s // 60}:{s % 60:02d}"


class RunnerInput(BaseModel):
    gender: str                    # "male" or "female"
    typical_pace: float            # min/km (e.g. 5.5 = 5:30)
    easy_hr: float                 # avg HR on easy runs (bpm)
    max_hr: float                  # highest HR seen (bpm)
    longest_run_km: float          # longest recent run (km)
    weekly_mileage_km: list[float] # one number per week (km)


@app.get("/")
def root():
    return {"status": "ok", "message": "5K Predictor API is running"}


@app.post("/predict")
def predict(data: RunnerInput):
    # Need at least 2 weeks so consistency (std dev) is meaningful.
    weeks = [w for w in data.weekly_mileage_km if w and w > 0]
    if len(weeks) < 2:
        return {"error": "Enter at least 2 weeks of mileage."}

    # Weekly mileage stats (convert km -> meters to match training units).
    weeks_m = np.array(weeks) * 1000.0
    avg_weekly = float(weeks_m.mean())
    consistency = float(weeks_m.std(ddof=1))   # ddof=1 = sample std, matches pandas
    active_weeks = len(weeks_m)

    aerobic = data.typical_pace / data.easy_hr
    gender_num = 0 if data.gender.lower().startswith("m") else 1

    # Build one row in the exact column order the model expects.
    row = pd.DataFrame([{
        "longest_run": data.longest_run_km * 1000.0,   # km -> meters
        "max_hr": data.max_hr,
        "easy_pace": data.typical_pace,
        "easy_hr": data.easy_hr,
        "aerobic": aerobic,
        "avg_weekly_distance_m": avg_weekly,
        "active_weeks": active_weeks,
        "consistency_std": consistency,
        "gender": gender_num,
    }])[FEATURE_COLS]

    pred = float(model.predict(row)[0])

    return {
        "predicted_seconds": round(pred, 1),
        "predicted_time": fmt(pred),
        "range_low": fmt(pred - CV_MAE),    # faster end
        "range_high": fmt(pred + CV_MAE),   # slower end
        "note": "Estimated best current 5K, +/- ~1:22 (cross-validated error).",
    }
