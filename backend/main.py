"""
5K Predictor API (FastAPI)

Prediction:
  POST /predict            -> manual inputs (works logged out or in)
  POST /predict-from-file  -> Strava activities.csv (last 16 weeks)

Auth (SQLite-backed accounts + session tokens):
  POST /signup   -> create account, returns a token
  POST /login    -> returns a token
  POST /logout   -> invalidates the token
  GET  /me       -> who am I (from the token)

Per-user data:
  POST /feedback -> attach real 5K PR to a prediction
  POST /claim    -> attach an anonymous prediction to the logged-in user
  GET  /history  -> the logged-in user's predictions

Passwords are salted + hashed with PBKDF2 (stdlib). Tokens are opaque random
strings stored in a sessions table (a simple, real alternative to JWT).
"""
import io
import json
import sqlite3
import hashlib
import secrets
from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, File, Form, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

FEATURE_COLS = ["longest_run", "max_hr", "easy_pace", "easy_hr", "aerobic",
                "avg_weekly_distance_m", "active_weeks", "consistency_std", "gender"]
CV_MAE = 82
DB_PATH = "app.db"

app = FastAPI(title="5K Predictor API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ---------------- database ----------------
def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    con = db()
    con.execute("""CREATE TABLE IF NOT EXISTS users (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        username   TEXT UNIQUE NOT NULL,
        salt       TEXT NOT NULL,
        pw_hash    TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS sessions (
        token      TEXT PRIMARY KEY,
        user_id    INTEGER NOT NULL,
        created_at TEXT NOT NULL
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at        TEXT NOT NULL,
        user_id           INTEGER,
        source            TEXT,
        gender            TEXT,
        predicted_seconds REAL,
        actual_pr_seconds REAL
    )""")
    # lightweight migration: add the input-data columns if they don't exist yet
    for col, coltype in [("typical_pace", "REAL"), ("avg_weekly_km", "REAL"),
                         ("active_weeks", "INTEGER"), ("longest_km", "REAL"),
                         ("easy_hr", "REAL"), ("max_hr", "REAL")]:
        try:
            con.execute(f"ALTER TABLE predictions ADD COLUMN {col} {coltype}")
        except sqlite3.OperationalError:
            pass  # column already exists
    con.commit()
    con.close()


init_db()


# ---------------- auth helpers ----------------
def hash_pw(password: str, salt: str | None = None):
    salt = salt or secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()
    return salt, h


def verify_pw(password: str, salt: str, pw_hash: str) -> bool:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex() == pw_hash


def new_session(user_id: int) -> str:
    token = secrets.token_hex(24)
    con = db()
    con.execute("INSERT INTO sessions (token, user_id, created_at) VALUES (?,?,?)",
                (token, user_id, datetime.now(timezone.utc).isoformat()))
    con.commit()
    con.close()
    return token


def get_user_id(authorization: str | None = Header(default=None)):
    """Resolve the Authorization header -> user_id, or None if not logged in."""
    if not authorization:
        return None
    token = authorization.replace("Bearer", "").strip()
    con = db()
    row = con.execute("SELECT user_id FROM sessions WHERE token=?", (token,)).fetchone()
    con.close()
    return row["user_id"] if row else None


# ---------------- model ----------------
def train_model() -> LinearRegression:
    df = pd.read_csv("features.csv")
    df["gender"] = df["gender"].map({"male": 0, "female": 1})
    X = df[FEATURE_COLS]
    y = df["fastest_5k"]
    mask = X.notna().all(axis=1) & y.notna()
    return LinearRegression().fit(X[mask], y[mask])


model = train_model()


def fmt(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    return f"{s // 60}:{s % 60:02d}"


def parse_time(t: str) -> float:
    t = t.strip()
    if ":" in t:
        m, s = t.split(":")
        return int(m) * 60 + float(s)
    return float(t) * 60


def core_predict(gender, typical_pace, easy_hr, max_hr, longest_km, weekly_km):
    weeks = [w for w in weekly_km if w and w > 0]
    if len(weeks) < 2:
        return {"error": "Need at least 2 weeks of mileage."}

    weeks_m = np.array(weeks) * 1000.0
    row = pd.DataFrame([{
        "longest_run": longest_km * 1000.0,
        "max_hr": max_hr,
        "easy_pace": typical_pace,
        "easy_hr": easy_hr,
        "aerobic": typical_pace / easy_hr,
        "avg_weekly_distance_m": float(weeks_m.mean()),
        "active_weeks": len(weeks_m),
        "consistency_std": float(weeks_m.std(ddof=1)),
        "gender": 0 if gender.lower().startswith("m") else 1,
    }])[FEATURE_COLS]

    pred = float(model.predict(row)[0])
    return {
        "predicted_seconds": round(pred, 1),
        "predicted_time": fmt(pred),
        "range_low": fmt(pred - CV_MAE),
        "range_high": fmt(pred + CV_MAE),
        "note": "Estimated best current 5K, +/- ~1:22 (cross-validated error).",
        "weeks_used": len(weeks_m),
        # the actual input data — stored to SQLite so each prediction is self-contained
        "_inputs": {
            "typical_pace": round(typical_pace, 2),
            "avg_weekly_km": round(float(weeks_m.mean()) / 1000.0, 2),
            "active_weeks": len(weeks_m),
            "longest_km": round(longest_km, 1),
            "easy_hr": round(easy_hr, 0),
            "max_hr": round(max_hr, 0),
        },
    }


def save_prediction(source, gender, predicted_seconds, user_id, inp):
    con = db()
    cur = con.execute(
        """INSERT INTO predictions
           (created_at, user_id, source, gender, predicted_seconds, actual_pr_seconds,
            typical_pace, avg_weekly_km, active_weeks, longest_km, easy_hr, max_hr)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (datetime.now(timezone.utc).isoformat(), user_id, source, gender, predicted_seconds, None,
         inp["typical_pace"], inp["avg_weekly_km"], inp["active_weeks"],
         inp["longest_km"], inp["easy_hr"], inp["max_hr"]),
    )
    con.commit()
    rid = cur.lastrowid
    con.close()
    return rid


def finalize(result, source, gender, user_id):
    if "error" not in result:
        result["id"] = save_prediction(source, gender, result["predicted_seconds"],
                                       user_id, result.pop("_inputs"))
    return result


# ---------------- models ----------------
class Credentials(BaseModel):
    username: str
    password: str


class RunnerInput(BaseModel):
    gender: str
    typical_pace: float
    easy_hr: float
    max_hr: float
    longest_run_km: float
    weekly_mileage_km: list[float]


class Feedback(BaseModel):
    id: int
    pr_time: str


class Claim(BaseModel):
    id: int


# ---------------- auth endpoints ----------------
@app.get("/")
def root():
    return {"status": "ok", "message": "5K Predictor API is running"}


@app.post("/signup")
def signup(data: Credentials):
    if len(data.username) < 3 or len(data.password) < 4:
        return {"error": "Username needs 3+ chars, password 4+."}
    con = db()
    if con.execute("SELECT id FROM users WHERE username=?", (data.username,)).fetchone():
        con.close()
        return {"error": "That username is taken."}
    salt, h = hash_pw(data.password)
    cur = con.execute(
        "INSERT INTO users (username, salt, pw_hash, created_at) VALUES (?,?,?,?)",
        (data.username, salt, h, datetime.now(timezone.utc).isoformat()),
    )
    con.commit()
    uid = cur.lastrowid
    con.close()
    return {"token": new_session(uid), "username": data.username}


@app.post("/login")
def login(data: Credentials):
    con = db()
    row = con.execute("SELECT * FROM users WHERE username=?", (data.username,)).fetchone()
    con.close()
    if not row or not verify_pw(data.password, row["salt"], row["pw_hash"]):
        return {"error": "Wrong username or password."}
    return {"token": new_session(row["id"]), "username": row["username"]}


@app.post("/logout")
def logout(authorization: str | None = Header(default=None)):
    if authorization:
        token = authorization.replace("Bearer", "").strip()
        con = db()
        con.execute("DELETE FROM sessions WHERE token=?", (token,))
        con.commit()
        con.close()
    return {"ok": True}


@app.get("/me")
def me(user_id: int | None = Depends(get_user_id)):
    if user_id is None:
        return {"user": None}
    con = db()
    row = con.execute("SELECT username FROM users WHERE id=?", (user_id,)).fetchone()
    con.close()
    return {"user": row["username"] if row else None}


# ---------------- prediction endpoints ----------------
@app.post("/predict")
def predict(data: RunnerInput, user_id: int | None = Depends(get_user_id)):
    result = core_predict(data.gender, data.typical_pace, data.easy_hr,
                          data.max_hr, data.longest_run_km, data.weekly_mileage_km)
    return finalize(result, "manual", data.gender, user_id)


def find_col(df: pd.DataFrame, *candidates: str):
    lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for cand in candidates:
        for c in df.columns:
            if cand.lower() in c.lower():
                return c
    return None


@app.post("/predict-from-file")
async def predict_from_file(gender: str = Form(...), file: UploadFile = File(...),
                            user_id: int | None = Depends(get_user_id)):
    try:
        raw = await file.read()
        name = (file.filename or "").lower()
        if name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(raw))          # Excel export
        else:
            df = pd.read_csv(io.BytesIO(raw))            # CSV export
    except Exception:
        return {"error": "Could not read the file. Upload your Strava activities .csv or .xlsx."}

    type_col = find_col(df, "Activity Type")
    date_col = find_col(df, "Activity Date", "Date")
    dist_col = find_col(df, "Distance")
    time_col = find_col(df, "Moving Time", "Elapsed Time")
    ahr_col = find_col(df, "Average Heart Rate", "Average Heart")
    mhr_col = find_col(df, "Max Heart Rate", "Max Heart")

    if not (type_col and date_col and dist_col and time_col):
        return {"error": "This doesn't look like a Strava activities.csv export "
                         "(missing Activity Type / Date / Distance / Time columns)."}

    runs = df[df[type_col].astype(str).str.contains("Run", case=False, na=False)].copy()
    if runs.empty:
        return {"error": "No running activities found in the file."}

    dist = pd.to_numeric(runs[dist_col], errors="coerce")
    if dist.median() > 1000:
        dist = dist / 1000.0
    runs["dist_km"] = dist
    runs["dur_min"] = pd.to_numeric(runs[time_col], errors="coerce") / 60.0
    runs["date"] = pd.to_datetime(runs[date_col], errors="coerce")
    runs = runs.dropna(subset=["dist_km", "dur_min", "date"])
    runs = runs[(runs["dist_km"] > 1) & (runs["dur_min"] > 5)]
    if len(runs) < 3:
        return {"error": "Not enough valid runs found (need at least 3 with distance + time)."}

    cutoff = runs["date"].max() - pd.Timedelta(weeks=16)
    recent = runs[runs["date"] >= cutoff].copy()
    recent["pace"] = recent["dur_min"] / recent["dist_km"]
    recent = recent[(recent["pace"] >= 3) & (recent["pace"] <= 12)]
    if len(recent) < 2:
        return {"error": "Not enough recent runs (last 16 weeks) to make a prediction."}

    iso = recent["date"].dt.isocalendar()
    recent["yw"] = iso.year.astype(str) + "-" + iso.week.astype(str)
    weekly = recent.groupby("yw")["dist_km"].sum().tolist()

    typical_pace = float(recent["pace"].median())
    ahr = pd.to_numeric(recent[ahr_col], errors="coerce") if ahr_col else pd.Series(dtype=float)
    easy_hr = float(ahr.median()) if ahr.notna().any() else 145.0
    if mhr_col and pd.to_numeric(recent[mhr_col], errors="coerce").notna().any():
        max_hr = float(pd.to_numeric(recent[mhr_col], errors="coerce").max())
    elif ahr.notna().any():
        max_hr = float(ahr.max())
    else:
        max_hr = 185.0
    longest = float(recent["dist_km"].max())

    result = core_predict(gender, typical_pace, easy_hr, max_hr, longest, weekly)
    if "error" not in result:
        result["detected"] = {
            "runs_used": len(recent),
            "typical_pace": round(typical_pace, 2),
            "avg_hr": round(easy_hr, 0),
            "longest_km": round(longest, 1),
        }
    return finalize(result, "strava", gender, user_id)


# ---------------- per-user data ----------------
@app.post("/feedback")
def feedback(fb: Feedback):
    try:
        actual = parse_time(fb.pr_time)
    except Exception:
        return {"error": "Enter your PR as MM:SS, e.g. 22:30"}

    con = db()
    prow = con.execute("SELECT predicted_seconds FROM predictions WHERE id=?", (fb.id,)).fetchone()
    if prow is None:
        con.close()
        return {"error": "Prediction not found."}
    con.execute("UPDATE predictions SET actual_pr_seconds=? WHERE id=?", (actual, fb.id))
    con.commit()
    con.close()

    predicted = prow["predicted_seconds"]
    diff = round(abs(predicted - actual))
    verdict = ("Spot on — within the model's margin." if diff <= CV_MAE
               else "Pretty close." if diff <= 2 * CV_MAE else "Off this time.")
    return {"ok": True, "predicted_time": fmt(predicted), "actual_time": fmt(actual),
            "diff_seconds": diff, "verdict": verdict}


@app.post("/claim")
def claim(data: Claim, user_id: int | None = Depends(get_user_id)):
    """Attach an anonymous prediction to the now-logged-in user."""
    if user_id is None:
        return {"error": "Not logged in."}
    con = db()
    con.execute("UPDATE predictions SET user_id=? WHERE id=? AND user_id IS NULL",
                (user_id, data.id))
    con.commit()
    con.close()
    return {"ok": True}


@app.get("/history")
def history(user_id: int | None = Depends(get_user_id)):
    if user_id is None:
        return {"error": "Log in to see your history.", "history": []}
    con = db()
    rows = con.execute(
        "SELECT * FROM predictions WHERE user_id=? ORDER BY id DESC LIMIT 50", (user_id,)
    ).fetchall()
    con.close()
    out = []
    for r in rows:
        actual = r["actual_pr_seconds"]
        out.append({
            "id": r["id"],
            "date": r["created_at"][:10],
            "source": r["source"],
            "predicted_time": fmt(r["predicted_seconds"]),
            "actual_pr_time": fmt(actual) if actual is not None else None,
            "diff_seconds": round(abs(r["predicted_seconds"] - actual)) if actual is not None else None,
            "typical_pace": r["typical_pace"],
            "avg_weekly_km": r["avg_weekly_km"],
            "active_weeks": r["active_weeks"],
        })
    return {"history": out}
