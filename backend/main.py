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
import os
import json
import sqlite3
import hashlib
import secrets
from datetime import datetime, timezone

# Optional: load a local .env for dev (safe if the file/library isn't there).
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from fastapi import FastAPI, UploadFile, File, Form, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import pytorch_model  # safe to import — it does NOT load torch until a request needs it

FEATURE_COLS = ["longest_run", "max_hr", "easy_pace", "easy_hr", "aerobic",
                "avg_weekly_distance_m", "active_weeks", "consistency_std", "gender"]
# full column order of features.csv (features + label), used when appending new rows
CSV_COLS = ["userId", "longest_run", "gender", "max_hr", "easy_pace", "easy_hr",
            "aerobic", "avg_weekly_distance_m", "active_weeks", "consistency_std",
            "fastest_5k", "fastest_5k_str"]
CV_MAE = 82
PYTORCH_MAE = 76  # held-out MAE of the PyTorch HistoryModel (models/pytorch/metadata.json)
DB_PATH = "app.db"
FEATURES_PATH = "features.csv"
MIN_NEW_TO_RETRAIN = 10   # append real-PR rows to features.csv once this many accumulate

# DATABASE_URL (env var) picks Postgres for prod. Empty -> SQLite for local dev.
# NEVER hardcode the URL here — set it in Render dashboard or a local .env (gitignored).
_DB_URL = os.environ.get("DATABASE_URL", "").strip()
if _DB_URL.startswith("postgres://"):
    _DB_URL = "postgresql://" + _DB_URL[len("postgres://"):]  # Render hands "postgres://", psycopg wants "postgresql://"
IS_PG = bool(_DB_URL)
PK = "SERIAL PRIMARY KEY" if IS_PG else "INTEGER PRIMARY KEY AUTOINCREMENT"

app = FastAPI(title="5K Predictor API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ---------------- database (SQLite locally, Postgres in production) ----------------
class Db:
    """Thin wrapper so the same code speaks to both SQLite and Postgres.

    - Translates SQLite-style '?' placeholders into psycopg-style '%s'.
    - Returns dict-like rows on both (access with row["col"]).
    - Same .execute / .executemany / .commit / .close surface as sqlite3.Connection.
    """
    def __init__(self):
        if IS_PG:
            import psycopg  # type: ignore
            from psycopg.rows import dict_row  # type: ignore
            self.con = psycopg.connect(_DB_URL, row_factory=dict_row)
        else:
            self.con = sqlite3.connect(DB_PATH)
            self.con.row_factory = sqlite3.Row

    def _sql(self, s: str) -> str:
        return s.replace("?", "%s") if IS_PG else s

    def execute(self, sql, params=()):
        if IS_PG:
            cur = self.con.cursor()
            cur.execute(self._sql(sql), params)
            return cur
        return self.con.execute(sql, params)

    def executemany(self, sql, seq):
        if IS_PG:
            cur = self.con.cursor()
            cur.executemany(self._sql(sql), seq)
            return cur
        return self.con.executemany(sql, seq)

    def commit(self):
        self.con.commit()

    def close(self):
        try:
            self.con.close()
        except Exception:
            pass


def db() -> Db:
    return Db()


def _try_alter(sql: str):
    """ALTER TABLE ADD COLUMN, ignoring 'already exists'. Each ALTER gets its
    own connection — on Postgres a failed statement aborts the transaction,
    so isolating them keeps the rest of migration going."""
    con = db()
    try:
        con.execute(sql)
        con.commit()
    except Exception:
        pass
    con.close()


def init_db():
    con = db()
    con.execute(f"""CREATE TABLE IF NOT EXISTS users (
        id         {PK},
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
    con.execute(f"""CREATE TABLE IF NOT EXISTS predictions (
        id                {PK},
        created_at        TEXT NOT NULL,
        user_id           INTEGER,
        source            TEXT,
        gender            TEXT,
        predicted_seconds REAL,
        actual_pr_seconds REAL
    )""")
    con.execute(f"""CREATE TABLE IF NOT EXISTS runs (
        id       {PK},
        user_id  INTEGER NOT NULL,
        date     TEXT,
        year     INTEGER,
        week     INTEGER,
        dist_km  REAL,
        pace     REAL,
        hr       REAL
    )""")
    con.commit()
    con.close()
    # additive columns (safe to re-run; each uses its own conn so pg tx doesn't abort)
    for col, coltype in [("typical_pace", "REAL"), ("avg_weekly_km", "REAL"),
                         ("active_weeks", "INTEGER"), ("longest_km", "REAL"),
                         ("easy_hr", "REAL"), ("max_hr", "REAL"),
                         ("weekly_km_json", "TEXT"), ("exported", "INTEGER"),
                         ("weeks_meta_json", "TEXT")]:
        _try_alter(f"ALTER TABLE predictions ADD COLUMN {col} {coltype}")


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
    df = pd.read_csv(FEATURES_PATH)
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
            "weekly_km": [round(w, 1) for w in weeks],  # the individual weeks, to reload later
        },
    }


def save_prediction(source, gender, predicted_seconds, user_id, inp, meta=None):
    """
    Save a prediction. `meta` is a list of {"year", "week"} per bar — the STABLE
    identity used to attach runs later. Manual weeks use sentinel year=9000; Strava
    weeks use the real year/week from the upload.
    """
    if meta is None:
        meta = [{"year": 9000, "week": i} for i in range(len(inp["weekly_km"]))]
    con = db()
    cur = con.execute(
        """INSERT INTO predictions
           (created_at, user_id, source, gender, predicted_seconds, actual_pr_seconds,
            typical_pace, avg_weekly_km, active_weeks, longest_km, easy_hr, max_hr,
            weekly_km_json, weeks_meta_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           RETURNING id""",
        (datetime.now(timezone.utc).isoformat(), user_id, source, gender, predicted_seconds, None,
         inp["typical_pace"], inp["avg_weekly_km"], inp["active_weeks"],
         inp["longest_km"], inp["easy_hr"], inp["max_hr"],
         json.dumps(inp["weekly_km"]), json.dumps(meta)),
    )
    rid = cur.fetchone()["id"]
    con.commit()
    con.close()
    return rid


def finalize(result, source, gender, user_id, meta=None):
    if "error" not in result:
        result["id"] = save_prediction(source, gender, result["predicted_seconds"],
                                       user_id, result.pop("_inputs"), meta)
    return result


def save_user_runs(user_id, recent):
    """Replace this user's stored runs with the latest Strava upload (for /progress)."""
    con = db()
    con.execute("DELETE FROM runs WHERE user_id=?", (user_id,))
    recs = []
    for _, row in recent.iterrows():
        hr = row.get("hr")
        recs.append((
            user_id, row["date"].date().isoformat(), int(row["yr"]), int(row["wk"]),
            round(float(row["dist_km"]), 2), round(float(row["pace"]), 2),
            None if pd.isna(hr) else round(float(hr), 0),
        ))
    con.executemany(
        "INSERT INTO runs (user_id, date, year, week, dist_km, pace, hr) VALUES (?,?,?,?,?,?,?)",
        recs,
    )
    con.commit()
    con.close()


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


@app.get("/models")
def models_info():
    """Model-comparison card: real held-out MAE for each model + whether torch is available here."""
    with open("model_cards.json") as f:
        cards = json.load(f)
    cards["pytorch_available"] = pytorch_model.is_installed()
    return cards


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
        "INSERT INTO users (username, salt, pw_hash, created_at) VALUES (?,?,?,?) RETURNING id",
        (data.username, salt, h, datetime.now(timezone.utc).isoformat()),
    )
    uid = cur.fetchone()["id"]
    con.commit()
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
                            model: str = Form("linear"),
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
    recent["yr"] = iso.year.values
    recent["wk"] = iso.week.values
    recent["yw"] = recent["yr"].astype(str) + "-" + recent["wk"].astype(str)
    recent["hr"] = pd.to_numeric(recent[ahr_col], errors="coerce") if ahr_col else np.nan
    # ordered weekly totals + the corresponding (year, week) meta for stable matching
    grouped = recent.sort_values("date").groupby("yw", sort=False)
    weekly = grouped["dist_km"].sum().tolist()
    yw_order = [k for k, _ in grouped]
    strava_meta = [{"year": int(k.split("-")[0]), "week": int(k.split("-")[1])} for k in yw_order]

    typical_pace = float(recent["pace"].median())
    ahr = recent["hr"]
    easy_hr = float(ahr.median()) if ahr.notna().any() else 145.0
    if mhr_col and pd.to_numeric(recent[mhr_col], errors="coerce").notna().any():
        max_hr = float(pd.to_numeric(recent[mhr_col], errors="coerce").max())
    elif ahr.notna().any():
        max_hr = float(ahr.max())
    else:
        max_hr = 185.0
    longest = float(recent["dist_km"].max())

    result = core_predict(gender, typical_pace, easy_hr, max_hr, longest, weekly)

    # Optional: re-price with the PyTorch model. torch is imported ONLY here, on
    # demand — the default 'linear' path above never loads it.
    if model == "pytorch" and "error" not in result:
        if not pytorch_model.is_installed():
            result["model_note"] = ("PyTorch model unavailable in this environment — "
                                    "showing the LinearRegression estimate.")
            result["model"] = "linear"
        else:
            mh = pd.to_numeric(recent[mhr_col], errors="coerce") if mhr_col else None
            run_rows = []
            for i, (_, r) in enumerate(recent.iterrows()):
                a_hr = float(r["hr"]) if pd.notna(r["hr"]) else None
                m_hr = float(mh.iloc[i]) if (mh is not None and pd.notna(mh.iloc[i])) else a_hr
                run_rows.append(pytorch_model.run_features(
                    float(r["dist_km"]), float(r["dur_min"]) * 60.0, a_hr, m_hr))
            pred = pytorch_model.predict(run_rows)
            result["predicted_seconds"] = round(pred, 1)
            result["predicted_time"] = fmt(pred)
            result["range_low"] = fmt(pred - PYTORCH_MAE)
            result["range_high"] = fmt(pred + PYTORCH_MAE)
            result["model"] = "pytorch"
            result["note"] = (f"PyTorch HistoryModel over your {len(recent)} runs, "
                              f"+/- ~{PYTORCH_MAE}s (held-out MAE).")

    if "error" not in result:
        result.setdefault("model", "linear")
        result["detected"] = {
            "runs_used": len(recent),
            "typical_pace": round(typical_pace, 2),
            "avg_hr": round(easy_hr, 0),
            "longest_km": round(longest, 1),
        }
    result = finalize(result, "strava", gender, user_id, meta=strava_meta)
    if user_id and "error" not in result:
        save_user_runs(user_id, recent)  # store runs for the Progress page
    return result


# ---------------- per-user data ----------------
def export_new_training_data(min_new: int = MIN_NEW_TO_RETRAIN):
    """
    Data flywheel: predictions that have a REAL PR (from /feedback) are real
    labeled examples. Once `min_new` of them accumulate, append them to
    features.csv (mapping units to match) and retrain the model on the spot.
    """
    con = db()
    rows = con.execute(
        "SELECT * FROM predictions WHERE actual_pr_seconds IS NOT NULL "
        "AND (exported IS NULL OR exported = 0)"
    ).fetchall()

    ready = []
    for r in rows:
        weekly = []
        if r["weekly_km_json"]:
            try:
                weekly = json.loads(r["weekly_km_json"])
            except Exception:
                weekly = []
        weekly_m = [w * 1000 for w in weekly if w and w > 0]
        # need complete, sane inputs to form a valid training row
        if len(weekly_m) >= 2 and r["easy_hr"] and r["typical_pace"] and r["gender"]:
            ready.append((r, weekly_m))

    if len(ready) < min_new:
        con.close()
        return {"added": 0, "pending": len(ready), "needed": min_new}

    new_rows, ids = [], []
    for r, weekly_m in ready:
        new_rows.append({
            "userId": f"user{r['user_id'] or 0}_{r['id']}",
            "longest_run": round((r["longest_km"] or 0) * 1000, 1),   # km -> m
            "gender": r["gender"],
            "max_hr": r["max_hr"],
            "easy_pace": r["typical_pace"],
            "easy_hr": r["easy_hr"],
            "aerobic": round(r["typical_pace"] / r["easy_hr"], 6),
            "avg_weekly_distance_m": round((r["avg_weekly_km"] or 0) * 1000, 1),
            "active_weeks": r["active_weeks"],
            "consistency_std": round(float(np.std(weekly_m, ddof=1)), 1),
            "fastest_5k": r["actual_pr_seconds"],                     # the REAL label
            "fastest_5k_str": fmt(r["actual_pr_seconds"]),
        })
        ids.append(r["id"])

    pd.DataFrame(new_rows)[CSV_COLS].to_csv(FEATURES_PATH, mode="a", header=False, index=False)
    con.executemany("UPDATE predictions SET exported = 1 WHERE id = ?", [(i,) for i in ids])
    con.commit()
    con.close()

    global model
    model = train_model()   # retrain on the enlarged dataset
    return {"added": len(new_rows), "retrained": True}


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

    # data flywheel: this new real-PR row may trigger a retrain
    training = export_new_training_data()

    return {"ok": True, "predicted_time": fmt(predicted), "actual_time": fmt(actual),
            "diff_seconds": diff, "verdict": verdict, "training": training}


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


@app.get("/progress")
def progress(user_id: int | None = Depends(get_user_id)):
    """
    Weekly mileage chart. The BARS come from the user's latest prediction's
    weekly totals, so manually-added weeks show up too. Per-run detail is layered
    in from the runs table (their latest Strava upload) wherever a week matches.
    """
    if user_id is None:
        return {"error": "Log in to see your progress.", "weeks": []}

    con = db()
    prow = con.execute(
        "SELECT id, weekly_km_json, weeks_meta_json FROM predictions "
        "WHERE user_id=? AND weekly_km_json IS NOT NULL ORDER BY id DESC LIMIT 1",
        (user_id,),
    ).fetchone()
    rows = con.execute(
        "SELECT * FROM runs WHERE user_id=? ORDER BY year, week, date", (user_id,)
    ).fetchall()

    latest_weeks: list[float] = []
    meta: list[dict] = []
    pred_id = None
    if prow:
        pred_id = prow["id"]
        try:
            latest_weeks = json.loads(prow["weekly_km_json"] or "[]")
        except Exception:
            latest_weeks = []
        try:
            meta = json.loads(prow["weeks_meta_json"] or "[]")
        except Exception:
            meta = []

    # Group all stored runs by identity (year, week).
    from collections import OrderedDict
    groups: "OrderedDict[tuple, list]" = OrderedDict()
    for r in rows:
        groups.setdefault((r["year"], r["week"]), []).append(r)
    strava_keys = [k for k in groups.keys() if k[0] < 9000]

    # Backfill meta for old predictions (before this fix). Strava predictions get
    # matched to Strava week keys; manual predictions get sentinel keys (9000, i).
    if latest_weeks and not meta:
        if len(strava_keys) == len(latest_weeks):
            meta = [{"year": k[0], "week": k[1]} for k in strava_keys]
        else:
            meta = [{"year": 9000, "week": i} for i in range(len(latest_weeks))]
        if pred_id is not None:
            con.execute("UPDATE predictions SET weeks_meta_json=? WHERE id=?",
                        (json.dumps(meta), pred_id))
            con.commit()
    con.close()

    n = len(latest_weeks)
    if n == 0:
        return {"weeks": []}

    def detail(runs, mileage, i, key, label):
        paces = [x["pace"] for x in runs if x["pace"] is not None]
        hrs = [x["hr"] for x in runs if x["hr"] is not None]
        return {
            "idx": i,
            "week_key": f"{key[0]}-{key[1]}",   # STABLE identity for /add-run
            "year": key[0], "week": key[1],
            "label": label,
            "mileage_km": mileage,
            "num_runs": len(runs),
            "avg_pace": round(sum(paces) / len(paces), 2) if paces else None,
            "avg_hr": round(sum(hrs) / len(hrs)) if hrs else None,
            "runs": [{
                "id": x["id"],
                "date": x["date"],
                "dist_km": round(x["dist_km"], 2),
                "pace": round(x["pace"], 2) if x["pace"] is not None else None,
                "hr": x["hr"],
            } for x in runs],
        }

    weeks = []
    for i in range(n):
        m = meta[i] if i < len(meta) else {"year": 9000, "week": i}
        key = (m["year"], m["week"])
        runs = groups.get(key, [])
        mileage = round(float(latest_weeks[i]), 1)
        label = f"W{key[1]}" if key[0] < 9000 else f"Wk {i + 1}"
        weeks.append(detail(runs, mileage, i, key, label))
    return {"weeks": weeks[-16:]}


class AddRun(BaseModel):
    week_key: str                   # "year-week" from /progress (e.g. "2026-15" or "9000-2")
    dist_km: float
    pace: str
    date: str | None = None
    hr: float | None = None


@app.post("/add-run")
def add_run(data: AddRun, user_id: int | None = Depends(get_user_id)):
    """Add a run to a specific week (matched by stable key, not position)."""
    if user_id is None:
        return {"error": "Log in first."}
    if data.dist_km <= 0:
        return {"error": "Distance must be greater than 0."}
    try:
        pace_dec = parse_time(data.pace) / 60.0
    except Exception:
        return {"error": "Enter pace as mm:ss, e.g. 5:30"}
    try:
        yr_s, wk_s = data.week_key.split("-", 1)
        year, week = int(yr_s), int(wk_s)
    except Exception:
        return {"error": "Invalid week key."}

    run_date = (data.date or "").strip() or datetime.now(timezone.utc).date().isoformat()
    hr = None
    if data.hr is not None:
        if data.hr < 60 or data.hr > 230:
            return {"error": "HR looks off (expected 60–230 bpm)."}
        hr = round(float(data.hr))

    con = db()
    con.execute(
        "INSERT INTO runs (user_id, date, year, week, dist_km, pace, hr) VALUES (?,?,?,?,?,?,?)",
        (user_id, run_date, year, week, round(data.dist_km, 2), round(pace_dec, 3), hr),
    )
    total = con.execute(
        "SELECT COALESCE(SUM(dist_km), 0) AS total FROM runs WHERE user_id=? AND year=? AND week=?",
        (user_id, year, week),
    ).fetchone()["total"]
    # keep the chart bar in sync by finding this week's position in the prediction's meta
    prow = con.execute(
        "SELECT id, weekly_km_json, weeks_meta_json FROM predictions "
        "WHERE user_id=? AND weekly_km_json IS NOT NULL ORDER BY id DESC LIMIT 1",
        (user_id,),
    ).fetchone()
    if prow and prow["weekly_km_json"] and prow["weeks_meta_json"]:
        try:
            wk_list = json.loads(prow["weekly_km_json"])
            meta = json.loads(prow["weeks_meta_json"])
        except Exception:
            wk_list, meta = [], []
        for i, m in enumerate(meta):
            if m.get("year") == year and m.get("week") == week and i < len(wk_list):
                wk_list[i] = round(float(total), 1)
                con.execute("UPDATE predictions SET weekly_km_json=? WHERE id=?",
                            (json.dumps(wk_list), prow["id"]))
                break
    con.commit()
    con.close()
    return {"ok": True, "week_total_km": round(float(total), 1)}


def resync_week_total(con, user_id: int, year: int, week: int):
    """Keep the prediction's bar in sync with the sum of the runs in that week."""
    total = con.execute(
        "SELECT COALESCE(SUM(dist_km), 0) AS total FROM runs WHERE user_id=? AND year=? AND week=?",
        (user_id, year, week),
    ).fetchone()["total"]
    prow = con.execute(
        "SELECT id, weekly_km_json, weeks_meta_json FROM predictions "
        "WHERE user_id=? AND weekly_km_json IS NOT NULL ORDER BY id DESC LIMIT 1",
        (user_id,),
    ).fetchone()
    if not prow:
        return
    try:
        wk_list = json.loads(prow["weekly_km_json"] or "[]")
        meta = json.loads(prow["weeks_meta_json"] or "[]")
    except Exception:
        return
    for i, m in enumerate(meta):
        if m.get("year") == year and m.get("week") == week and i < len(wk_list):
            wk_list[i] = round(float(total), 1)
            con.execute("UPDATE predictions SET weekly_km_json=? WHERE id=?",
                        (json.dumps(wk_list), prow["id"]))
            return


class EditRun(BaseModel):
    hr: float | None = None   # null = clear the HR


@app.post("/runs/{run_id}/edit")
def edit_run(run_id: int, data: EditRun, user_id: int | None = Depends(get_user_id)):
    """Edit a run's HR (null clears it)."""
    if user_id is None:
        return {"error": "Log in first."}
    hr = None
    if data.hr is not None:
        if data.hr < 60 or data.hr > 230:
            return {"error": "HR looks off (expected 60–230 bpm)."}
        hr = round(float(data.hr))
    con = db()
    row = con.execute(
        "SELECT user_id FROM runs WHERE id=?", (run_id,)
    ).fetchone()
    if not row or row["user_id"] != user_id:
        con.close()
        return {"error": "Run not found."}
    con.execute("UPDATE runs SET hr=? WHERE id=?", (hr, run_id))
    con.commit()
    con.close()
    return {"ok": True, "hr": hr}


@app.delete("/runs/{run_id}")
def delete_run(run_id: int, user_id: int | None = Depends(get_user_id)):
    """Delete a run; the week's bar syncs to the new sum."""
    if user_id is None:
        return {"error": "Log in first."}
    con = db()
    row = con.execute(
        "SELECT user_id, year, week FROM runs WHERE id=?", (run_id,)
    ).fetchone()
    if not row or row["user_id"] != user_id:
        con.close()
        return {"error": "Run not found."}
    con.execute("DELETE FROM runs WHERE id=?", (run_id,))
    resync_week_total(con, user_id, row["year"], row["week"])
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
        try:
            weekly_km = json.loads(r["weekly_km_json"]) if r["weekly_km_json"] else []
        except Exception:
            weekly_km = []
        out.append({
            "id": r["id"],
            "date": r["created_at"][:10],
            "source": r["source"],
            "predicted_time": fmt(r["predicted_seconds"]),
            "actual_pr_time": fmt(actual) if actual is not None else None,
            "diff_seconds": round(abs(r["predicted_seconds"] - actual)) if actual is not None else None,
            # full inputs so the frontend can reload this prediction into the form
            "gender": r["gender"],
            "typical_pace": r["typical_pace"],
            "avg_weekly_km": r["avg_weekly_km"],
            "active_weeks": r["active_weeks"],
            "longest_km": r["longest_km"],
            "easy_hr": r["easy_hr"],
            "max_hr": r["max_hr"],
            "weekly_km": weekly_km,
        })
    return {"history": out}
