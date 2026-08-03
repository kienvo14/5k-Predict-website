# 5K Time Predictor

A full-stack web app that predicts a runner's current 5K time within **±80 seconds** (cross-validated MAE) from their recent training — pace, mileage, heart rate, and consistency.

**Live:** [5k-predict-website.vercel.app](https://5k-predict-website.vercel.app) &nbsp;·&nbsp; **Backend:** [fivek-backend.onrender.com](https://fivek-backend.onrender.com)

---

## Data & attribution

This project's ML model is trained entirely on **public research datasets**:
- **FitRec / Endomondo dataset** — Jianmo Ni, Larry Muhlstein, Julian McAuley (UCSD).
  *"Modeling heart rate and activity data for personalized fitness recommendation."* **WWW 2019.**
  Dataset: [cseweb.ucsd.edu/~jmcauley/datasets/fitrec.html](https://cseweb.ucsd.edu/~jmcauley/datasets/fitrec.html)
- **Kaggle "Running races from Strava" dataset** — Oleg Oaer.
  [kaggle.com/datasets/olegoaer/running-races-strava](https://www.kaggle.com/datasets/olegoaer/running-races-strava)

Raw data files are **not** included in this repo (too large and licensed). Only the processed per-athlete feature set (`backend/features.csv`, 776 rows) is checked in.

---

## Stack
- **Frontend:** React + TypeScript (Vite), React Router, deployed on **Vercel**
- **Backend:** FastAPI (Python), deployed on **Render**
- **ML:** scikit-learn (linear regression), Pandas, NumPy
- **Database:** PostgreSQL on **Supabase** (SQLite for local dev)
- **Auth:** salted + hashed passwords (PBKDF2), session tokens

## Features
- Predict from **manually typed** training data OR **uploaded Strava export** (.csv / .xlsx)
- **Accounts** — sign up, log in, prediction history persists across sessions
- **Progress chart** — Strava-style connected-line chart of weekly mileage (last 16 weeks); click any week to see its runs
- **Editable weeks** — add / edit / delete runs directly on the chart
- **Feedback loop** — enter your real 5K PR to see how close the model was; verified PRs become new training data (retrains after 10 accumulate)

## Results
- **Cross-validated MAE:** 82 seconds (~1:22)
- Linear regression outperformed gradient boosting — pace/5K relationship is near-linear
- Beat Riegel-formula baseline

## Data pipeline
Raw → cleaned → training set (see [Data & attribution](#data--attribution) above for sources):
- **295,000+ raw workouts** ingested from the public datasets
- **117,000+ clean runs** (734,000+ miles) after filtering (dedupe, HR/pace sanity bounds, min-duration, etc.)
- **776 per-athlete feature rows** aggregated for training

## Engineered features
weekly mileage (avg, std) · typical pace (median) · easy-run HR · max HR · longest run · aerobic efficiency (pace/HR) · training consistency · active weeks

## Run locally

**Backend** (from `/backend`):
```bash
pip install -r requirements.txt
uvicorn main:app --port 8000 --reload
```

**Frontend** (from `/frontend`):
```bash
npm install
npm run dev
```
Open http://localhost:5173

**Optional:** set `DATABASE_URL` in `backend/.env` to test against Postgres locally. Otherwise falls back to SQLite (`app.db`) automatically.

## Honest caveats
- **Labels are Riegel estimates** derived from training runs, not actual race times. Therfore, it only predicts from the estimated performance not real PR race.
- Verified PR from the user are the only time real correct PR can be used.
