# Deploying the 5K Predictor backend to Google Cloud Run

The FastAPI backend (`backend/`) goes to **Cloud Run** (scale-to-zero, pay-per-request).
Frontend stays on **Vercel**, database stays on **Supabase Postgres**.
The only runtime secret is `DATABASE_URL` → stored in **Secret Manager**, never in the image or git.

Cloud Build reads `backend/Dockerfile` and `backend/.dockerignore` (already committed).
`features.csv` ships in the image because the model is trained from it at startup; the
local `*.db` files and `*.env` are excluded.

---

## 0. One-time install

Install the Google Cloud SDK (includes `gcloud`):
https://cloud.google.com/sdk/docs/install-sdk#windows  → run the installer, reopen PowerShell.

```powershell
gcloud --version   # confirm it's on PATH
```

## 1. Auth + project

```powershell
gcloud auth login
gcloud projects create fivek-predictor-123 --name="5K Predictor"   # pick a globally-unique id
gcloud config set project fivek-predictor-123
```
Then link a billing account (needed for Cloud Run — free tier still applies):
https://console.cloud.google.com/billing → attach it to the project.

## 2. Enable the APIs

```powershell
gcloud services enable run.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com artifactregistry.googleapis.com
```

## 3. Put the env → Secret Manager (DATABASE_URL)

This reads the value straight out of `backend/backend.env` so you never paste the
secret by hand, writes it to a temp file, uploads it, then deletes the temp file:

```powershell
$dburl = (Get-Content backend\backend.env | Where-Object { $_ -match '^DATABASE_URL' }) -replace '^DATABASE_URL\s*=\s*',''
$dburl.Trim() | Out-File -Encoding ascii -NoNewline .\_dburl.tmp
gcloud secrets create DATABASE_URL --data-file=.\_dburl.tmp
Remove-Item .\_dburl.tmp
```

To rotate it later (new Supabase password), same but `versions add`:
```powershell
# ...write _dburl.tmp as above...
gcloud secrets versions add DATABASE_URL --data-file=.\_dburl.tmp
```

Let Cloud Run's service account read the secret:
```powershell
$proj = gcloud config get-value project
$num  = gcloud projects describe $proj --format="value(projectNumber)"
gcloud secrets add-iam-policy-binding DATABASE_URL --member="serviceAccount:$num-compute@developer.gserviceaccount.com" --role="roles/secretmanager.secretAccessor"
```

## 4. Deploy

Run from the **`backend/` folder** so the build context matches `render.yaml`'s rootDir:

```powershell
cd backend
gcloud run deploy fivek-predictor `
  --source . `
  --region us-central1 `
  --allow-unauthenticated `
  --max-instances 3 `
  --memory 512Mi `
  --set-secrets DATABASE_URL=DATABASE_URL:latest
```

`--max-instances 3` caps cost (no runaway scaling). `--allow-unauthenticated`
is required because the Vercel frontend calls it publicly (CORS is already `*`).
The command prints a **Service URL** like `https://fivek-predictor-xxxx.a.run.app`.

Smoke test:
```powershell
curl https://fivek-predictor-xxxx.a.run.app/docs
```

## 5. Point the frontend at Cloud Run

On Vercel → project → Settings → Environment Variables, set your API base
(whatever the React app reads, e.g. `VITE_API_URL`) to the Cloud Run Service URL,
then redeploy the frontend.

## 6. Budget guard (so a surprise never happens)

Billing → Budgets & alerts → Create budget → $5/month, alert at 50/90/100%.
Combined with `--max-instances 3` and scale-to-zero, idle cost is ~$0.

---

## Optional — the MLOps talking point (Cloud Scheduler retrain)

Cloud Run is stateless, so `features.csv` edits are ephemeral; the source of truth is
Postgres. If you want a scheduled "retrain" story for the resume, add an authenticated
endpoint (e.g. `POST /retrain`) and hit it nightly:

```powershell
gcloud scheduler jobs create http fivek-nightly-retrain `
  --schedule="0 3 * * *" `
  --uri="https://fivek-predictor-xxxx.a.run.app/retrain" `
  --http-method=POST `
  --location=us-central1
```

Only add this once such an endpoint exists — don't claim it otherwise.
