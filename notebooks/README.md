# HR Imputation Notebook (Path A)

Standalone deep-learning experiment — **completely separate from the app**.
Doesn't touch `app.db`, `features.csv`, or the FastAPI backend.

## What it does
Trains a **bidirectional LSTM (PyTorch)** to fill missing heart-rate segments in a run,
using the surrounding pace + elevation signals as context.

- **Data:** FitRec / Endomondo raw file (`endomondoHR.json`, ~6.6GB, gitignored) —
  cite: Ni, Muhlstein, McAuley — *"Modeling heart rate and activity data for
  personalized fitness recommendation."* WWW 2019.
- **Model:** 2-layer BiLSTM (hidden=128), input = [pace, altitude, HR-with-gap, mask-flag]
- **Loss:** MSE on the gap timesteps only
- **Baseline:** linear interpolation between the gap edges (must beat this)

## How to run

### Option A — Google Colab (recommended, free GPU)
1. Open Colab → **New notebook**
2. Runtime → Change runtime type → **T4 GPU**
3. Upload `hr_imputation.py`  (or paste cells one-by-one)
4. Upload a subset of the raw FitRec file to Colab (the whole 6.6GB is too big;
   see below for making a smaller copy)
5. Change `FITREC_PATH` to `"endomondoHR.json"` in the config cell
6. Run all cells

### Option B — Local (CPU only, slow)
```bash
cd notebooks
pip install -r requirements.txt
py hr_imputation.py
```
Set `MAX_RUNS = 500` for a first run to test the pipeline on CPU.

## Making a smaller FitRec subset for Colab
The full raw file is 6.6GB. To upload just the first 10k runs:

```bash
py -c "
import gzip
n = 0
with open('endomondoHR.json') as f, open('endomondoHR_subset.json', 'w') as out:
    for line in f:
        out.write(line)
        n += 1
        if n >= 30000: break
print(f'wrote {n} lines')
"
```

## What gets saved
```
notebooks/out/
  hr_lstm.pt         trained weights + normalization stats
  example_fill.png   a sample gap being filled
```

## Expected results (starting point)
On a 5k-run subset, ~6 epochs, T4 GPU (~10 min):
- Linear-interp baseline: ~9–12 bpm RMSE
- BiLSTM model:            ~5–8 bpm RMSE
- Improvement:             ~3–5 bpm

Bump `MAX_RUNS = 30_000` + `EPOCHS = 15` for the real number to quote on the resume.

## Resume line (once you have real numbers)
> *"Trained a bidirectional LSTM (PyTorch) on FitRec/Endomondo data to impute
> missing heart-rate segments in GPS run data, achieving X bpm RMSE on 5-min gaps
> (vs Y bpm linear-interp baseline). Enables correct effort/zone recalculation
> when sensors drop out. Cite: Ni, Muhlstein, McAuley — WWW 2019."*
