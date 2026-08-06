"""Compute total mileage across the labelled runs actually used for training."""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
PARQUET      = os.path.join(_HERE, "fitrec_runs.parquet")
FEATURES_CSV = os.path.join(_HERE, "..", "features.csv")


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat = p2 - p1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


df = pq.read_table(PARQUET).to_pandas()
print(f"loaded {len(df)} total runs from parquet")

# same label join as training
feats = pd.read_csv(FEATURES_CSV).dropna(subset=["fastest_5k"])[["userId", "fastest_5k"]]
feats["userId"] = pd.to_numeric(feats["userId"], errors="coerce").astype("Int64").dropna().astype("int64")
df = df.merge(feats, on="userId", how="inner")
print(f"after label join: {len(df)} runs from {df['userId'].nunique()} athletes")

total_km = 0.0
run_km = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="distances"):
    lat = np.asarray(row["latitude"],  dtype=np.float64)
    lon = np.asarray(row["longitude"], dtype=np.float64)
    n = min(len(lat), len(lon))
    if n < 2:
        continue
    d = haversine_m(lat[:n-1], lon[:n-1], lat[1:n], lon[1:n]).sum() / 1000.0
    run_km.append(d)
    total_km += d

run_km = np.array(run_km)
print(f"\nTotal runs contributing: {len(run_km)}")
print(f"Total distance: {total_km:.0f} km  =  {total_km * 0.621371:.0f} miles")
print(f"Per-run median: {np.median(run_km):.1f} km  ({np.median(run_km)*0.621371:.1f} mi)")
print(f"Per-run mean:   {run_km.mean():.1f} km  ({run_km.mean()*0.621371:.1f} mi)")
