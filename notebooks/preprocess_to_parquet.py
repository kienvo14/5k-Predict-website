"""
One-time scan of the 6.6GB endomondoHR.json -> Parquet.

Keeps every valid run (sport=run, all 5 arrays present, length >= MIN_LEN)
with NO per-user cap. Streams to disk in batches so memory stays bounded.

After this runs once, all future training scripts load from the parquet
file in seconds (no more `eval()` line-by-line).

Output columns:
    userId        int64
    n             int32                (samples in this run)
    heart_rate    list<int16>
    altitude      list<float32>
    latitude      list<float32>
    longitude     list<float32>
    timestamp     list<int64>
"""
import os
import time

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

SRC        = "../endomondoHR.json"
DST        = "fitrec_runs.parquet"
MIN_LEN    = 150
BATCH_ROWS = 2000  # flush to disk every N runs


SCHEMA = pa.schema([
    ("userId",     pa.int64()),
    ("n",          pa.int32()),
    ("heart_rate", pa.list_(pa.int16())),
    ("altitude",   pa.list_(pa.float32())),
    ("latitude",   pa.list_(pa.float32())),
    ("longitude",  pa.list_(pa.float32())),
    ("timestamp",  pa.list_(pa.int64())),
])


def flush(writer, batch):
    if not batch["userId"]:
        return 0
    table = pa.Table.from_pydict(batch, schema=SCHEMA)
    writer.write_table(table)
    n = len(batch["userId"])
    for k in batch:
        batch[k].clear()
    return n


def main():
    src_size = os.path.getsize(SRC)
    print(f"scanning {SRC} ({src_size/1e9:.2f} GB) -> {DST}")

    batch = {k: [] for k in SCHEMA.names}
    kept = 0
    seen_users = set()
    t0 = time.time()

    with open(SRC, "rt") as f_in, \
         pq.ParquetWriter(DST, SCHEMA, compression="snappy") as writer:

        pbar = tqdm(total=src_size, unit="B", unit_scale=True, desc="scan", mininterval=1.0)
        for line in f_in:
            pbar.update(len(line))
            try:
                d = eval(line)
            except Exception:
                continue
            if d.get("sport") != "run":
                continue

            hr  = d.get("heart_rate") or []
            alt = d.get("altitude")   or []
            lat = d.get("latitude")   or []
            lon = d.get("longitude")  or []
            ts  = d.get("timestamp")  or []
            n = min(len(hr), len(alt), len(lat), len(lon), len(ts))
            if n < MIN_LEN:
                continue

            uid = d.get("userId")
            if uid is None:
                continue

            batch["userId"].append(int(uid))
            batch["n"].append(int(n))
            # trim to common length, coerce dtypes cheaply
            batch["heart_rate"].append([int(v) for v in hr[:n]])
            batch["altitude"].append([float(v) for v in alt[:n]])
            batch["latitude"].append([float(v) for v in lat[:n]])
            batch["longitude"].append([float(v) for v in lon[:n]])
            batch["timestamp"].append([int(v) for v in ts[:n]])
            seen_users.add(uid)

            if len(batch["userId"]) >= BATCH_ROWS:
                kept += flush(writer, batch)

        kept += flush(writer, batch)
        pbar.close()

    dt = time.time() - t0
    out_size = os.path.getsize(DST)
    print(f"\nkept {kept} runs from {len(seen_users)} unique users")
    print(f"output: {DST}  ({out_size/1e6:.1f} MB, {src_size/out_size:.0f}x smaller)")
    print(f"elapsed: {dt/60:.1f} min")


if __name__ == "__main__":
    main()
