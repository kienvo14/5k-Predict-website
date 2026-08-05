"""
One-time preprocessor: scans the 6.6GB FitRec file ONCE, keeps up to
MAX_PER_USER runs per user, and writes them to a compact gzipped file.

After running this once (~2-3 min), the training script loads instantly.

Usage:
    py preprocess_fitrec.py
    # produces: fitrec_diverse.json.gz  (~30-100MB)
"""
import gzip
import os
from tqdm import tqdm

SRC          = "../endomondoHR.json"       # your raw file
DST          = "fitrec_diverse.json.gz"    # compact output
MAX_PER_USER = 200                          # loosened — most users have <20 runs anyway
MIN_LEN      = 150                          # slightly shorter runs still useful

# Only keep the fields we need for HR imputation — massive size reduction
KEEP_FIELDS = ("userId", "sport", "heart_rate", "altitude",
               "latitude", "longitude", "timestamp")


def main():
    per_user = {}
    kept = 0
    src_size = os.path.getsize(SRC)
    print(f"scanning {SRC} ({src_size/1e9:.1f} GB) ...")

    with open(SRC, "rt") as f_in, gzip.open(DST, "wt") as f_out:
        pbar = tqdm(total=src_size, unit="B", unit_scale=True, desc="scan")
        for line in f_in:
            pbar.update(len(line))
            try:
                d = eval(line)
            except Exception:
                continue
            if d.get("sport") != "run":
                continue
            uid = d.get("userId")
            if per_user.get(uid, 0) >= MAX_PER_USER:
                continue
            hr = d.get("heart_rate") or []
            alt = d.get("altitude") or []
            lat = d.get("latitude") or []
            lon = d.get("longitude") or []
            ts = d.get("timestamp") or []
            n = min(len(hr), len(alt), len(lat), len(lon), len(ts))
            if n < MIN_LEN:
                continue
            slim = {k: d.get(k) for k in KEEP_FIELDS if k in d}
            # trim arrays to the common length to shrink the output
            for k in ("heart_rate", "altitude", "latitude", "longitude", "timestamp"):
                if k in slim and isinstance(slim[k], list):
                    slim[k] = slim[k][:n]
            f_out.write(repr(slim) + "\n")
            per_user[uid] = per_user.get(uid, 0) + 1
            kept += 1
        pbar.close()

    out_size = os.path.getsize(DST)
    print(f"\nkept {kept} runs from {len(per_user)} unique users")
    print(f"output: {DST}  ({out_size/1e6:.1f} MB)")
    print(f"compression: {src_size/out_size:.0f}x smaller")


if __name__ == "__main__":
    main()
