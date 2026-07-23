import csv
import itertools
import os
from datetime import datetime, timezone
from math import radians, sin, cos, asin, sqrt

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat, dlon = radians(lat2-lat1), radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*R*asin(sqrt(a))

file_name = "preprocess_data.csv"
header = ["userId","gender","start_ts","year","week","distance_m",
          "duration_s","pace_min_km","avg_hr","max_hr","elev_gain_m"]

if not os.path.exists(file_name):
        with open(file_name, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

limit = None 
batch_size = 5000 
buffer, kept, skipped = [], 0, 0

def flush():
    if buffer:
        with open(file_name, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(buffer)
        buffer.clear()

with open(file_name, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(header)

with open('endomondoHR.json') as f:
    lines = itertools.islice(f, limit) if limit else f
    for line in lines:
        #take the data from there
        try:
            d = eval(line)
        except Exception:
            continue
        
        if d['sport'] != 'run':
            continue
        
        lat, lon, ts = d.get('latitude'), d.get('longitude'), d.get('timestamp')
        hr, alt = d.get('heart_rate'), d.get('altitude')
        if not lat or not ts or not hr or len(lat) < 2:
            continue

        distance = sum(haversine(lat[i], lon[i], lat[i+1], lon[i+1])
                   for i in range(len(lat)-1))
        duration = ts[-1] - ts[0]
        if distance <= 0 or duration <= 0:                 # guard divide-by-zero
            skipped += 1; continue

        km, mins = distance/1000, duration/60
        elev = sum(max(0, alt[i+1]-alt[i]) for i in range(len(alt)-1)) if alt else 0
        year, week, _ = datetime.fromtimestamp(ts[0], tz=timezone.utc).isocalendar()

        buffer.append([d['userId'], d.get('gender'), ts[0], year, week,
                       round(distance,1), duration, round(mins/km,3),
                       round(sum(hr)/len(hr),1), max(hr), round(elev,1)])
        kept += 1

        if len(buffer) >= batch_size:                  # == → len() check
            flush()
            print(f"wrote {kept} runs...")

flush()
print(f"DONE — kept {kept}, skipped {skipped} → {file_name}")
