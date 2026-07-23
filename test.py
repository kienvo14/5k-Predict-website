import itertools
from math import radians, sin, cos, asin, sqrt

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat, dlon = radians(lat2-lat1), radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*R*asin(sqrt(a))

n = 0
with open('endomondoHR.json') as f:
    for line in f:
        d = eval(line)
        if d['sport'] != 'run':
            continue
        lat, lon, ts = d['latitude'], d['longitude'], d['timestamp']
        dist = sum(haversine(lat[i], lon[i], lat[i+1], lon[i+1])
                   for i in range(len(lat)-1))
        dur = ts[-1] - ts[0]
        km, mins = dist/1000, dur/60
        pace = mins/km if km > 0 else 0
        print(f"n={len(lat)}  {km:5.2f}km  {mins:5.1f}min  pace={pace:4.2f}min/km  hr={sum(d['heart_rate'])/len(d['heart_rate']):.0f}")
        n += 1
        if n >= 15:
            break