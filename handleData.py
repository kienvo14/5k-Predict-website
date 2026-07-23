import pandas as pd 

df = pd.read_csv('preprocess_data.csv')

#second file
df2 = pd.read_csv('raw-data-kaggle.csv', sep=';')
df2['dt'] = pd.to_datetime(df2['timestamp'], format='%d/%m/%Y %H:%M')
iso = df2['dt'].dt.isocalendar()
df2['year'], df2['week'] = iso.year, iso.week
df2['start_ts'] = df2['dt'].astype('int64') // 10**9    
df2 = df2.rename(columns={'athlete':'userId','distance (m)':'distance_m',
                          'elapsed time (s)':'duration_s',
                          'average heart rate (bpm)':'avg_hr'})
df2['pace_min_km'] = (df2['duration_s']/60) / (df2['distance_m']/1000)
df2['max_hr'] = df2['avg_hr']
df2['gender'] = df2['gender'].map({'M':'male','F':'female'})   
df2['userId'] = 'k' + df2['userId'].astype(str)     

# stack — pick the shared columns so both align
cols = ['userId','gender','year','week','start_ts',
        'distance_m','duration_s','pace_min_km','avg_hr','max_hr']
df = pd.concat([df[cols], df2[cols]], ignore_index=True)

#print("before:", df.shape)
#print("before:", df.describe())

df = df[(2000 <= df['distance_m']) & ((df['distance_m']) <= 100000)] #run distance between 2 and 100km
df = df[(600 <= df['duration_s']) & ((df['duration_s']) <= 4*3600)] #between 10 min and 4 hour 
df = df[(3.0 <= df['pace_min_km']) & ((df['pace_min_km']) <= 12.0)] #reasonable pace not cycling pace 
df = df[(80 <= df['avg_hr']) & ((df['avg_hr']) <= 200)]
df = df[(80 <= df['max_hr']) & ((df['max_hr']) <= 200)]
df = df[(2008 <= df['year']) & ((df['year']) <= 2026)]

df = df.drop_duplicates(['userId', 'start_ts']) #drop duplicate start time

#keep only users with >= 10 clean runs
counts = df['userId'].value_counts()
keep = counts[counts >= 10].index
df = df[df['userId'].isin(keep)]

#can use transform  which maybe faster df['pace_75th'] = df.groupby('userId')['pace_min_km'].transform(lambda x: x.quantile(0.75))
weekly_totals = (
        df.groupby(['userId', 'year', 'week'])['distance_m']
        .sum()
        .reset_index()
    )

#Step 2: Calculate average weekly mileage and total active weeks per user
avg_weekly_mileage = (
        weekly_totals.groupby('userId')['distance_m']
        .agg(
            avg_weekly_distance_m='mean', #Sum of all weekly mileages / total count of active weeks
            active_weeks='count',          #Total unique weeks logged by user
        )
        .reset_index()
        #mean is the shrotcut to divide some of weekly total which will be on different users to the mileage 
    )

#take the standard deviation 
consistency =  (
        weekly_totals.groupby('userId')['distance_m']
        .agg(consistency_std='std').reset_index()
    )

df['pace_75'] = df.groupby('userId')['pace_min_km'].transform(lambda x: x.quantile(0.75))
#easy = df[df['pace_min_km'] >= df['pace_75']]

easy = df[(df['pace_min_km'] >= df.groupby('userId')['pace_min_km'].transform(lambda x: x.quantile(0.4))) &
          (df['pace_min_km'] <= df.groupby('userId')['pace_min_km'].transform(lambda x: x.quantile(0.6)))]

easy_stats = easy.groupby('userId').agg(
    easy_pace = ('pace_min_km', 'mean'), 
    easy_hr = ('avg_hr', 'mean')
).reset_index()

easy_stats['aerobic'] = easy_stats['easy_pace'] / easy_stats['easy_hr']

# only extrapolate from runs 3–15km — Riegel is unreliable outside that
label_runs = df[(df['distance_m'] >= 3000) & (df['distance_m'] <= 15000)].copy()
label_runs['fastest_5k'] = label_runs['duration_s'] * (5000 / label_runs['distance_m']) ** 1.06

# robust: 5th percentile instead of raw min, so one glitch can't set the label
label = (label_runs.groupby('userId')['fastest_5k']
         .quantile(0.05)
         .reset_index(name='fastest_5k'))

#agg collapse multiple data into one single row 
base = df.groupby('userId').agg(
    longest_run = ('distance_m', 'max'),
    gender = ('gender', 'first'),
    max_hr = ('max_hr', 'max'),
)

features = (
    base
    .merge(easy_stats, on='userId')
    .merge(avg_weekly_mileage,  on='userId')
    .merge(consistency,         on='userId')
    .merge(label,               on='userId')
)
features['fastest_5k_str'] = pd.to_datetime(features['fastest_5k'], unit='s').dt.strftime('%M:%S')



print(features.sort_values('fastest_5k').head(10))
total_m = (features['avg_weekly_distance_m'] * features['active_weeks']).sum()
print(total_m / 1600, "mile")
print(features.shape)

features.to_csv('features.csv', index=False)  