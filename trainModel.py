import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('features.csv')

#Clean more data and pick X(features) and Y(predict)
df['gender'] = df['gender'].map({'male': 0, 'female': 1})

#pick X and Y 
feature_cols = ['longest_run','max_hr','easy_pace','easy_hr','aerobic',
                'avg_weekly_distance_m','active_weeks','consistency_std','gender']
X = df[feature_cols]
y = df['fastest_5k']

mask = X.notna().all(axis=1) & y.notna() #remove the NaN value 
X, y = X[mask], y[mask]


print("training on", len(X), "athletes")

#Split 80/20 X = questions Y = answer 
# X_train + Y_train = 80% of the answer 
#X_test and Y_test is the last 20% 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#BASELINE — just predict the average 5k time
baseline = [y_train.mean()] * len(y_test)
print("Baseline MAE:", round(mean_absolute_error(y_test, baseline), 1), "sec")

#Predict using different model 
#linear 
lin = LinearRegression().fit(X_train, y_train)
print("Linear MAE:", round(mean_absolute_error(y_test, lin.predict(X_test)), 1), "sec")

#Gradient boosting
gb = GradientBoostingRegressor(random_state=42).fit(X_train, y_train)
print("GBoost MAE:", round(mean_absolute_error(y_test, gb.predict(X_test)), 1), "sec")

# --- diagnostic 1: is the MAE stable, or a lucky split? ---
scores = cross_val_score(gb, X, y, cv=5, scoring='neg_mean_absolute_error')
print("CV MAE:", round(-scores.mean(), 1), "sec  (±", round(scores.std(), 1), ")")

# --- diagnostic 2: the 10 worst predictions (likely bad labels) ---
preds = gb.predict(X_test)
err = np.abs(preds - y_test)
worst = X_test.assign(true=y_test.round(0), pred=preds.round(0), err=err.round(0))

print(worst.sort_values('err').tail(10))

imp = pd.Series(gb.feature_importances_, index=X.columns).sort_values(ascending=False)
print(imp)