import pickle
import pandas as pd
from xgboost import XGBRegressor


df = pd.read_csv('beer_reviews_deployment.csv')

X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]  # average overall rating

model = XGBRegressor(max_depth=10, max_features='auto',
                              min_samples_leaf=1, min_samples_split=2,
                              reg_alpha=10, reg_lambda=10, random_state=10)
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
