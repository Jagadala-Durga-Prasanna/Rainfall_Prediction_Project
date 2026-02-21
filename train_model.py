import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

df = pd.read_csv("rainfall_india.csv")
df = df.dropna()

X = df[['JF', 'MAM', 'JJAS', 'OND']]

# Create balanced target
df['Rain_Chance'] = df['ANNUAL'].apply(lambda x: 1 if x > 800 else 0)

y = df['Rain_Chance']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

# Save both model and scaler
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model trained successfully")
