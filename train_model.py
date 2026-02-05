import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
df = pd.read_csv("rainfall_india.csv")   # <-- adjust name if needed

# Remove rows with missing values
df = df.dropna()

# Features (months)
X = df[['JAN','FEB','MAR','APR','MAY','JUN',
        'JUL','AUG','SEP','OCT','NOV','DEC']]

# Target
y = df['ANNUAL']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("MODEL TRAINED SUCCESSFULLY")