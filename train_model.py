import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump
import os

# Create folders if not present
os.makedirs("model", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Simulate 300 samples
np.random.seed(42)
df = pd.DataFrame({
    'alpha_wave_ratio': np.random.normal(0.4, 0.1, 300),
    'theta_wave_ratio': np.random.normal(0.3, 0.1, 300),
    'step_variance': np.random.normal(0.5, 0.2, 300),
    'blink_rate': np.random.normal(15, 5, 300),
    'heart_rate': np.random.normal(75, 10, 300),
    'temperature': np.random.normal(36.5, 0.4, 300),
    'abeta_level': np.random.normal(5.0, 1.0, 300)
})

# Simple rule-based risk logic
df['risk_score'] = (
    (df['alpha_wave_ratio'] < 0.35).astype(int) +
    (df['step_variance'] > 0.7).astype(int) +
    (df['abeta_level'] < 4.5).astype(int)
)
df['risk_level'] = df['risk_score'].apply(lambda x: 0 if x <= 1 else 1 if x == 2 else 2)
df.drop(columns='risk_score', inplace=True)

# Save data
df.to_csv("data/log.csv", index=False)

# Train model
X = df.drop("risk_level", axis=1)
y = df["risk_level"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
dump(model, "model/alz_model.joblib")
print("✅ Model trained and saved as model/alz_model.joblib")
