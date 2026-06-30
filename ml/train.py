import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from ml.feature import FEATURES

df = pd.read_csv("data/heart_failure.csv")

X = df[FEATURES]
y = df["DEATH_EVENT"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

joblib.dump(model, "models/model.pkl")
joblib.dump(FEATURES, "models/features.pkl")

print("✅ Model trained + saved")