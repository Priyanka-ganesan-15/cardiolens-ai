import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

X = df.drop("DEATH_EVENT", axis=1)
y = df["DEATH_EVENT"]

feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

print("\n📊 Classification Report:\n")
print(classification_report(y_test, preds))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
joblib.dump(feature_names, "models/features.pkl")

print("\n✅ Model + features saved")