import joblib
import pandas as pd

model = joblib.load("models/model.pkl")
FEATURES = joblib.load("models/features.pkl")


def predict(input_dict: dict):
    df = pd.DataFrame([input_dict])[FEATURES]
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0]
    return pred, proba, df