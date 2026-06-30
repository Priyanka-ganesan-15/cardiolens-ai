from fastapi import FastAPI
from ml.model import predict
from ml.explainer import explain

app = FastAPI(title="CardioLens AI API")

@app.post("/predict")
def predict_risk(payload: dict):
    pred, proba, df = predict(payload)
    shap_values = explain(df)

    return {
        "prediction": int(pred),
        "probability": {
            "survival": float(proba[0]),
            "mortality": float(proba[1])
        },
        "shap": shap_values.tolist(),
        "features": df.columns.tolist()
    }