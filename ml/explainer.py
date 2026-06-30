import shap
import joblib
import numpy as np

model = joblib.load("models/model.pkl")

explainer = shap.TreeExplainer(model)


def explain(df):
    shap_values = explainer.shap_values(df)

    if isinstance(shap_values, list):
        values = shap_values[1][0]
    else:
        values = shap_values[0]

    values = np.array(values).flatten()
    return values