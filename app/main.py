import streamlit as st
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(
    page_title="CardioLens AI",
    page_icon="🫀",
    layout="wide"
)

API_URL = "http://localhost:8000/predict"

# -----------------------
# FEATURE ORDER (CRITICAL)
# -----------------------
FEATURES = [
    "age",
    "anaemia",
    "creatinine_phosphokinase",
    "diabetes",
    "ejection_fraction",
    "high_blood_pressure",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "sex",
    "smoking",
    "time"
]

# -----------------------
# UI HEADER
# -----------------------
st.title("🫀 CardioLens AI")
st.caption("Clinical Risk Prediction + Explainable AI System")

st.markdown("---")

# -----------------------
# INPUT UI
# -----------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Profile")
    age = st.slider("Age", 40, 95, 65)
    anaemia = st.selectbox("Anaemia", [0, 1])
    diabetes = st.selectbox("Diabetes", [0, 1])
    high_bp = st.selectbox("High Blood Pressure", [0, 1])
    sex = st.selectbox("Sex (1=Male)", [0, 1])
    smoking = st.selectbox("Smoking", [0, 1])

with col2:
    st.subheader("Clinical Metrics")
    cpk = st.slider("CPK", 20, 8000, 250)
    ef = st.slider("Ejection Fraction", 10, 80, 35)
    platelets = st.slider("Platelets", 10000, 400000, 250000)
    creatinine = st.slider("Serum Creatinine", 0.5, 10.0, 1.2)
    sodium = st.slider("Serum Sodium", 110, 150, 137)
    time = st.slider("Follow-up Days", 0, 300, 100)

# -----------------------
# BUILD INPUT
# -----------------------
input_dict = {
    "age": age,
    "anaemia": anaemia,
    "creatinine_phosphokinase": cpk,
    "diabetes": diabetes,
    "ejection_fraction": ef,
    "high_blood_pressure": high_bp,
    "platelets": platelets,
    "serum_creatinine": creatinine,
    "serum_sodium": sodium,
    "sex": sex,
    "smoking": smoking,
    "time": time
}

# Ensure correct order ALWAYS
input_df = pd.DataFrame([[input_dict[f] for f in FEATURES]], columns=FEATURES)

# -----------------------
# CALL API
# -----------------------
if st.button("Run Risk Analysis"):

    try:
        response = requests.post(API_URL, json=input_dict)
        res = response.json()

        pred = res["prediction"]
        proba = res["probability"]

        shap_vals = np.array(res.get("shap", [0]*len(FEATURES)))

        # -----------------------
        # SAFE SHAP NORMALIZATION
        # -----------------------
        shap_vals = shap_vals.flatten()

        if len(shap_vals) > len(FEATURES):
            shap_vals = shap_vals[:len(FEATURES)]
        elif len(shap_vals) < len(FEATURES):
            shap_vals = np.pad(shap_vals, (0, len(FEATURES) - len(shap_vals)))

        # -----------------------
        # RESULT UI
        # -----------------------
        st.markdown("## 🧠 Prediction Result")

        if pred == 1:
            st.error("🔴 High Risk of Mortality")
        else:
            st.success("🟢 Low Risk Patient")

        st.metric("Mortality Probability", f"{proba['mortality']:.2%}")

        # -----------------------
        # PROBABILITY CHART
        # -----------------------
        fig = go.Figure(data=[
            go.Bar(
                x=["Survival", "Mortality"],
                y=[proba["survival"], proba["mortality"]],
                marker_color=["#22c55e", "#ef4444"]
            )
        ])

        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # -----------------------
        # SHAP TABLE
        # -----------------------
        st.markdown("## 🧠 Explainability (Feature Impact)")

        shap_df = pd.DataFrame({
            "feature": FEATURES,
            "impact": shap_vals
        }).sort_values("impact", key=abs, ascending=False)

        for _, row in shap_df.iterrows():
            direction = "↑ increases risk" if row["impact"] > 0 else "↓ reduces risk"
            st.write(f"**{row['feature']}** → {row['impact']:.3f} {direction}")

        # -----------------------
        # SHAP VISUALIZATION
        # -----------------------
        fig2 = go.Figure()

        fig2.add_trace(go.Bar(
            x=shap_df["impact"],
            y=shap_df["feature"],
            orientation="h",
            marker_color=[
                "#ef4444" if x > 0 else "#22c55e"
                for x in shap_df["impact"]
            ]
        ))

        fig2.update_layout(template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

        # -----------------------
        # CLINICAL SUMMARY
        # -----------------------
        top = shap_df.iloc[0]

        st.info(f"""
**Most influential factor:** {top['feature']}

This feature has the strongest contribution to the model's decision.

Positive impact → increases risk  
Negative impact → decreases risk
""")

    except Exception as e:
        st.error(f"Error calling API: {str(e)}")