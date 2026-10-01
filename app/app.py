from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diabetes Risk Prediction", page_icon="🩺", layout="wide")

MODEL_PATH = Path("models/diabetes_pipeline.pkl")
METRICS_PATH = Path("models/metrics.json")

if not MODEL_PATH.exists():
    import subprocess
    import sys
    
    with st.spinner("Training model for the first time... this may take a moment."):
        try:
            subprocess.run([sys.executable, "-m", "src.make_demo_data"], check=True)
            subprocess.run([sys.executable, "-m", "src.train"], check=True)
        except subprocess.CalledProcessError as e:
            st.error(f"Failed to generate model: {e}")
            st.stop()


model = joblib.load(MODEL_PATH)
metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}

if "history" not in st.session_state:
    st.session_state.history = []

st.title("🩺 Diabetes Risk Prediction")
st.caption("End-to-end ML classification + explainability demo")

st.warning("Educational machine-learning project only. This output is not a medical diagnosis and should not be used for medical decisions.")

with st.sidebar:
    st.header("Project")
    st.write("Preprocessing → model comparison → CV tuning → evaluation → explainability → deployment")
    if metrics:
        st.subheader("Validation snapshot")
        for name, vals in metrics.items():
            st.write(f"**{name}** — ROC-AUC: {vals['roc_auc']:.3f}")

st.subheader("Input features")
with st.form("prediction_form"):
    c1, c2 = st.columns(2)
    with c1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose", 1.0, 300.0, 120.0)
        blood_pressure = st.number_input("Blood Pressure", 1.0, 200.0, 70.0)
        skin_thickness = st.number_input("Skin Thickness", 1.0, 100.0, 20.0)
    with c2:
        insulin = st.number_input("Insulin", 1.0, 1000.0, 80.0)
        bmi = st.number_input("BMI", 1.0, 80.0, 25.0)
        pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        age = st.number_input("Age", 1, 120, 30)
    submitted = st.form_submit_button("Predict Risk")

if submitted:
    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age
    }])

    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0, 1])

    st.divider()
    st.subheader("Prediction")

    if prediction == 1:
        st.warning("The model predicts the positive class.")
    else:
        st.success("The model predicts the negative class.")

    st.metric("Model probability for positive class", f"{probability * 100:.2f}%")
    st.progress(min(max(probability, 0.0), 1.0))

    st.session_state.history.append({
        "Glucose": glucose, "BMI": bmi, "Age": age,
        "Prediction": prediction,
        "Probability (%)": round(probability * 100, 2)
    })

    st.subheader("Local explanation")
    try:
        import shap
        import matplotlib.pyplot as plt
        preprocessor = model.named_steps["preprocessor"]
        estimator = model.named_steps["model"]
        transformed = preprocessor.transform(input_data)
        names = preprocessor.get_feature_names_out()
        transformed_df = pd.DataFrame(transformed, columns=names)
        explainer = shap.TreeExplainer(estimator)
        explanation = explainer(transformed_df)
        fig = plt.figure()
        shap.plots.waterfall(explanation[0,:,1], show=False)
        st.pyplot(fig, clear_figure=True)
        st.caption("SHAP shows how model features contributed to this individual prediction. It does not establish causation.")
    except Exception as exc:
        st.info(f"Explanation unavailable for this configuration: {exc}")

if st.session_state.history:
    st.divider()
    st.subheader("Prediction history")
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
