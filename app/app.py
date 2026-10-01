import io
import os
import json
import base64
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import shap

app = Flask(__name__)

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "diabetes_pipeline.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"

# Load model and metrics globally
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
else:
    model = None
    print(f"Model not found at {MODEL_PATH}. Please run training first.")

if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text())
else:
    metrics = {}

@app.route("/")
def index():
    return render_template("index.html", metrics=metrics)

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Model not trained yet."}), 500

    data = request.json
    
    # Create DataFrame
    input_data = pd.DataFrame([{
        "Pregnancies": float(data.get("Pregnancies", 0)),
        "Glucose": float(data.get("Glucose", 120)),
        "BloodPressure": float(data.get("BloodPressure", 70)),
        "SkinThickness": float(data.get("SkinThickness", 20)),
        "Insulin": float(data.get("Insulin", 80)),
        "BMI": float(data.get("BMI", 25)),
        "DiabetesPedigreeFunction": float(data.get("DiabetesPedigreeFunction", 0.5)),
        "Age": float(data.get("Age", 30))
    }])

    # Predict
    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0, 1])

    # Generate SHAP explanation
    shap_base64 = None
    try:
        preprocessor = model.named_steps["preprocessor"]
        estimator = model.named_steps["model"]
        transformed = preprocessor.transform(input_data)
        names = preprocessor.get_feature_names_out()
        transformed_df = pd.DataFrame(transformed, columns=names)
        
        explainer = shap.TreeExplainer(estimator)
        explanation = explainer(transformed_df)
        
        fig = plt.figure(figsize=(8, 4))
        shap.plots.waterfall(explanation[0, :, 1], show=False)
        
        # Save plot to base64 string
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", bbox_inches='tight', dpi=100)
        plt.close(fig)
        buf.seek(0)
        shap_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as exc:
        print(f"SHAP Error: {exc}")

    return jsonify({
        "prediction": prediction,
        "probability": probability,
        "shap_image": shap_base64
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
