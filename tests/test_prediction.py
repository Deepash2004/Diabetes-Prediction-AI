import os
import sys
import joblib
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def sample():
    return pd.DataFrame([{
        "Pregnancies": 2, "Glucose": 130, "BloodPressure": 70,
        "SkinThickness": 25, "Insulin": 100, "BMI": 32.0,
        "DiabetesPedigreeFunction": 0.5, "Age": 35
    }])

def test_model_exists():
    assert os.path.exists("models/diabetes_pipeline.pkl")

def test_prediction_shape():
    model = joblib.load("models/diabetes_pipeline.pkl")
    prediction = model.predict(sample())
    assert len(prediction) == 1

def test_prediction_is_binary():
    model = joblib.load("models/diabetes_pipeline.pkl")
    prediction = int(model.predict(sample())[0])
    assert prediction in [0, 1]
