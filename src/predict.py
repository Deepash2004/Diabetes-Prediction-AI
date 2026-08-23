import joblib
import pandas as pd

MODEL_PATH = "models/diabetes_pipeline.pkl"

def predict_diabetes(input_data: dict):
    model = joblib.load(MODEL_PATH)
    frame = pd.DataFrame([input_data])
    prediction = int(model.predict(frame)[0])
    probability = float(model.predict_proba(frame)[0, 1])
    return {"prediction": prediction, "probability": probability}

if __name__ == "__main__":
    sample = {
        "Pregnancies": 2, "Glucose": 130, "BloodPressure": 70,
        "SkinThickness": 25, "Insulin": 100, "BMI": 32.0,
        "DiabetesPedigreeFunction": 0.5, "Age": 35
    }
    print(predict_diabetes(sample))
