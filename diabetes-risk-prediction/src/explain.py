import joblib
import pandas as pd
import shap

def explain_prediction(input_data: pd.DataFrame):
    pipeline = joblib.load("models/diabetes_pipeline.pkl")
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]
    transformed = preprocessor.transform(input_data)
    names = preprocessor.get_feature_names_out()
    transformed_df = pd.DataFrame(transformed, columns=names)
    explainer = shap.TreeExplainer(estimator)
    explanation = explainer(transformed_df)
    return explanation, transformed_df

if __name__ == "__main__":
    sample = pd.DataFrame([{
        "Pregnancies": 2, "Glucose": 130, "BloodPressure": 70,
        "SkinThickness": 25, "Insulin": 100, "BMI": 32.0,
        "DiabetesPedigreeFunction": 0.5, "Age": 35
    }])
    values, _ = explain_prediction(sample)
    print(values)
