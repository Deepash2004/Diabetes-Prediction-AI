# Diabetes Risk Prediction & Explainable ML System

An end-to-end machine-learning classification project demonstrating data preprocessing, exploratory analysis, model comparison, cross-validation, hyperparameter tuning, evaluation, explainability, testing, and Streamlit deployment.

> **Important:** This is an educational ML project. It is not a medical diagnostic system.

## Features

- Data validation and cleaning
- Median imputation and feature scaling
- Stratified train/test split
- Logistic Regression, Random Forest and Gradient Boosting comparison
- Stratified 5-fold cross-validation
- Randomized hyperparameter optimization
- Accuracy, precision, recall, F1 and ROC-AUC
- Confusion matrix and ROC curve
- SHAP local explainability
- Persisted sklearn pipeline
- Streamlit interface
- Prediction history
- Pytest checks
- GitHub-ready structure

## Dataset

The app expects `data/diabetes.csv` with the common Pima-style columns:

`Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`

A synthetic demo generator is included solely to verify the software pipeline. Replace demo data with a documented public dataset before reporting final model metrics on a resume.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Quick test without a CSV

```bash
python -m src.make_demo_data
python -m src.train
python -m src.evaluate
pytest
streamlit run app/app.py
```

## With your own dataset

Place it at:

`data/diabetes.csv`

Then run:

```bash
python -m src.train
python -m src.evaluate
pytest
streamlit run app/app.py
```

## Suggested resume description

**Explainable Diabetes Risk Prediction System | Python, Scikit-learn, XGBoost, SHAP, Streamlit**

- Built an end-to-end ML classification pipeline with data preprocessing, model comparison, stratified cross-validation and hyperparameter optimization.
- Evaluated classifiers using precision, recall, F1-score and ROC-AUC and implemented SHAP-based local model explanations.
- Deployed an interactive Streamlit application with validation, prediction history and reproducible model artifacts.

Replace generic wording with your actual measured results after training on a documented public dataset.

## Limitations

- Dataset quality and representativeness constrain real-world performance.
- Model probabilities are not clinical probabilities.
- This project does not establish medical causation or diagnosis.
