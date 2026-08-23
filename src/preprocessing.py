import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "Outcome"
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if col != TARGET:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ZERO_AS_MISSING:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
    df = df.drop_duplicates().reset_index(drop=True)
    return df

def split_features_target(df: pd.DataFrame):
    if TARGET not in df.columns:
        raise ValueError(f"Dataset must contain target column '{TARGET}'.")
    X = df.drop(columns=[TARGET])
    y = pd.to_numeric(df[TARGET], errors="coerce")
    mask = y.notna()
    return X.loc[mask].reset_index(drop=True), y.loc[mask].astype(int).reset_index(drop=True)

def build_preprocessor(X):
    numeric_features = X.columns.tolist()
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features)
    ])
