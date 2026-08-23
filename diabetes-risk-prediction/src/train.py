import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from src.preprocessing import clean_dataset, split_features_target, build_preprocessor

RANDOM_STATE = 42
DATA_PATH = Path("data/diabetes.csv")
MODEL_DIR = Path("models")

def metrics(model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
    }

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("data/diabetes.csv not found. Run 'python -m src.make_demo_data' for a test dataset or add your own dataset.")

    df = clean_dataset(pd.read_csv(DATA_PATH))
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor(X_train)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    results = {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        results[name] = metrics(pipe, X_test, y_test)

    rf_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))
    ])
    params = {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [None, 4, 6, 8, 10],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        rf_pipe, params, n_iter=20, scoring="roc_auc", cv=cv,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=1
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    results["Tuned Random Forest"] = metrics(best, X_test, y_test)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best, MODEL_DIR / "diabetes_pipeline.pkl")
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(MODEL_DIR / "feature_names.json", "w") as f:
        json.dump(list(X.columns), f, indent=2)
    with open(MODEL_DIR / "best_params.json", "w") as f:
        json.dump(search.best_params_, f, indent=2)

    print("\nMODEL RESULTS")
    for name, values in results.items():
        print(f"\n{name}")
        for k, v in values.items():
            print(f"  {k}: {v:.4f}")
    print("\nSaved models/diabetes_pipeline.pkl")

if __name__ == "__main__":
    main()
