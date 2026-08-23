from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from src.preprocessing import clean_dataset, split_features_target

def main():
    model_path = Path("models/diabetes_pipeline.pkl")
    if not model_path.exists():
        raise FileNotFoundError("Train the model first with: python -m src.train")

    df = clean_dataset(pd.read_csv("data/diabetes.csv"))
    X, y = split_features_target(df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    model = joblib.load(model_path)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, pred, target_names=["No Diabetes", "Diabetes"], zero_division=0))
    print(f"ROC-AUC: {roc_auc_score(y_test, prob):.4f}")

    Path("models").mkdir(exist_ok=True)
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=["No Diabetes", "Diabetes"]).plot()
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png", dpi=200)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, prob)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig("models/roc_curve.png", dpi=200)
    plt.close()
    print("Saved evaluation plots to models/")

if __name__ == "__main__":
    main()
