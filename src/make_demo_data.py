"""Generate a reproducible synthetic dataset for testing the project.
This is NOT suitable for final resume metrics or medical conclusions.
"""
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    rng = np.random.default_rng(42)
    n = 1200
    age = rng.integers(21, 75, n)
    pregnancies = np.clip(rng.poisson(2.5, n), 0, 15)
    glucose = np.clip(rng.normal(120, 30, n), 45, 220)
    bp = np.clip(rng.normal(72, 12, n), 40, 130)
    skin = np.clip(rng.normal(25, 8, n), 5, 60)
    insulin = np.clip(rng.lognormal(np.log(100), 0.55, n), 10, 500)
    bmi = np.clip(rng.normal(31, 6, n), 16, 55)
    pedigree = np.clip(rng.lognormal(np.log(0.45), 0.45, n), 0.05, 2.5)

    logit = (
        -7.2 + 0.035*glucose + 0.055*bmi + 0.018*age
        + 0.16*pregnancies + 0.55*pedigree + 0.008*bp
    )
    p = 1 / (1 + np.exp(-logit))
    outcome = rng.binomial(1, np.clip(p, 0.02, 0.98))

    df = pd.DataFrame({
        "Pregnancies": pregnancies,
        "Glucose": glucose.round(1),
        "BloodPressure": bp.round(1),
        "SkinThickness": skin.round(1),
        "Insulin": insulin.round(1),
        "BMI": bmi.round(1),
        "DiabetesPedigreeFunction": pedigree.round(3),
        "Age": age,
        "Outcome": outcome
    })

    path = Path("data/diabetes.csv")
    path.parent.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Created {path} with {len(df)} rows.")
    print("Replace this file with a documented public dataset before using final resume metrics.")

if __name__ == "__main__":
    main()
