# 🩺 Diabetes Risk Prediction AI

An AI-powered machine learning application that estimates diabetes risk from patient health-related inputs and presents the prediction through an easy-to-use interface.

> **A practical ML project combining data preprocessing, model training, evaluation, prediction, and an interactive application.**

---

## ✨ Features

* 🧠 Machine-learning-based diabetes risk prediction
* 📊 Data preprocessing and feature handling
* 🔍 Exploratory Data Analysis (EDA)
* 📈 Model evaluation with performance metrics
* 📉 Confusion matrix and ROC curve analysis
* 💡 Prediction/explanation workflow
* 🖥️ Interactive application interface
* 🧪 Prediction testing with automated test cases
* 📁 Organized ML project structure

---

## 🛠️ Tech Stack

| Technology                | Purpose                  |
| ------------------------- | ------------------------ |
| 🐍 Python                 | Core development         |
| 🤖 Machine Learning       | Diabetes risk prediction |
| 📊 Pandas / NumPy         | Data processing          |
| 📈 Matplotlib             | Data visualization       |
| 📓 Jupyter Notebook       | Exploratory analysis     |
| 🖥️ Application Interface | User interaction         |
| 🧪 Testing                | Prediction validation    |

---

## 🧠 Machine Learning Workflow

```text
User Input
    ↓
Data Preprocessing
    ↓
Feature Processing
    ↓
Trained ML Model
    ↓
Risk Prediction
    ↓
Result / Explanation
```

The project follows a complete machine-learning workflow rather than only implementing a prediction model.

---

## 📊 Model Evaluation

The project includes several evaluation artifacts:

* **Confusion Matrix** — to analyze classification performance
* **ROC Curve** — to evaluate the model's ability to distinguish between classes
* **Metrics JSON** — stores model evaluation results
* **Best Parameters JSON** — stores selected model parameters

These files are available inside the `models/` directory.

---

## 📂 Project Structure

```text
diabetes-risk-prediction/
│
├── app/
│   └── app.py
│
├── data/
│   ├── diabetes.csv
│   └── README.txt
│
├── models/
│   ├── best_params.json
│   ├── confusion_matrix.png
│   ├── feature_names.json
│   ├── metrics.json
│   └── roc_curve.png
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── __init__.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── make_demo_data.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
│
├── tests/
│   └── test_prediction.py
│
├── .gitignore
├── requirements.txt
├── run_project.bat
└── README.md
```

---

## 🚀 Future Improvements

Possible future enhancements include:

* Improving model performance with additional datasets
* Adding more advanced explainability techniques
* Expanding the application interface
* Adding user history and prediction tracking
* Deploying the application online
* Adding additional health-risk prediction modules

---

## 👩‍💻 Author

**Ishita Bhargava**

B.Tech — Computer Science & Technology

---

## ⭐ If You Find This Project Interesting

Feel free to explore the repository and the machine-learning workflow behind the application.
