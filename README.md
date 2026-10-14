# Student Stress Level Prediction Using Machine Learning

> An ML-based approach for early identification of academic stress among students, built as a CA-III/CA4 mini-project at MGM University, Institute of Information and Communication Technology (IICT), B.Tech Data Science.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## Team

| Name | Roll No. |
|---|---|
| Husen Bagwan | 2407038 |
| Umar Shaikh | 2407003 |
| Uzaif Shaikh | 2407034 |

**Guide:** Prof. Priyanka Hajare

## Problem Statement

Can machine learning identify a student's stress level (**Low / Moderate / High**) using everyday academic and lifestyle factors — without needing specialized wearable or physiological sensors?

## Dataset

`student_lifestyle_dataset.csv` — 2,000 student records, 8 columns:

| Column | Description |
|---|---|
| Student_ID | Row identifier (not used as a feature) |
| Study_Hours_Per_Day | Hours spent studying per day |
| Extracurricular_Hours_Per_Day | Hours in extracurricular activities |
| Sleep_Hours_Per_Day | Hours of sleep per day |
| Social_Hours_Per_Day | Hours of social interaction |
| Physical_Activity_Hours_Per_Day | Hours of physical activity |
| GPA | Current grade point average |
| **Stress_Level** | **Target** — Low / Moderate / High |

Class distribution: High 51.5% · Moderate 33.7% · Low 14.8% (imbalanced — macro-averaged metrics are used throughout).

## Approach

1. **Preprocessing** — drop `Student_ID`, check/remove duplicates, label-encode the target, stratified 80/20 train-test split, `StandardScaler` feature scaling.
2. **Modelling** — four classifiers trained on identical data: Logistic Regression, Decision Tree, Random Forest, XGBoost (falls back to `GradientBoostingClassifier` if `xgboost` isn't installed).
3. **Evaluation** — accuracy, macro-precision/recall/F1, and confusion matrices for each model.
4. **Live demo** — a `predict_stress()` function that takes a student's inputs and returns a predicted stress level, for viva/demonstration.

## Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 84.75% | 0.844 | 0.855 | 0.848 |
| Decision Tree | 100% | 1.000 | 1.000 | 1.000 |
| Random Forest | 100% | 1.000 | 1.000 | 1.000 |
| XGBoost / Gradient Boosting | 100% | 1.000 | 1.000 | 1.000 |

**Key finding:** Feature-importance analysis shows `Study_Hours_Per_Day` and `Sleep_Hours_Per_Day` alone account for almost all of the predictive signal (~73% and ~27% of importance respectively) — tree-based models recover what looks like a simple threshold rule on these two features, which is why they fit this test set perfectly. This is a genuine result worth discussing rather than a bug: it also means this dataset is unusually clean compared to noisier real-world survey data, and the model should be validated on an independent dataset before being trusted for real decisions.

![Model comparison](results/model_comparison.png)
![Confusion matrices](results/confusion_matrices.png)
![Class distribution](results/class_distribution.png)
![Features vs stress](results/features_vs_stress.png)

## Project Structure

```
.
├── stress_management_system.py          # Single-file script: train, evaluate, live demo menu
├── Student_Stress_Level_Prediction.ipynb # Notebook version (Colab-ready)
├── student_lifestyle_dataset.csv         # Dataset (2,000 records)
├── requirements.txt
├── results/                              # Generated charts
│   ├── class_distribution.png
│   ├── features_vs_stress.png
│   ├── confusion_matrices.png
│   └── model_comparison.png
└── docs/
    └── Student_Stress_Level_Prediction_Report.docx  # Full case study report
```

## How to Run

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd <your-repo-folder>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python stress_management_system.py
```

Or open `Student_Stress_Level_Prediction.ipynb` directly in [Google Colab](https://colab.research.google.com) — no local setup needed.

## Tech Stack

- Python, pandas, numpy
- scikit-learn (Logistic Regression, Decision Tree, Random Forest)
- XGBoost
- matplotlib, seaborn

## Future Scope

- Larger, multi-institution datasets to test generalization
- Explainable AI (SHAP) for per-prediction explanations
- Streamlit web app for live student self-assessment
- Real-time monitoring and periodic self-check-ins

## License

This project is licensed under the MIT License.
