\# ML Pipeline with MLflow Tracking



A complete machine learning pipeline that trains multiple models on the Titanic survival dataset, with automated experiment tracking using MLflow.



\## What it does



\- Loads and preprocesses the Titanic dataset (feature engineering: family size, missing value imputation, encoding)

\- Trains 3 models: Logistic Regression, Random Forest, and XGBoost

\- Logs all hyperparameters and metrics automatically to MLflow

\- Saves trained models for version comparison



\## Results



| Model | Accuracy | F1 Score | AUC-ROC |

|---|---|---|---|

| Logistic Regression | 0.799 | 0.746 | 0.824 |

| Random Forest | 0.821 | 0.781 | 0.898 |

| XGBoost | 0.782 | 0.738 | 0.876 |



Random Forest performed best on this run.



\## Tech Stack



\- Python, scikit-learn, XGBoost

\- MLflow for experiment tracking

\- Pandas for data processing



\## How to run



pip install -r requirements.txt

python pipeline.py

mlflow ui



Then open http://localhost:5000 to view the experiment dashboard.

