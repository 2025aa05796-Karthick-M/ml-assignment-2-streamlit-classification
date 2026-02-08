# Machine Learning Assignment 2 (BITS WILP)

## 1. Problem Statement
Build and compare multiple machine learning classification models on a single public dataset, then deploy an interactive Streamlit web app that supports:
- CSV upload for test data
- Model selection
- Evaluation metric display
- Confusion matrix and classification report display

The objective is to demonstrate an end-to-end workflow: data handling, model training, evaluation, UI development, and deployment.

## 2. Dataset Description
- **Dataset Name:** UCI Spambase (via OpenML)
- **Source:** https://archive.ics.uci.edu/dataset/94/spambase
- **Task Type:** Binary Classification (Spam = 1, Not Spam = 0)
- **Instances:** 4601
- **Features:** 57 numeric input features
- **Why chosen:** Meets assignment constraints (minimum 12 features, minimum 500 instances) and is suitable for comparing linear, tree-based, probabilistic, and ensemble classifiers.

## 3. Models Used
- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors (KNN)
- Naive Bayes (Gaussian)
- Random Forest (Ensemble)
- XGBoost (Ensemble)

## 4. Comparison Table (All Required Metrics)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9294 | 0.9702 | 0.9209 | 0.8981 | 0.9093 | 0.8518 |
| Decision Tree | 0.9110 | 0.9078 | 0.8828 | 0.8926 | 0.8877 | 0.8140 |
| KNN | 0.9088 | 0.9566 | 0.8930 | 0.8733 | 0.8830 | 0.8084 |
| Naive Bayes | 0.8339 | 0.9449 | 0.7178 | 0.9532 | 0.8189 | 0.6941 |
| Random Forest (Ensemble) | 0.9457 | 0.9835 | 0.9510 | 0.9091 | 0.9296 | 0.8860 |
| XGBoost (Ensemble) | 0.9446 | 0.9873 | 0.9333 | 0.9256 | 0.9295 | 0.8839 |

## 5. Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong baseline with high overall performance and stable balance between precision and recall. |
| Decision Tree | Good interpretability, but slightly lower generalization performance than ensemble methods. |
| KNN | Competitive AUC, but accuracy and MCC are lower than Logistic Regression and ensemble models on this dataset. |
| Naive Bayes | Very high recall, but lower precision; catches more spam but produces more false positives. |
| Random Forest (Ensemble) | Best overall balance in Accuracy, Precision, F1, and MCC among all models. |
| XGBoost (Ensemble) | Highest AUC and strong F1/Recall, indicating excellent ranking performance and robust classification quality. |

## 6. Streamlit App Features Implemented
- CSV dataset upload option (`.csv`)
- Model selection dropdown (all 6 models)
- Evaluation metrics display (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix and classification report

## 7. Project Structure

```text
project-folder/
|-- app.py
|-- requirements.txt
|-- README.md
|-- data/
|   |-- test_data_features_only.csv
|   |-- test_data_with_target.csv
|-- model/
|   |-- train_models.py
|   |-- logistic_regression.joblib
|   |-- decision_tree.joblib
|   |-- knn.joblib
|   |-- naive_bayes.joblib
|   |-- random_forest_ensemble.joblib
|   |-- xgboost_ensemble.joblib
|   |-- model_metadata.json
|   |-- model_comparison_metrics.csv
```

## 8. How to Run Locally

```bash
pip install -r requirements.txt
python model/train_models.py
streamlit run app.py
```

## 9. Required Submission Links (to include in final PDF)
- **GitHub Repository Link:** `https://github.com/<your-username>/<your-repo>`
- **Live Streamlit App Link:** `https://<your-app-name>.streamlit.app`
- **BITS Virtual Lab Screenshot:** Add one screenshot proving assignment execution on BITS lab.
