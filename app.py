from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


st.set_page_config(page_title="ML Assignment 2 - Classifier Dashboard", layout="wide")

MODEL_DIR = Path("model")
METADATA_PATH = MODEL_DIR / "model_metadata.json"

MODEL_FILE_MAP = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
    "XGBoost (Ensemble)": "xgboost_ensemble.joblib",
}


@st.cache_data
def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            "model/model_metadata.json not found. Run `python model/train_models.py` first."
        )
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@st.cache_resource
def load_model(model_name: str):
    model_file = MODEL_DIR / MODEL_FILE_MAP[model_name]
    if not model_file.exists():
        raise FileNotFoundError(f"{model_file} not found. Run training script first.")
    return joblib.load(model_file)


def display_confusion_matrix(cm: list[list[int]], title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    st.pyplot(fig)


def calculate_uploaded_metrics(y_true: pd.Series, y_pred: pd.Series, y_proba: pd.Series) -> dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": float(roc_auc_score(y_true, y_proba)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


def main() -> None:
    st.title("Machine Learning Assignment 2: Classification Dashboard")
    st.write("Models trained on UCI Spambase dataset and deployed with Streamlit.")

    metadata = load_metadata()
    feature_columns = metadata["feature_columns"]
    model_names = list(metadata["metrics_by_model"].keys())

    st.subheader("Dataset Summary")
    ds = metadata["dataset"]
    st.write(
        f"**Dataset:** {ds['name']}  \n"
        f"**Source:** {ds['source']}  \n"
        f"**Task:** {ds['task']}  \n"
        f"**Instances:** {ds['instances']}  \n"
        f"**Features:** {ds['features']}"
    )

    st.subheader("Model Comparison Metrics")
    comparison_df = pd.DataFrame(metadata["metrics_by_model"]).T.reset_index(names="ML Model Name")
    comparison_df = comparison_df.rename(
        columns={
            "accuracy": "Accuracy",
            "auc": "AUC",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "mcc": "MCC",
        }
    )
    st.dataframe(comparison_df.style.format({c: "{:.4f}" for c in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]}), use_container_width=True)

    st.subheader("Select Model")
    selected_model_name = st.selectbox("Choose a classifier", model_names)
    model = load_model(selected_model_name)

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Selected Model Metrics (Holdout Test Set)**")
        selected_metrics = metadata["metrics_by_model"][selected_model_name]
        selected_metrics_df = pd.DataFrame(
            {
                "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
                "Value": [
                    selected_metrics["accuracy"],
                    selected_metrics["auc"],
                    selected_metrics["precision"],
                    selected_metrics["recall"],
                    selected_metrics["f1"],
                    selected_metrics["mcc"],
                ],
            }
        )
        st.dataframe(selected_metrics_df.style.format({"Value": "{:.4f}"}), use_container_width=True)

    with c2:
        st.write("**Confusion Matrix (Holdout Test Set)**")
        display_confusion_matrix(
            metadata["confusion_matrices"][selected_model_name],
            f"{selected_model_name} - Holdout Set",
        )

    st.subheader("Upload CSV for Prediction")
    st.caption("Upload only test data CSV. If `target` column is included, metrics and report will be computed.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data")
        st.dataframe(user_df.head(), use_container_width=True)

        has_target = "target" in user_df.columns
        inference_df = user_df.drop(columns=["target"]) if has_target else user_df.copy()

        missing_features = [c for c in feature_columns if c not in inference_df.columns]
        if missing_features:
            st.error(f"Missing required feature columns: {missing_features[:10]}{'...' if len(missing_features) > 10 else ''}")
            st.stop()

        inference_df = inference_df[feature_columns]
        preds = model.predict(inference_df)
        result_df = user_df.copy()
        result_df["predicted_target"] = preds

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(inference_df)[:, 1]
            result_df["predicted_probability"] = proba
        else:
            proba = preds

        st.write("Predictions")
        st.dataframe(result_df.head(20), use_container_width=True)

        if has_target:
            y_true = user_df["target"]
            y_pred = result_df["predicted_target"]
            uploaded_metrics = calculate_uploaded_metrics(y_true=y_true, y_pred=y_pred, y_proba=proba)
            st.write("**Evaluation on Uploaded File**")
            st.dataframe(
                pd.DataFrame(
                    {"Metric": list(uploaded_metrics.keys()), "Value": list(uploaded_metrics.values())}
                ).style.format({"Value": "{:.4f}"}),
                use_container_width=True,
            )

            st.write("**Confusion Matrix (Uploaded File)**")
            upload_cm = confusion_matrix(y_true, y_pred).astype(int).tolist()
            display_confusion_matrix(upload_cm, f"{selected_model_name} - Uploaded CSV")

            st.write("**Classification Report (Uploaded File)**")
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).T
            st.dataframe(report_df, use_container_width=True)


if __name__ == "__main__":
    main()
