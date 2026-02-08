from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


RANDOM_STATE = 42
TEST_SIZE = 0.2


def ensure_dirs() -> tuple[Path, Path]:
    model_dir = Path("model")
    data_dir = Path("data")
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    return model_dir, data_dir


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load UCI Spambase via OpenML (public dataset)."""
    dataset = fetch_openml(name="spambase", version=1, as_frame=True)
    x = dataset.data.copy()
    y = dataset.target.astype(int)

    # Coerce any non-numeric values (none expected for Spambase).
    x = x.apply(pd.to_numeric, errors="coerce")
    if x.isna().any().any():
        x = x.fillna(x.median(numeric_only=True))

    return x, y


def build_models() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
            ]
        ),
        "KNN": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        "Naive Bayes": Pipeline(
            steps=[
                ("model", GaussianNB()),
            ]
        ),
        "Random Forest (Ensemble)": Pipeline(
            steps=[
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "XGBoost (Ensemble)": Pipeline(
            steps=[
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=5,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=RANDOM_STATE,
                        eval_metric="logloss",
                    ),
                )
            ]
        ),
    }


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict, list[list[int]], dict]:
    y_pred = model.predict(x_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(x_test)[:, 1]
    else:
        # Fallback for classifiers without predict_proba.
        y_proba = y_pred

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, y_proba)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_test, y_pred)),
    }
    cm = confusion_matrix(y_test, y_pred).astype(int).tolist()
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return metrics, cm, report


def main() -> None:
    model_dir, data_dir = ensure_dirs()
    x, y = load_dataset()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = build_models()
    metrics_by_model: dict[str, dict] = {}
    confusion_by_model: dict[str, list[list[int]]] = {}
    reports_by_model: dict[str, dict] = {}

    for model_name, pipeline in models.items():
        pipeline.fit(x_train, y_train)
        metrics, cm, report = evaluate_model(pipeline, x_test, y_test)
        metrics_by_model[model_name] = metrics
        confusion_by_model[model_name] = cm
        reports_by_model[model_name] = report

        model_path = model_dir / f"{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.joblib"
        joblib.dump(pipeline, model_path)
        print(f"Saved model: {model_path}")

    metadata = {
        "dataset": {
            "name": "UCI Spambase (via OpenML)",
            "source": "https://archive.ics.uci.edu/dataset/94/spambase",
            "instances": int(x.shape[0]),
            "features": int(x.shape[1]),
            "task": "Binary Classification",
        },
        "target_column": "target",
        "feature_columns": list(x.columns),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "metrics_by_model": metrics_by_model,
        "confusion_matrices": confusion_by_model,
        "classification_reports": reports_by_model,
    }

    with (model_dir / "model_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("Saved metadata: model/model_metadata.json")

    metrics_table = (
        pd.DataFrame(metrics_by_model)
        .T.rename(
            columns={
                "accuracy": "Accuracy",
                "auc": "AUC",
                "precision": "Precision",
                "recall": "Recall",
                "f1": "F1",
                "mcc": "MCC",
            }
        )
        .reset_index(names="ML Model Name")
    )
    metrics_table.to_csv(model_dir / "model_comparison_metrics.csv", index=False)
    print("Saved metrics table: model/model_comparison_metrics.csv")

    test_df_with_target = x_test.copy()
    test_df_with_target["target"] = y_test.values
    test_df_with_target.to_csv(data_dir / "test_data_with_target.csv", index=False)
    x_test.to_csv(data_dir / "test_data_features_only.csv", index=False)
    print("Saved sample test files in data/")


if __name__ == "__main__":
    main()
