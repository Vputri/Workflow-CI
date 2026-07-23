"""
modelling.py — Kriteria 3 (MLflow Project entry point)
======================================================
Training model churn yang dijalankan oleh `mlflow run` pada CI GitHub Actions.
Melatih RandomForest, mencatat parameter/metrik/model ke MLflow (local file store
di dalam folder MLProject), lalu menyimpan run_id agar step CI berikutnya dapat
membangun Docker image via `mlflow models build-docker`.
"""

import argparse
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

RANDOM_STATE = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), "telco_preprocessing")


def load_split(data_dir):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    return (
        train.drop(columns=["Churn"]),
        test.drop(columns=["Churn"]),
        train["Churn"],
        test["Churn"],
    )


def main(n_estimators, max_depth):
    X_train, X_test, y_train, y_test = load_split(DATA_DIR)

    with mlflow.start_run(run_name="ci_retraining") as run:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth > 0 else None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
        mlflow.log_metric("precision", precision_score(y_test, preds))
        mlflow.log_metric("recall", recall_score(y_test, preds))
        mlflow.log_metric("f1_score", f1_score(y_test, preds))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, proba))

        # Log model agar tersedia sebagai artefak untuk build-docker.
        mlflow.sklearn.log_model(model, artifact_path="model")

        run_id = run.info.run_id
        # Simpan run_id agar step CI berikutnya dapat mereferensikan model.
        with open(os.path.join(os.path.dirname(__file__), "run_id.txt"), "w") as f:
            f.write(run_id)

        print(f"[OK] Training selesai. run_id={run_id}")
        print(f"     accuracy={accuracy_score(y_test, preds):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()
    main(args.n_estimators, args.max_depth)
