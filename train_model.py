"""
train_model.py
Trains a multi-label genre classifier using Random Forest (scikit-learn).
Mirrors what runs inside an AWS SageMaker training job.

SageMaker entry-point usage:
  In sagemaker_deploy.py, this script is passed as:
    SKLearn(entry_point='train_model.py', framework_version='1.2-1', ...)

Local usage:
  python train_model.py
"""

import pandas as pd
import numpy as np
import json
import os
import pickle
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, accuracy_score,
    hamming_loss, f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ── Config ────────────────────────────────────────────────────────────────────
PROC_PATH   = "data/processed/tmdb_movies_processed.csv"
MODEL_DIR   = "models"
MODEL_PATH  = f"{MODEL_DIR}/rf_classifier.pkl"
MLB_PATH    = f"{MODEL_DIR}/mlb.pkl"
METRICS_PATH= f"{MODEL_DIR}/metrics.json"

FEATURE_COLS = [
    "vote_average", "rating_norm", "runtime_min", "log_revenue",
    "genre_count", "decade",
    "runtime_short", "runtime_medium", "runtime_long",
]

TARGET_GENRES = [
    "Action", "Adventure", "Comedy", "Crime", "Drama",
    "Horror", "Mystery", "Romance", "Science Fiction", "Thriller"
]

RF_PARAMS = {
    "n_estimators":   200,
    "max_depth":       18,
    "min_samples_split": 4,
    "min_samples_leaf":  2,
    "max_features":   "sqrt",
    "class_weight":   "balanced",
    "random_state":    42,
    "n_jobs":          -1,
}


def log(level, msg):
    ts = datetime.now().strftime("%H:%M:%S")
    colors = {"INFO":"\033[36m","OK":"\033[32m","WARN":"\033[33m","ERR":"\033[31m"}
    reset = "\033[0m"
    print(f"{ts}  {colors.get(level,'')}{level:4s}{reset}  {msg}")


# ── Load & prepare data ───────────────────────────────────────────────────────
def load_data():
    log("INFO", f"Loading processed data from {PROC_PATH}")
    df = pd.read_csv(PROC_PATH)

    # Parse genres_clean back from string representation
    import ast
    df["genres_clean"] = df["genres_clean"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )

    # Keep only records with at least one target genre
    df["genres_filtered"] = df["genres_clean"].apply(
        lambda gs: [g for g in gs if g in TARGET_GENRES]
    )
    df = df[df["genres_filtered"].apply(len) > 0].copy()

    # Fill missing dummy columns with 0
    for col in ["runtime_short", "runtime_medium", "runtime_long"]:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Ensure all feature cols exist and are numeric
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    log("OK", f"Loaded {len(df)} training-ready records")
    return df


def build_features(df):
    X = df[FEATURE_COLS].values.astype(np.float32)
    return X


def build_labels(df):
    mlb = MultiLabelBinarizer(classes=TARGET_GENRES)
    Y = mlb.fit_transform(df["genres_filtered"])
    log("INFO", f"Label classes: {list(mlb.classes_)}")
    log("INFO", f"Label matrix shape: {Y.shape}")
    return Y, mlb


# ── Train ─────────────────────────────────────────────────────────────────────
def train(X_train, Y_train):
    log("INFO", f"Training RandomForest — n_estimators={RF_PARAMS['n_estimators']}, "
                f"max_depth={RF_PARAMS['max_depth']}")

    base_rf = RandomForestClassifier(**RF_PARAMS)
    model = MultiOutputClassifier(base_rf, n_jobs=-1)
    model.fit(X_train, Y_train)

    log("OK", "Training complete")
    return model


# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(model, X_test, Y_test, mlb):
    log("INFO", "Evaluating model on test split")

    Y_pred = model.predict(X_test)

    # Exact match (subset accuracy)
    exact = accuracy_score(Y_test, Y_pred)

    # Hamming loss (fraction of wrong labels)
    hl = hamming_loss(Y_test, Y_pred)

    # Macro F1
    f1_macro = f1_score(Y_test, Y_pred, average="macro", zero_division=0)
    f1_micro = f1_score(Y_test, Y_pred, average="micro", zero_division=0)

    # Per-class report
    report = classification_report(
        Y_test, Y_pred,
        target_names=list(mlb.classes_),
        output_dict=True,
        zero_division=0
    )

    metrics = {
        "exact_match_accuracy": round(exact, 4),
        "hamming_loss":         round(hl, 4),
        "f1_macro":             round(f1_macro, 4),
        "f1_micro":             round(f1_micro, 4),
        "per_class":            {
            cls: {
                "precision": round(report[cls]["precision"], 3),
                "recall":    round(report[cls]["recall"], 3),
                "f1":        round(report[cls]["f1-score"], 3),
            }
            for cls in list(mlb.classes_) if cls in report
        }
    }

    log("OK", f"Exact match accuracy : {exact*100:.1f}%")
    log("OK", f"Hamming loss         : {hl:.4f}")
    log("OK", f"F1 macro             : {f1_macro*100:.1f}%")
    log("OK", f"F1 micro             : {f1_micro*100:.1f}%")

    return metrics, Y_pred


# ── Save artifacts ────────────────────────────────────────────────────────────
def save_artifacts(model, mlb, metrics):
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    log("OK", f"Model saved → {MODEL_PATH}")

    with open(MLB_PATH, "wb") as f:
        pickle.dump(mlb, f)
    log("OK", f"MultiLabelBinarizer saved → {MLB_PATH}")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    log("OK", f"Metrics saved → {METRICS_PATH}")


# ── SageMaker entrypoint ──────────────────────────────────────────────────────
def sagemaker_train():
    """
    Called by SageMaker training job. Reads from /opt/ml/input/data/train/
    and writes model artifacts to /opt/ml/model/.
    Locally, we use data/processed/ and models/ instead.
    """
    df    = load_data()
    X     = build_features(df)
    Y, mlb = build_labels(df)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    log("INFO", f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    model   = train(X_train, Y_train)
    metrics, _ = evaluate(model, X_test, Y_test, mlb)
    save_artifacts(model, mlb, metrics)

    return model, mlb, metrics


if __name__ == "__main__":
    print("=" * 60)
    print("  CineTag — SageMaker Training Job (local simulation)")
    print("=" * 60)
    model, mlb, metrics = sagemaker_train()

    print("\n── Final metrics ──")
    print(f"  Exact match accuracy : {metrics['exact_match_accuracy']*100:.1f}%")
    print(f"  Hamming loss         : {metrics['hamming_loss']}")
    print(f"  F1 macro             : {metrics['f1_macro']*100:.1f}%")
    print(f"  F1 micro             : {metrics['f1_micro']*100:.1f}%")
    print("\n── Per-class F1 ──")
    for cls, v in metrics["per_class"].items():
        bar = "█" * int(v["f1"] * 20)
        print(f"  {cls:<20} {bar:<20} {v['f1']*100:.1f}%")
