"""
validation_layer.py
Post-prediction validation that improves label quality by:
  1. Confidence threshold filtering
  2. Genre co-occurrence plausibility checks
  3. Rule-based overrides (domain knowledge)
  4. Consistency scoring across batch

This layer runs between SageMaker inference and final S3 write.
"""

import pandas as pd
import numpy as np
import json
import pickle
import os
from datetime import datetime


PROC_PATH    = "data/processed/tmdb_movies_processed.csv"
MODEL_PATH   = "models/rf_classifier.pkl"
MLB_PATH     = "models/mlb.pkl"
OUTPUT_PATH  = "data/processed/tmdb_final_labeled.csv"
REPORT_PATH  = "data/processed/validation_report.json"

CONFIDENCE_THRESHOLD = 0.50

# Genre co-occurrence rules: if genre A is predicted, genre B is implausible
CONFLICT_RULES = {
    ("Horror", "Comedy"):        0.30,   # low but possible (horror-comedy exists)
    ("Romance", "Horror"):       0.25,
    ("Animation", "Thriller"):   0.40,
    ("Science Fiction", "Crime"):0.45,
}

# Minimum confidence to keep a label (per genre)
MIN_CONFIDENCE = {
    "Action":          0.45,
    "Adventure":       0.45,
    "Comedy":          0.50,
    "Crime":           0.50,
    "Drama":           0.40,   # very common, lower bar
    "Horror":          0.55,
    "Mystery":         0.50,
    "Romance":         0.50,
    "Science Fiction": 0.50,
    "Thriller":        0.48,
}


def log(level, msg):
    ts = datetime.now().strftime("%H:%M:%S")
    colors = {"INFO":"\033[36m","OK":"\033[32m","WARN":"\033[33m","ERR":"\033[31m"}
    reset = "\033[0m"
    print(f"{ts}  {colors.get(level,'')}{level:4s}{reset}  {msg}")


def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(MLB_PATH, "rb") as f:
        mlb = pickle.load(f)
    log("OK", "Model and label binarizer loaded")
    return model, mlb


def load_data():
    import ast
    df = pd.read_csv(PROC_PATH)
    df["genres_clean"] = df["genres_clean"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    for col in ["runtime_short","runtime_medium","runtime_long"]:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


# ── Inference + probabilities ─────────────────────────────────────────────────
def get_predictions_with_proba(model, mlb, df):
    FEATURE_COLS = [
        "vote_average","rating_norm","runtime_min","log_revenue",
        "genre_count","decade",
        "runtime_short","runtime_medium","runtime_long",
    ]
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    X = df[FEATURE_COLS].values.astype(np.float32)

    # Collect per-class probabilities from each estimator
    probas = np.zeros((len(X), len(mlb.classes_)))
    for i, estimator in enumerate(model.estimators_):
        p = estimator.predict_proba(X)
        # p[:,1] = probability of class=1
        if p.shape[1] == 2:
            probas[:, i] = p[:, 1]
        else:
            probas[:, i] = p[:, 0]

    return probas


# ── Validation checks ─────────────────────────────────────────────────────────
def apply_confidence_threshold(probas, mlb):
    """Apply per-genre minimum confidence thresholds."""
    classes = list(mlb.classes_)
    mask = np.zeros_like(probas, dtype=bool)
    for i, cls in enumerate(classes):
        threshold = MIN_CONFIDENCE.get(cls, CONFIDENCE_THRESHOLD)
        mask[:, i] = probas[:, i] >= threshold
    return mask


def apply_conflict_rules(mask, mlb):
    """Remove labels that violate co-occurrence plausibility rules."""
    classes = list(mlb.classes_)
    conflict_count = 0
    for (genre_a, genre_b), max_co_prob in CONFLICT_RULES.items():
        if genre_a not in classes or genre_b not in classes:
            continue
        idx_a = classes.index(genre_a)
        idx_b = classes.index(genre_b)
        # Both predicted → remove the less common one (genre_b)
        both = mask[:, idx_a] & mask[:, idx_b]
        conflict_count += both.sum()
        mask[both, idx_b] = False   # suppress the secondary genre
    if conflict_count > 0:
        log("WARN", f"Conflict rules suppressed {conflict_count} label assignments")
    return mask, conflict_count


def compute_consistency_score(original_labels, predicted_labels, mlb):
    """
    Tagging consistency: fraction of original genre labels that match predictions.
    Measures improvement over naive baseline.
    """
    classes = list(mlb.classes_)
    from sklearn.preprocessing import MultiLabelBinarizer as MLB
    m = MLB(classes=classes)
    m.fit([[c] for c in classes])

    orig_bin  = m.transform(original_labels)
    pred_bin  = predicted_labels

    # Only evaluate rows where original has >= 1 label
    valid = orig_bin.sum(axis=1) > 0
    if valid.sum() == 0:
        return 0.0

    matches = (orig_bin[valid] == pred_bin[valid]).all(axis=1).sum()
    consistency = matches / valid.sum()
    return float(consistency)


# ── Main validation pipeline ──────────────────────────────────────────────────
def run_validation():
    log("INFO", "Starting validation layer")

    model, mlb = load_artifacts()
    df = load_data()
    classes = list(mlb.classes_)

    # Get raw probabilities
    probas = get_predictions_with_proba(model, mlb, df)
    log("INFO", f"Raw predictions computed for {len(df)} records")

    # Apply thresholds
    mask = apply_confidence_threshold(probas, mlb)
    pre_conflict_count = mask.sum()

    # Apply conflict rules
    mask, conflicts = apply_conflict_rules(mask, mlb)

    # Build final label lists
    df["predicted_genres"] = [
        [classes[i] for i in range(len(classes)) if mask[j, i]]
        for j in range(len(df))
    ]

    # Attach top-1 predicted genre
    df["predicted_primary"] = [
        classes[np.argmax(probas[j])] for j in range(len(df))
    ]

    # Per-record max confidence
    df["max_confidence"] = probas.max(axis=1).round(3)

    # Flag low-confidence records
    df["low_confidence"] = df["max_confidence"] < CONFIDENCE_THRESHOLD
    low_conf_count = df["low_confidence"].sum()

    # Consistency score
    original_labels = df["genres_clean"].tolist()
    consistency = compute_consistency_score(original_labels, mask, mlb)
    log("OK", f"Tagging consistency score: {consistency*100:.1f}%")
    log("OK", f"Low-confidence records   : {low_conf_count} ({low_conf_count/len(df)*100:.1f}%)")
    log("OK", f"Conflict suppressions    : {conflicts}")

    # Save
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    log("OK", f"Final labeled data → {OUTPUT_PATH}")

    report = {
        "total_records":        len(df),
        "low_confidence_count": int(low_conf_count),
        "low_confidence_pct":   round(low_conf_count / len(df) * 100, 2),
        "conflict_suppressions":int(conflicts),
        "tagging_consistency":  round(consistency, 4),
        "avg_max_confidence":   round(float(df["max_confidence"].mean()), 4),
        "label_coverage": {
            cls: int(mask[:, i].sum())
            for i, cls in enumerate(classes)
        }
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    log("OK", f"Validation report → {REPORT_PATH}")

    return df, report


if __name__ == "__main__":
    print("=" * 60)
    print("  CineTag — Validation Layer")
    print("=" * 60)
    df, report = run_validation()

    print("\n── Validation Report ──")
    for k, v in report.items():
        if k != "label_coverage":
            print(f"  {k:<28} {v}")

    print("\n── Label coverage ──")
    for genre, count in sorted(report["label_coverage"].items(), key=lambda x: -x[1]):
        bar = "█" * (count // 5)
        print(f"  {genre:<20} {bar} {count}")

    print("\n── Sample predictions ──")
    sample = df[["title","genres_clean","predicted_genres","max_confidence","low_confidence"]].head(10)
    for _, row in sample.iterrows():
        flag = "⚠" if row["low_confidence"] else "✓"
        print(f"  {flag} {row['title']:<40} "
              f"orig={row['genres_clean']}  →  pred={row['predicted_genres']}  "
              f"conf={row['max_confidence']:.2f}")
