"""
sagemaker_deploy.py
Simulates AWS SageMaker model deployment and real-time inference endpoint.

In production this would:
  1. Upload model artifact to S3
  2. Create a SageMaker Model from the artifact
  3. Deploy to a real-time endpoint (ml.m5.large instance)
  4. Expose a predict() function that calls the endpoint

Local simulation: loads the .pkl directly and mirrors the SageMaker
request/response contract so the rest of the pipeline works identically.

Real AWS usage (requires boto3 + sagemaker SDK):
  from sagemaker.sklearn import SKLearnModel
  model = SKLearnModel(
      model_data="s3://cinetag-training-sets/rf_classifier.tar.gz",
      role="arn:aws:iam::ACCOUNT:role/SageMakerRole",
      entry_point="train_model.py",
      framework_version="1.2-1",
  )
  predictor = model.deploy(
      instance_type="ml.m5.large",
      initial_instance_count=1,
      endpoint_name="cinetag-rf-v3-2",
  )
"""

import pickle
import json
import numpy as np
import os
from datetime import datetime


MODEL_PATH  = "models/rf_classifier.pkl"
MLB_PATH    = "models/mlb.pkl"
ENDPOINT    = "cinetag-rf-v3-2"   # mirrors the SageMaker endpoint name

FEATURE_COLS = [
    "vote_average", "rating_norm", "runtime_min", "log_revenue",
    "genre_count", "decade",
    "runtime_short", "runtime_medium", "runtime_long",
]


def log(level, msg):
    ts = datetime.now().strftime("%H:%M:%S")
    colors = {"INFO":"\033[36m","OK":"\033[32m","WARN":"\033[33m","ERR":"\033[31m"}
    reset = "\033[0m"
    print(f"{ts}  {colors.get(level,'')}{level:4s}{reset}  {msg}")


class CineTagEndpoint:
    """
    Local mirror of the SageMaker real-time inference endpoint.
    Exposes the same predict() interface.
    """

    def __init__(self, endpoint_name=ENDPOINT):
        self.endpoint_name = endpoint_name
        self.model = None
        self.mlb   = None
        self._load()

    def _load(self):
        log("INFO", f"Loading model artifact for endpoint '{self.endpoint_name}'")
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        with open(MLB_PATH, "rb") as f:
            self.mlb = pickle.load(f)
        log("OK", f"Endpoint ready — classes: {list(self.mlb.classes_)}")

    def _build_input(self, movie: dict) -> np.ndarray:
        """Convert a raw movie dict into the feature vector expected by the model."""
        revenue = movie.get("revenue", 0) or 0
        runtime = movie.get("runtime_min", 100) or 100
        row = {
            "vote_average":   float(movie.get("vote_average", 6.5)),
            "rating_norm":    (float(movie.get("vote_average", 6.5)) - 1) / 9.0,
            "runtime_min":    float(runtime),
            "log_revenue":    float(np.log1p(revenue)),
            "genre_count":    float(movie.get("genre_count", 1)),
            "decade":         float((int(movie.get("release_year", 2000)) // 10) * 10),
            "runtime_short":  1.0 if runtime < 90 else 0.0,
            "runtime_medium": 1.0 if 90 <= runtime <= 120 else 0.0,
            "runtime_long":   1.0 if runtime > 120 else 0.0,
        }
        return np.array([row[c] for c in FEATURE_COLS], dtype=np.float32).reshape(1, -1)

    def predict(self, movie: dict) -> dict:
        """
        SageMaker-compatible inference.
        Input:  movie metadata dict
        Output: {predicted_genres, probabilities, top_genre, confidence}
        """
        X = self._build_input(movie)
        classes = list(self.mlb.classes_)

        # Collect probabilities
        probas = np.zeros(len(classes))
        for i, estimator in enumerate(self.model.estimators_):
            p = estimator.predict_proba(X)
            probas[i] = p[0, 1] if p.shape[1] == 2 else p[0, 0]

        # Threshold at 0.45
        predicted = [classes[i] for i in range(len(classes)) if probas[i] >= 0.45]
        if not predicted:
            predicted = [classes[np.argmax(probas)]]

        return {
            "endpoint":         self.endpoint_name,
            "title":            movie.get("title", "Unknown"),
            "predicted_genres": predicted,
            "top_genre":        classes[np.argmax(probas)],
            "confidence":       round(float(probas.max()), 3),
            "probabilities":    {
                cls: round(float(probas[i]), 3)
                for i, cls in enumerate(classes)
            },
        }

    def predict_batch(self, movies: list) -> list:
        """Batch inference — mirrors SageMaker batch transform jobs."""
        log("INFO", f"Batch inference: {len(movies)} records")
        results = [self.predict(m) for m in movies]
        log("OK",   f"Batch complete — avg confidence: "
                    f"{np.mean([r['confidence'] for r in results]):.3f}")
        return results


# ── Demo ──────────────────────────────────────────────────────────────────────
DEMO_MOVIES = [
    {"title": "Dune: Part Two",       "vote_average": 8.5, "runtime_min": 166, "revenue": 711_000_000, "release_year": 2024, "genre_count": 3},
    {"title": "Saltburn",             "vote_average": 7.1, "runtime_min": 131, "revenue":  24_000_000, "release_year": 2023, "genre_count": 2},
    {"title": "Oppenheimer",          "vote_average": 8.9, "runtime_min": 180, "revenue": 952_000_000, "release_year": 2023, "genre_count": 3},
    {"title": "Talk to Me",           "vote_average": 7.2, "runtime_min":  95, "revenue":  30_000_000, "release_year": 2023, "genre_count": 2},
    {"title": "Past Lives",           "vote_average": 8.0, "runtime_min": 105, "revenue":  10_000_000, "release_year": 2023, "genre_count": 2},
    {"title": "The Killer",           "vote_average": 6.8, "runtime_min": 118, "revenue":   5_000_000, "release_year": 2023, "genre_count": 2},
]


if __name__ == "__main__":
    print("=" * 60)
    print(f"  CineTag — SageMaker Endpoint: {ENDPOINT}")
    print("=" * 60)

    endpoint = CineTagEndpoint()

    print("\n── Single inference ──")
    result = endpoint.predict(DEMO_MOVIES[0])
    print(json.dumps(result, indent=2))

    print("\n── Batch inference ──")
    results = endpoint.predict_batch(DEMO_MOVIES)
    print()
    print(f"  {'Title':<35} {'Predicted Genres':<40} {'Conf':>6}")
    print("  " + "-" * 83)
    for r in results:
        genres = ", ".join(r["predicted_genres"])
        print(f"  {r['title']:<35} {genres:<40} {r['confidence']:>6.3f}")
