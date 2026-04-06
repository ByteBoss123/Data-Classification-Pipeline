"""
lambda_ingest.py
Simulates AWS Lambda handler for automated ingestion and preprocessing.

In production:
  - Triggered by S3 PutObject events on s3://cinetag-raw-ingest/
  - Reads raw CSV/JSON, cleans and normalizes, writes to s3://cinetag-processed/
  - Deployed via: aws lambda create-function --function-name cinetag-ingest ...

Local simulation writes to data/processed/ instead of S3.
"""

import pandas as pd
import numpy as np
import json
import os
import re
from datetime import datetime

# ── Constants ────────────────────────────────────────────────────────────────
VALID_GENRES = {
    "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime",
    "Drama", "Fantasy", "Horror", "Music", "Mystery", "Romance",
    "Science Fiction", "Thriller", "War"
}
CONFIDENCE_THRESHOLD = 0.70
RAW_PATH   = "data/raw/tmdb_movies_raw.csv"
PROC_PATH  = "data/processed/tmdb_movies_processed.csv"
FLAG_PATH  = "data/processed/flagged_records.csv"


# ── Logging helper ───────────────────────────────────────────────────────────
def log(level, msg):
    ts = datetime.now().strftime("%H:%M:%S")
    colors = {"INFO": "\033[36m", "OK": "\033[32m", "WARN": "\033[33m", "ERR": "\033[31m"}
    reset = "\033[0m"
    print(f"{ts}  {colors.get(level,'')}{level:4s}{reset}  {msg}")


# ── Step 1: Load raw data (simulates S3 read) ────────────────────────────────
def load_raw(path=RAW_PATH):
    log("INFO", f"Loading raw data from {path}")
    df = pd.read_csv(path)
    log("OK",   f"Loaded {len(df)} records, {df.shape[1]} columns")
    return df


# ── Step 2: Clean & validate ─────────────────────────────────────────────────
def clean(df: pd.DataFrame):
    log("INFO", "Starting cleaning pass")
    original_len = len(df)
    issues = []

    # 2a. Parse genres from JSON string
    def parse_genres(raw):
        try:
            g = json.loads(raw) if isinstance(raw, str) else []
            return [x for x in g if x in VALID_GENRES]
        except Exception:
            return []

    df["genres_clean"] = df["genres_raw"].apply(parse_genres)

    # 2b. Flag records with no valid genres
    missing_genre_mask = df["genres_clean"].apply(len) == 0
    issues += list(df[missing_genre_mask]["movie_id"])
    log("WARN", f"{missing_genre_mask.sum()} records have no valid genres → flagged")

    # 2c. Clip and validate numerics
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce")
    df["runtime_min"]  = pd.to_numeric(df["runtime_min"],  errors="coerce")
    df["revenue"]      = pd.to_numeric(df["revenue"],      errors="coerce").fillna(0)

    bad_rating = (df["vote_average"] < 1) | (df["vote_average"] > 10) | df["vote_average"].isna()
    bad_runtime = (df["runtime_min"] < 1) | (df["runtime_min"] > 300) | df["runtime_min"].isna()
    issues += list(df[bad_rating | bad_runtime]["movie_id"])

    df["vote_average"] = df["vote_average"].clip(1, 10).fillna(df["vote_average"].median())
    df["runtime_min"]  = df["runtime_min"].clip(1, 300).fillna(df["runtime_min"].median())

    # 2d. Normalize title text
    df["title_clean"] = df["title"].astype(str).str.strip().str.title()

    # 2e. Release year sanity
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(2000).astype(int)
    df["release_year"] = df["release_year"].clip(1900, 2030)

    flagged_ids = set(issues)
    df["flagged"] = df["movie_id"].isin(flagged_ids)

    clean_df  = df[~df["flagged"]].copy()
    flagged_df = df[df["flagged"]].copy()

    log("OK",  f"Cleaning complete: {len(clean_df)} clean, {len(flagged_df)} flagged "
               f"({len(flagged_df)/original_len*100:.1f}%)")
    return clean_df, flagged_df


# ── Step 3: Feature engineering ──────────────────────────────────────────────
def engineer_features(df: pd.DataFrame):
    log("INFO", "Engineering features for ML model")

    # Revenue buckets (log-scaled, 0 if no revenue data)
    df["log_revenue"] = np.log1p(df["revenue"])

    # Rating normalized 0-1
    df["rating_norm"] = (df["vote_average"] - 1) / 9.0

    # Runtime buckets: short(<90), medium(90-120), long(>120)
    df["runtime_bucket"] = pd.cut(
        df["runtime_min"],
        bins=[0, 90, 120, 300],
        labels=["short", "medium", "long"]
    ).astype(str)

    # Decade feature
    df["decade"] = (df["release_year"] // 10) * 10

    # Genre count
    df["genre_count"] = df["genres_clean"].apply(len)

    # One-hot encode runtime bucket
    runtime_dummies = pd.get_dummies(df["runtime_bucket"], prefix="runtime")
    df = pd.concat([df, runtime_dummies], axis=1)

    log("OK", f"Features engineered: {df.shape[1]} columns total")
    return df


# ── Step 4: Save processed data ──────────────────────────────────────────────
def save(clean_df, flagged_df):
    os.makedirs("data/processed", exist_ok=True)
    clean_df.to_csv(PROC_PATH, index=False)
    flagged_df.to_csv(FLAG_PATH, index=False)
    log("OK", f"Processed data → {PROC_PATH}  ({len(clean_df)} rows)")
    log("OK", f"Flagged data   → {FLAG_PATH}  ({len(flagged_df)} rows)")


# ── Lambda handler entrypoint ─────────────────────────────────────────────────
def lambda_handler(event=None, context=None):
    """
    AWS Lambda handler. In production, `event` contains S3 trigger metadata:
      event = {
        "Records": [{
          "s3": {"bucket": {"name": "cinetag-raw-ingest"},
                 "object": {"key": "batch_0041.json"}}
        }]
      }
    Locally we just pass None and read from disk.
    """
    log("INFO", "Lambda invoked — starting ingestion pipeline")

    raw_df = load_raw()
    clean_df, flagged_df = clean(raw_df)
    clean_df = engineer_features(clean_df)
    save(clean_df, flagged_df)

    summary = {
        "total_ingested":  len(raw_df),
        "total_clean":     len(clean_df),
        "total_flagged":   len(flagged_df),
        "flag_rate_pct":   round(len(flagged_df) / len(raw_df) * 100, 2),
        "columns":         list(clean_df.columns),
    }
    log("OK", f"Lambda complete — {summary['total_clean']} records ready for training")
    return {"statusCode": 200, "body": json.dumps(summary)}


if __name__ == "__main__":
    result = lambda_handler()
    print("\n── Lambda return value ──")
    print(json.dumps(json.loads(result["body"]), indent=2))
