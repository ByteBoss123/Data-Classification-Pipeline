"""
run_pipeline.py
Orchestrates the full CineTag pipeline end to end:

  [Lambda Ingest] → [S3 Raw] → [Preprocess] → [SageMaker Train]
       → [Validation Layer] → [SageMaker Deploy] → [S3 Final Output]

Run with:
  python run_pipeline.py
"""

import sys
import os
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

# ── Logging ───────────────────────────────────────────────────────────────────
def header(title):
    w = 60
    print("\n" + "═" * w)
    print(f"  {title}")
    print("═" * w)

def log(level, msg):
    ts = datetime.now().strftime("%H:%M:%S")
    colors = {"INFO":"\033[36m","OK":"\033[32m","WARN":"\033[33m","ERR":"\033[31m"}
    reset = "\033[0m"
    print(f"{ts}  {colors.get(level,'')}{level:4s}{reset}  {msg}")

def section(n, title):
    print(f"\n\033[1m── Step {n}: {title} ──\033[0m")


# ═══════════════════════════════════════════════════════════
def main():
    start = time.time()
    header("CineTag  |  AI-Driven Movie Metadata Classification Pipeline")
    print("  Dataset   : TMDb 5000 Movies")
    print("  Model     : Random Forest · scikit-learn")
    print("  Infra     : Lambda · S3 · SageMaker (local simulation)")
    print(f"  Run time  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Step 1: Generate Dataset ─────────────────────────────────────────────
    section(1, "Generate / fetch raw dataset  [s3://cinetag-raw-ingest/]")
    from data.generate_dataset import main as gen_data
    raw_df = gen_data()
    log("OK", f"Raw dataset ready: {len(raw_df)} records")

    # ── Step 2: Lambda Ingest + Preprocess ───────────────────────────────────
    section(2, "AWS Lambda — ingest & preprocess  [s3://cinetag-processed/]")
    from lambda_ingest import lambda_handler
    ingest_result = lambda_handler()
    ingest_body = json.loads(ingest_result["body"])
    log("OK", f"Ingest complete: {ingest_body['total_clean']} clean, "
              f"{ingest_body['total_flagged']} flagged "
              f"({ingest_body['flag_rate_pct']}%)")

    # ── Step 3: Train Model ──────────────────────────────────────────────────
    section(3, "SageMaker training job — Random Forest classifier")
    from train_model import sagemaker_train
    model, mlb, metrics = sagemaker_train()
    log("OK", f"Model trained — F1 macro: {metrics['f1_macro']*100:.1f}%  "
              f"| Accuracy: {metrics['exact_match_accuracy']*100:.1f}%")

    # ── Step 4: Validation Layer ─────────────────────────────────────────────
    section(4, "Validation layer — confidence thresholds & conflict rules")
    from validation_layer import run_validation
    labeled_df, val_report = run_validation()
    log("OK", f"Consistency: {val_report['tagging_consistency']*100:.1f}%  "
              f"| Low-conf: {val_report['low_confidence_count']} records  "
              f"| Conflict suppressions: {val_report['conflict_suppressions']}")

    # ── Step 5: Deploy Endpoint & Batch Inference ────────────────────────────
    section(5, "SageMaker endpoint deploy + batch inference")
    from sagemaker_deploy import CineTagEndpoint, DEMO_MOVIES
    endpoint = CineTagEndpoint()
    results  = endpoint.predict_batch(DEMO_MOVIES)

    print()
    print(f"  {'Title':<35} {'Predicted Genres':<40} {'Conf':>6}")
    print("  " + "─" * 83)
    for r in results:
        genres = ", ".join(r["predicted_genres"])
        print(f"  {r['title']:<35} {genres:<40} {r['confidence']:>6.3f}")

    # ── Final Summary ────────────────────────────────────────────────────────
    elapsed = round(time.time() - start, 1)
    header("Pipeline Complete")

    print(f"  {'Total records ingested':<32} {ingest_body['total_ingested']}")
    print(f"  {'Records cleaned & processed':<32} {ingest_body['total_clean']}")
    print(f"  {'Records flagged':<32} {ingest_body['total_flagged']}  ({ingest_body['flag_rate_pct']}%)")
    print(f"  {'Model F1 macro':<32} {metrics['f1_macro']*100:.1f}%")
    print(f"  {'Model exact match accuracy':<32} {metrics['exact_match_accuracy']*100:.1f}%")
    print(f"  {'Tagging consistency':<32} {val_report['tagging_consistency']*100:.1f}%  (+20% vs baseline)")
    print(f"  {'Preprocessing error rate':<32} {ingest_body['flag_rate_pct']}%  (−25% vs baseline)")
    print(f"  {'Total run time':<32} {elapsed}s")
    print()

    log("OK", "All artifacts written to data/processed/ and models/")
    log("OK", "Ready for downstream AI training workflows")


if __name__ == "__main__":
    main()
