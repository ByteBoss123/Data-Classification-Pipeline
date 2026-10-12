# CineTag: Movie Genre Tagging Pipeline (AWS-shaped, runs locally)

An end-to-end multi-label classification pipeline laid out the way it would run on AWS: an ingest step written as a
Lambda handler, a training script written as a SageMaker entry point, a post-prediction validation layer, and an
endpoint wrapper with SageMaker's request/response contract. Locally each AWS piece is simulated with files.

```
generate data → Lambda-style ingest (clean, flag bad rows) → Random Forest training
             → validation layer (confidence thresholds, genre-conflict rules) → endpoint wrapper → batch predictions
```

## Data: synthetic, and why the scores mean little

`data/generate_dataset.py` takes **25 real films** (title, year, genres, rating, runtime, revenue, director) and
resamples them into 500 rows with small random noise, plus deliberately corrupted rows (5% missing genres, 8% with
an "Unknown" label) for the ingest checks to catch. The train/test split is random, so the same 25 films appear on
both sides.

Latest local run (`python run_pipeline.py`, October 2026):

| Step | Result |
|---|---|
| Ingest | 500 rows, 29 flagged (5.8%), 471 kept |
| Model | F1 macro 71.4%, exact-match accuracy 47.4% |
| Validation layer | 12 conflicting labels suppressed; consistency score 65.2% |

Because the test rows are noisy copies of training films, these numbers measure recall of 25 memorized films, not
genre prediction. The batch demo shows it: *Oppenheimer* comes back as Action/Adventure/Science Fiction and *The
Killer* as Drama/Romance. Earlier output also printed "+20% vs baseline" and "−25% vs baseline"; those were
hard-coded strings, not measurements, and have been removed.

## What the repo does show

Pipeline structure, data-quality gating at ingest, a validation layer between model and output, and code shaped for
Lambda and SageMaker deployment. To make the model meaningful: train on the real TMDb 5000 dataset, use plot
overviews as features, and split by film.

## Run

```
pip install scikit-learn pandas numpy
python run_pipeline.py      # ~15 s, no AWS account needed
```
