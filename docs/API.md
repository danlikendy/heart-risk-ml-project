# API

## `HeartRiskPreprocessor`

Fit on train. Median for numeric holes. Integer maps for `Gender` and `Diet` (unknown → `-1`). Feature list is sorted and reused on transform so train/serve don’t drift.

`fit` / `transform` / `fit_transform` / `save` / `load`.

## `HeartRiskModel`

Thin wrapper. Predicts 0/1. Saves sklearn via joblib; CatBoost `.cbm` if you ask for that suffix. `load` prefers `artifacts/model.cbm` when it exists.

## `HeartRiskPipeline`

`load()` then `predict_from_csv` / `predict_from_dataframe` → `id, prediction`.

## HTTP (`src.app`)

Lazy-loads the pipeline on first request.

| Method | Path | Body |
|---|---|---|
| GET | `/` | HTML form |
| GET | `/health` | — |
| POST | `/predict` | `{"csv_path": "..."}` relative to repo root or absolute |
| POST | `/predict/upload` | multipart `file` |

404 if the path is missing. 422 if the CSV can’t be scored.

## CLI

```bash
python scripts/train.py
python scripts/generate_predictions.py [test.csv] [-o predictions.csv]
python scripts/evaluate.py --student predictions.csv --correct correct_answers.csv
```
