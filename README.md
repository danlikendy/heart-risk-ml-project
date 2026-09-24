# Heart risk, as a service

Binary classifier for heart-attack risk on clinical tabular data. Not a notebook dump: preprocessor and model are objects, artifacts are files, inference is a FastAPI service (path or upload → JSON).

Live write-up (static, no backend): **[danlikendy.github.io/heart-risk-ml-project](https://danlikendy.github.io/heart-risk-ml-project/)**

This is **not** a diagnostic product. It scores a CSV the way a screening model would sit behind a hospital batch job.

---

## Problem

Train: **8 685** patients, **26** clinical fields (labs, vitals, habits, income). Target: high vs low attack risk. **~35% / 65%** split. Nine columns missing on the same **243** rows — classic “row didn’t get the questionnaire” hole, not MCAR noise you can ignore.

If you optimize accuracy you mostly predict “low”. Screening cares about the minority class. I used **F1-macro**.

## What I shipped

| Piece | Choice |
|---|---|
| Model | CatBoost, `depth=6`, `lr=0.05`, 500 iterations, `auto_class_weights=Balanced` |
| Preprocess | Median fill on numerics; `Gender` / `Diet` maps fitted on train; frozen column order |
| Validation | Stratified 5-fold. **CV F1-macro 0.54 ± 0.008** |
| Serve | FastAPI: `POST /predict` (path) and `POST /predict/upload` |
| Batch | `python scripts/generate_predictions.py` → `predictions.csv` (`id`, `prediction`) |

In-sample classification report looks ~0.90. I don’t put that number in a slide. Trees memorize this table; CV is the number I stand behind. The interesting part of the repo is the **pipeline you can actually call**, not a leaderboard screenshot.

Hold-out labels for the 966-row test file are not in git (contest setup). Test predictions: **380 / 966** flagged high (~39%), close to the train base rate — the service isn’t collapsing to the majority class.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/train.py                 # artifacts/preprocessor.joblib + model
python scripts/generate_predictions.py  # predictions.csv
uvicorn src.app:app --reload            # http://127.0.0.1:8000
```

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"csv_path":"heart_test.csv"}'
```

Tests: `pytest tests/` (skips inference if you haven’t trained yet).

More: [docs/RUN.md](docs/RUN.md) · [docs/API.md](docs/API.md)

## Layout

```
src/           preprocessor, model wrapper, pipeline, FastAPI
scripts/       train / predict / evaluate
notebooks/     EDA + the same training path as scripts/train.py
tests/
heart_*.csv    train (with target) / test (features only)
predictions.csv
```

`artifacts/` is gitignored. Train locally; don’t commit `.cbm` / `.joblib`.

---

Artem Tsygantsov · [tsygantsov.ru](https://tsygantsov.ru) · MIT
