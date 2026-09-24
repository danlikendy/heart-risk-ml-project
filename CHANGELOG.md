# Changelog

## 1.0.0 — 2026-09-24

First cut I would actually hand a teammate.

- Pipeline objects in `src/` (preprocessor, model, FastAPI).
- `scripts/train.py` with stratified 5-fold F1-macro (same CatBoost setup as the notebook).
- Batch inference → `predictions.csv`.
- Static project page under `docs/` for GitHub Pages.
- CI: install + pytest (inference tests skip until artifacts exist).
