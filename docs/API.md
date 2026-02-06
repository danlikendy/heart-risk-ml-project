# API and classes reference

Overview of the main modules, classes, and methods in the Heart Risk application

---

## Package: `src`

### `src.config`

Constants and paths used across the pipeline

| Name | Description |
|------|-------------|
| `TARGET_COL` | Target column name: `"Heart Attack Risk (Binary)"` |
| `ID_COL` | Identifier column: `"id"` |
| `INDEX_COL` | Unnamed index column: `"Unnamed: 0"` |
| `DROP_COLS` | Columns dropped before modeling: index, id, target |
| `CATEGORICAL_COLS` | Columns encoded as categorical: `["Gender", "Diet"]` |
| `NUMERIC_FILL_COLS` | Numeric columns; missing values filled with median |
| `PROJECT_ROOT` | Project root directory (`Path`) |
| `ARTIFACTS_DIR` | Directory for saved model and preprocessor |
| `MODEL_PATH` | Default path for the model file (`.joblib` or `.cbm`) |
| `PREPROCESSOR_PATH` | Default path for the preprocessor (`.joblib`) |
| `PREDICTION_CSV_COLUMNS` | Output CSV columns: `["id", "prediction"]` |

---

### `src.preprocessing`: `HeartRiskPreprocessor`

Preprocessing pipeline: fit on training data, transform train/test

**Constructor**

- `HeartRiskPreprocessor()` — no arguments

**Methods**

| Method | Description |
|--------|-------------|
| `fit(df: pd.DataFrame) -> HeartRiskPreprocessor` | Learn medians and encodings from `df` (must contain target for column set). Returns `self`. |
| `transform(df: pd.DataFrame) -> pd.DataFrame` | Transform raw data to model-ready features (same column order as in `fit`). Raises if not fitted. |
| `fit_transform(df: pd.DataFrame) -> pd.DataFrame` | Equivalent to `fit(df).transform(df)`. |
| `save(path: Optional[Path] = None) -> None` | Serialize preprocessor to disk (default: `PREPROCESSOR_PATH`). |
| `load(path: Optional[Path] = None) -> HeartRiskPreprocessor` | Class method: load preprocessor from disk. |

**Properties**

- `feature_columns: list[str]` — list of feature column names after `fit`

**Behaviour**

- Fills missing numeric columns with medians learned in `fit`
- Encodes `Gender` and `Diet` to integers; unknown categories become `-1`
- Output is a single DataFrame of numeric features in a fixed order

---

### `src.model`: `HeartRiskModel`

Wrapper around a trained classifier (CatBoost or sklearn-compatible) for loading, saving, and predicting

**Constructor**

- `HeartRiskModel(estimator=None)` — `estimator`: fitted classifier or `None` (e.g. when loading later)

**Methods**

| Method | Description |
|--------|-------------|
| `predict(X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray` | Binary class predictions (0/1). |
| `predict_proba(X: ...) -> np.ndarray \| None` | Class probabilities if the estimator supports it; otherwise `None`. |
| `save(path: Optional[Path] = None) -> None` | Save estimator: CatBoost to `.cbm`, others to `.joblib`. |
| `load(path: Optional[Path] = None) -> HeartRiskModel` | Class method: load model from `.cbm` (CatBoost) or `.joblib`. Prefers `artifacts/model.cbm` if it exists. |

---

### `src.pipeline`: `HeartRiskPipeline`

End-to-end pipeline: load preprocessor and model, run prediction from CSV or DataFrame

**Constructor**

- `HeartRiskPipeline(model_path=None, preprocessor_path=None)` — optional paths; defaults from `config`

**Methods**

| Method | Description |
|--------|-------------|
| `load() -> HeartRiskPipeline` | Load preprocessor and model from disk. Returns `self`. |
| `predict_from_csv(csv_path: Union[str, Path]) -> pd.DataFrame` | Read CSV, preprocess, predict. Returns DataFrame with columns `id`, `prediction`. |
| `predict_from_dataframe(df: pd.DataFrame) -> pd.DataFrame` | Preprocess `df` and return DataFrame with `id`, `prediction`. |
| `save_predictions_to_csv(csv_path, output_path) -> Path` | Run `predict_from_csv(csv_path)`, save result to `output_path`, return `output_path`. |

---

### `src.app`: FastAPI application

**Endpoints**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | HTML page with short instructions and CSV upload form. |
| `GET` | `/health` | Health check; returns `{"status": "ok"}`. |
| `POST` | `/predict` | Body: `{"csv_path": "path/to/file.csv"}` (relative to project root or absolute). Returns JSON: `{"predictions": [{"id": int, "prediction": int}, ...]}`. |
| `POST` | `/predict/upload` | Form: file field `file` (CSV). Returns same JSON structure as `/predict`. |
| `GET` | `/docs` | Swagger UI. |

**Usage**

- Start with: `uvicorn src.app:app --reload` (see [RUN.md](RUN.md))
- On first request, the app loads the pipeline from `artifacts/` (lazy load)

---

## Scripts

### `scripts/generate_predictions.py`

- **Usage:** `python scripts/generate_predictions.py [test_csv] [-o output.csv]`
- **Default:** reads `heart_test.csv`, writes `predictions.csv`
- Loads `HeartRiskPipeline`, runs `predict_from_csv`, saves CSV with columns `id`, `prediction`

### `scripts/evaluate.py`

- **Usage:** `python scripts/evaluate.py --student <predictions.csv> [--correct correct_answers.csv]`
- Compares student predictions to ground truth and prints classification report

### `test.py` (project root)

- Provided evaluation script: same idea as `scripts/evaluate.py`; expects CSV with columns `id`, `prediction` (and optional index column)