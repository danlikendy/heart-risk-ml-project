# How to run the Heart Risk application

## Prerequisites

- Python 3.10+
- Project dependencies (see below)

## 1. Clone and install

```bash
git clone https://github.com/danlikendy/heart-risk-ml-project.git
cd heart-risk-ml-project
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Train the model (if artifacts are missing)

If the `artifacts/` folder does not contain a saved model and preprocessor:

- Open and run the Jupyter notebook: `notebooks/eda_and_training.ipynb`  
  (EDA, preprocessing, training, saving to `artifacts/`, and optional test predictions), **or**
- Run training from the project root (example with sklearn):

```bash
python -c "
import sys; from pathlib import Path; sys.path.insert(0, str(Path.cwd()))
import pandas as pd
from src.preprocessing import HeartRiskPreprocessor
from src.config import TARGET_COL, ARTIFACTS_DIR, MODEL_PATH, PREPROCESSOR_PATH
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
train = pd.read_csv('heart_train.csv')
preprocessor = HeartRiskPreprocessor().fit(train)
preprocessor.save(PREPROCESSOR_PATH)
from sklearn.ensemble import RandomForestClassifier
from src.model import HeartRiskModel
X = preprocessor.transform(train)
y = train[TARGET_COL].astype(int)
model = HeartRiskModel(estimator=RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42).fit(X, y))
model.save(MODEL_PATH)
print('Artifacts saved.')
"
```

After training, `artifacts/` will contain `preprocessor.joblib` and `model.joblib` (or `model.cbm` if using CatBoost in the notebook)

## 3. Start the API

From the project root (with the virtualenv activated):

```bash
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

- API: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- Simple HTML page: `http://localhost:8000/`

## 4. Get predictions

### Option A: Path to CSV (POST JSON)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"csv_path": "heart_test.csv"}'
```

`csv_path` can be relative to the project root or absolute

### Option B: Upload CSV file

```bash
curl -X POST http://localhost:8000/predict/upload \
  -F "file=@heart_test.csv"
```

### Option C: Web form

Open `http://localhost:8000/` in a browser and use the upload form

### Response format

JSON with a list of objects `id` and `prediction` (0 or 1):

```json
{
  "predictions": [
    {"id": 7746, "prediction": 0},
    {"id": 4202, "prediction": 1}
  ]
}
```

## 5. Generate predictions CSV (id, prediction)

Without starting the API:

```bash
python scripts/generate_predictions.py heart_test.csv -o predictions.csv
```

Or with defaults (reads `heart_test.csv`, writes `predictions.csv`):

```bash
python scripts/generate_predictions.py
```

## 6. Evaluate predictions (with ground truth)

If you have `correct_answers.csv` with the true labels:

```bash
python test.py --student predictions.csv --correct correct_answers.csv
```

---

For a description of the application classes and methods, see [API.md](API.md)