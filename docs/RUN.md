# Run

Python 3.10+. From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

Writes `artifacts/preprocessor.joblib` and `artifacts/model.joblib`.

```bash
python scripts/train.py
```

Same hyperparameters as the notebook (CatBoost, balanced class weights, stratified 5-fold printed to stdout).

## Infer without the API

```bash
python scripts/generate_predictions.py heart_test.csv -o predictions.csv
```

## API

Needs artifacts on disk.

```bash
uvicorn src.app:app --reload --host 127.0.0.1 --port 8000
```

| | |
|---|---|
| Upload form | http://127.0.0.1:8000/ |
| OpenAPI | http://127.0.0.1:8000/docs |
| Health | `GET /health` |
| Path | `POST /predict` `{"csv_path":"heart_test.csv"}` |
| File | `POST /predict/upload` field `file` |

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"csv_path":"heart_test.csv"}'

curl -s -X POST http://127.0.0.1:8000/predict/upload \
  -F 'file=@heart_test.csv'
```

Response: `{"predictions":[{"id":7746,"prediction":0}, ...]}`.

## Score against labels

If you have `correct_answers.csv` with `id,prediction`:

```bash
python test.py --student predictions.csv --correct correct_answers.csv
```

Classes: [API.md](API.md).
