"""FastAPI application: accept path to CSV or file upload, return JSON predictions"""

import io
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .config import PROJECT_ROOT
from .pipeline import HeartRiskPipeline

app = FastAPI(
    title="Heart Risk Prediction API",
    description="Predict heart attack risk (binary) from patient CSV data.",
    version="0.1.0",
)

# Lazy-loaded pipeline (load on first request)
_pipeline: Optional[HeartRiskPipeline] = None


def get_pipeline() -> HeartRiskPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = HeartRiskPipeline().load()
    return _pipeline


class PredictFromPathRequest(BaseModel):
    """Request body: path to CSV file (relative to project root or absolute)."""
    csv_path: str


class PredictionItem(BaseModel):
    id: int
    prediction: int


class PredictResponse(BaseModel):
    """Response: list of id and prediction."""
    predictions: list[PredictionItem]


@app.post("/predict", response_model=PredictResponse)
def predict_from_path(body: PredictFromPathRequest) -> PredictResponse:
    """
    Run prediction on a CSV file given its path.
    Path can be relative to project root or absolute.
    """
    path = Path(body.csv_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    try:
        pipeline = get_pipeline()
        result = pipeline.predict_from_csv(path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    items = [PredictionItem(id=int(r["id"]), prediction=int(r["prediction"])) for _, r in result.iterrows()]
    return PredictResponse(predictions=items)


@app.post("/predict/upload", response_model=PredictResponse)
async def predict_from_upload(file: UploadFile = File(...)) -> PredictResponse:
    """
    Run prediction on an uploaded CSV file.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Expected a CSV file.")
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid CSV: {e}")
    try:
        pipeline = get_pipeline()
        result = pipeline.predict_from_dataframe(df)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    items = [PredictionItem(id=int(r["id"]), prediction=int(r["prediction"])) for _, r in result.iterrows()]
    return PredictResponse(predictions=items)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    """Simple HTML page with instructions and form for CSV upload."""
    return """
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Heart Risk Prediction</title></head>
<body>
  <h1>Heart Risk Prediction API</h1>
  <p>Endpoints:</p>
  <ul>
    <li><b>POST /predict</b> — JSON body: <code>{"csv_path": "heart_test.csv"}</code> (path relative to project root)</li>
    <li><b>POST /predict/upload</b> — upload a CSV file</li>
    <li><b>GET /health</b> — health check</li>
    <li><b>GET /docs</b> — Swagger UI</li>
  </ul>
  <h2>Upload CSV</h2>
  <form action="/predict/upload" method="post" enctype="multipart/form-data">
    <input type="file" name="file" accept=".csv" required />
    <button type="submit">Predict</button>
  </form>
</body>
</html>
"""