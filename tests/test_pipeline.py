"""Minimal tests for pipeline and preprocessor"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def train_sample():
    path = PROJECT_ROOT / "heart_train.csv"
    if not path.exists():
        pytest.skip("heart_train.csv not found")
    return pd.read_csv(path, nrows=100)


def test_preprocessor_fit_transform(train_sample):
    from src.preprocessing import HeartRiskPreprocessor
    from src.config import TARGET_COL

    preprocessor = HeartRiskPreprocessor()
    X = preprocessor.fit_transform(train_sample)
    assert X.shape[0] == len(train_sample)
    assert X.shape[1] == len(preprocessor.feature_columns)
    assert train_sample[TARGET_COL].shape[0] == X.shape[0]


def test_pipeline_load_and_predict():
    from src.pipeline import HeartRiskPipeline
    from src.config import ARTIFACTS_DIR

    if not (ARTIFACTS_DIR / "preprocessor.joblib").exists():
        pytest.skip("artifacts not found (run training first)")
    pipeline = HeartRiskPipeline().load()
    test_path = PROJECT_ROOT / "heart_test.csv"
    if not test_path.exists():
        pytest.skip("heart_test.csv not found")
    result = pipeline.predict_from_csv(test_path)
    assert list(result.columns) == ["id", "prediction"]
    assert result["prediction"].isin([0, 1]).all()