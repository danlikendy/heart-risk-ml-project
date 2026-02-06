"""Model wrapper for heart risk prediction (load/save, predict)"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from .config import MODEL_PATH, ARTIFACTS_DIR


class HeartRiskModel:
    """Wraps a trained classifier (CatBoost or sklearn-compatible) for prediction."""

    def __init__(self, estimator=None) -> None:
        self._estimator = estimator

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Return binary class predictions (0/1)."""
        if self._estimator is None:
            raise RuntimeError("Model not loaded or fitted.")
        return np.asarray(self._estimator.predict(X), dtype=np.int64)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Return class probabilities if available."""
        if self._estimator is None:
            raise RuntimeError("Model not loaded or fitted.")
        if hasattr(self._estimator, "predict_proba"):
            return self._estimator.predict_proba(X)
        return None

    def save(self, path: Optional[Path] = None) -> None:
        import joblib
        p = path or MODEL_PATH
        p = p if p.suffix in (".joblib", ".pkl", ".cbm") else p.with_suffix(".joblib")
        p.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(self._estimator, "save_model") and p.suffix == ".cbm":
            self._estimator.save_model(str(p))
        else:
            joblib.dump(self._estimator, p)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "HeartRiskModel":
        import joblib
        p = path or MODEL_PATH
        # Prefer CatBoost .cbm if present
        cbm = ARTIFACTS_DIR / "model.cbm"
        if cbm.exists():
            import catboost
            estimator = catboost.CatBoostClassifier()
            estimator.load_model(str(cbm))
            return cls(estimator=estimator)
        if p.suffix == ".cbm":
            import catboost
            estimator = catboost.CatBoostClassifier()
            estimator.load_model(str(p))
        else:
            estimator = joblib.load(p)
        return cls(estimator=estimator)