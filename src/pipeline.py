"""End-to-end pipeline: load CSV, preprocess, predict, output id + prediction"""

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from .config import ID_COL, PREDICTION_CSV_COLUMNS, MODEL_PATH, PREPROCESSOR_PATH
from .preprocessing import HeartRiskPreprocessor
from .model import HeartRiskModel


class HeartRiskPipeline:
    """Loads preprocessor and model, runs prediction from a CSV path or DataFrame."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        preprocessor_path: Optional[Path] = None,
    ) -> None:
        self._model_path = model_path or MODEL_PATH
        self._preprocessor_path = preprocessor_path or PREPROCESSOR_PATH
        self._preprocessor: Optional[HeartRiskPreprocessor] = None
        self._model: Optional[HeartRiskModel] = None

    def load(self) -> "HeartRiskPipeline":
        """Load preprocessor and model from disk."""
        self._preprocessor = HeartRiskPreprocessor.load(self._preprocessor_path)
        self._model = HeartRiskModel.load(self._model_path)
        return self

    def predict_from_csv(
        self,
        csv_path: Union[str, Path],
    ) -> pd.DataFrame:
        """
        Read CSV, preprocess, predict. Returns DataFrame with columns id, prediction.
        """
        df = pd.read_csv(csv_path)
        return self.predict_from_dataframe(df)

    def predict_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess DataFrame and return predictions with ids."""
        if self._preprocessor is None or self._model is None:
            self.load()
        X = self._preprocessor.transform(df)
        preds = self._model.predict(X)
        ids = df[ID_COL].values if ID_COL in df.columns else range(len(df))
        return pd.DataFrame({"id": ids, "prediction": preds}, columns=PREDICTION_CSV_COLUMNS)

    def save_predictions_to_csv(
        self,
        csv_path: Union[str, Path],
        output_path: Union[str, Path],
    ) -> Path:
        """Generate predictions from csv_path and save to output_path (id, prediction)."""
        result = self.predict_from_csv(csv_path)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out, index=True)
        return out