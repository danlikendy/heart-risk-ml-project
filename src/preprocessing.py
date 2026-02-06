"""Preprocessing pipeline for heart risk data (OOP)"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib

from .config import (
    TARGET_COL,
    ID_COL,
    DROP_COLS,
    CATEGORICAL_COLS,
    NUMERIC_FILL_COLS,
)


class HeartRiskPreprocessor:
    """
    Fits on training data (learns fill values and encodings),
    transforms train or test data for model input.
    """

    def __init__(self) -> None:
        self._numeric_medians: dict[str, float] = {}
        self._gender_map: dict[str, int] = {}
        self._diet_map: dict[int, int] = {}
        self._feature_columns: Optional[list[str]] = None

    def fit(self, df: pd.DataFrame) -> "HeartRiskPreprocessor":
        """Learn medians and encodings from training data (must include target for column alignment)."""
        # Work on a copy; ensure we have target for column set consistency
        work = df.copy()

        # Fill numeric missing with median
        for col in NUMERIC_FILL_COLS:
            if col in work.columns:
                self._numeric_medians[col] = work[col].median()
                work[col] = work[col].fillna(self._numeric_medians[col])

        # Encode Gender
        if "Gender" in work.columns:
            uniq = work["Gender"].dropna().unique().tolist()
            self._gender_map = {v: i for i, v in enumerate(sorted(uniq))}
            work["Gender"] = work["Gender"].map(self._gender_map).fillna(-1).astype(int)

        # Encode Diet (already numeric; map to contiguous if needed)
        if "Diet" in work.columns:
            uniq = sorted(work["Diet"].dropna().unique().tolist())
            self._diet_map = {v: i for i, v in enumerate(uniq)}
            work["Diet"] = work["Diet"].map(self._diet_map).fillna(-1).astype(int)

        # Define feature columns (all used for modeling, excluding drop cols)
        all_cols = [c for c in work.columns if c not in DROP_COLS]
        self._feature_columns = sorted(all_cols)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw DataFrame to model-ready features (same order as fit)."""
        work = df.copy()

        for col in NUMERIC_FILL_COLS:
            if col in work.columns:
                median = self._numeric_medians.get(col, work[col].median())
                work[col] = work[col].fillna(median)

        if "Gender" in work.columns:
            work["Gender"] = work["Gender"].map(self._gender_map).fillna(-1).astype(int)

        if "Diet" in work.columns:
            work["Diet"] = work["Diet"].map(self._diet_map).fillna(-1).astype(int)

        if self._feature_columns is None:
            raise RuntimeError("Preprocessor must be fitted before transform.")
        # Ensure column order and only feature columns
        for c in self._feature_columns:
            if c not in work.columns:
                work[c] = 0
        return work[self._feature_columns].astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on data and return transformed features."""
        return self.fit(df).transform(df)

    @property
    def feature_columns(self) -> list[str]:
        if self._feature_columns is None:
            raise RuntimeError("Preprocessor not fitted.")
        return self._feature_columns.copy()

    def save(self, path: Optional[Path] = None) -> None:
        from .config import PREPROCESSOR_PATH
        p = path or PREPROCESSOR_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "HeartRiskPreprocessor":
        from .config import PREPROCESSOR_PATH
        p = path or PREPROCESSOR_PATH
        return joblib.load(p)