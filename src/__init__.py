"""Heart Risk ML — library and API for heart attack risk prediction"""

from .config import (
    TARGET_COL,
    ID_COL,
    ARTIFACTS_DIR,
    MODEL_PATH,
    PREPROCESSOR_PATH,
)
from .preprocessing import HeartRiskPreprocessor
from .model import HeartRiskModel
from .pipeline import HeartRiskPipeline

__version__ = "0.1.0"
__all__ = [
    "HeartRiskPreprocessor",
    "HeartRiskModel",
    "HeartRiskPipeline",
    "TARGET_COL",
    "ID_COL",
    "ARTIFACTS_DIR",
    "MODEL_PATH",
    "PREPROCESSOR_PATH",
]