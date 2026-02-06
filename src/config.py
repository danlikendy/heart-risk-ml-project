"""Configuration and constants for the heart risk prediction pipeline"""

from pathlib import Path

# Column names
TARGET_COL = "Heart Attack Risk (Binary)"
ID_COL = "id"
INDEX_COL = "Unnamed: 0"

# Columns to drop before modeling (identifiers / non-features)
DROP_COLS = [INDEX_COL, ID_COL, TARGET_COL]

# Categorical columns (encoded in preprocessing)
CATEGORICAL_COLS = ["Gender", "Diet"]

# Numeric columns (filled with median in preprocessing if missing)
NUMERIC_FILL_COLS = [
    "Age", "Cholesterol", "Heart rate", "Diabetes", "Family History",
    "Smoking", "Obesity", "Alcohol Consumption", "Exercise Hours Per Week",
    "Previous Heart Problems", "Medication Use", "Stress Level",
    "Sedentary Hours Per Day", "Income", "BMI", "Triglycerides",
    "Physical Activity Days Per Week", "Sleep Hours Per Day",
    "Blood sugar", "CK-MB", "Troponin",
    "Systolic blood pressure", "Diastolic blood pressure",
]

# Artifacts (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"

# Output format for predictions CSV
PREDICTION_CSV_COLUMNS = ["id", "prediction"]