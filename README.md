# Heart Risk ML Project

This repository contains an end-to-end solution for predicting binary heart attack risk from patient data (anthropometrics, habits, blood pressure, chronic conditions, blood biochemistry). It includes exploratory analysis, preprocessing, model training (e.g. CatBoost / scikit-learn), a FastAPI application that accepts a path to a test CSV and returns predictions in JSON, and evaluation tooling.

---

## Project structure

```
heart-risk-ml-project/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── heart_train.csv           # Training data (with target)
├── heart_test.csv            # Test data (no target)
├── test.py                   # Script to evaluate predictions (provided)
├── src/                      # Application and library code (OOP)
├── notebooks/                # EDA, experiments, model training
├── scripts/                  # Evaluation and utility scripts
├── tests/                    # Unit and integration tests
├── docs/                     # User guide and API/developer docs
└── data/                     # Optional: processed data / artifacts
```

---

## Setup and run

1. **Clone and install**
   ```bash
   git clone https://github.com/danlikendy/heart-risk-ml-project.git
   cd heart-risk-ml-project
   python -m venv .venv
   source .venv/bin/activate   # or `.venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

2. **Train and export artifacts**  
   Use the notebook in `notebooks/` for EDA, preprocessing, and training; the application in `src/` will load the saved model and preprocessing objects.

3. **Start the API**
   ```bash
   uvicorn src.app:app --reload
   ```
   (Exact module path will be set when the app is implemented; see `docs/` for the final run instructions.)

4. **Get predictions**  
   Send a path to the test CSV (e.g. via POST); the service returns JSON with predictions. Optionally use the provided web UI or a small script that calls the API.

5. **Evaluate predictions**  
   Save your predictions as a CSV with columns `id` and `prediction`, then run:
   ```bash
   python test.py --student path/to/predictions.csv --correct path/to/correct_answers.csv
   ```

---

## Data and target

- **Training**: `heart_train.csv` includes a binary target (e.g. `Heart Attack Risk (Binary)`).
- **Test**: `heart_test.csv` has the same features but no target; predictions must be in a CSV with columns `id` and `prediction`.

---

## Deliverables (as per specification)

- [x] Jupyter notebook: EDA, preprocessing, model training (`notebooks/eda_and_training.ipynb`)
- [x] Application code in `src/` (OOP: `HeartRiskPreprocessor`, `HeartRiskModel`, `HeartRiskPipeline`, FastAPI app)
- [x] Predictions on the test set in CSV format (`predictions.csv` with `id`, `prediction`)
- [x] Run instructions: [docs/RUN.md](docs/RUN.md); optional notebook demo or live demo
- [x] Documentation: [docs/RUN.md](docs/RUN.md) (how to run), [docs/API.md](docs/API.md) (classes and methods)