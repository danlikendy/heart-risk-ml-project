"""Train preprocessor + CatBoost, print stratified CV F1-macro, write artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import ARTIFACTS_DIR, MODEL_PATH, PREPROCESSOR_PATH, TARGET_COL
from src.model import HeartRiskModel
from src.preprocessing import HeartRiskPreprocessor


def main() -> None:
    train = pd.read_csv(ROOT / "heart_train.csv")
    y = train[TARGET_COL].astype(int).to_numpy()
    preprocessor = HeartRiskPreprocessor().fit(train)
    X = preprocessor.transform(train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores: list[float] = []
    for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
        import catboost as cb

        m = cb.CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            random_seed=42,
            verbose=False,
            auto_class_weights="Balanced",
        )
        m.fit(X.iloc[tr], y[tr])
        pred = m.predict(X.iloc[va])
        score = f1_score(y[va], pred, average="macro")
        fold_scores.append(score)
        print(f"fold {fold}: F1-macro {score:.4f}")

    arr = np.array(fold_scores)
    print(f"CV F1-macro: {arr.mean():.4f} ± {arr.std():.4f}")

    import catboost as cb

    model = cb.CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        random_seed=42,
        verbose=False,
        auto_class_weights="Balanced",
    )
    model.fit(X, y)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    preprocessor.save(PREPROCESSOR_PATH)
    HeartRiskModel(estimator=model).save(MODEL_PATH)
    print(f"saved {PREPROCESSOR_PATH}")
    print(f"saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
