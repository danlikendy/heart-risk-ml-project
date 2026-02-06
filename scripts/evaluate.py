"""Evaluation script: compares student predictions to correct answers."""
import argparse
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.metrics import classification_report


def main(args):
    corr_df = pd.read_csv(args.correct, index_col=0)
    stud_df = pd.read_csv(args.student, index_col=0)
    assert list(stud_df.columns) == ["id", "prediction"], (
        f"Expected columns ['id', 'prediction'], got {list(stud_df.columns)}"
    )
    assert len(stud_df) == len(corr_df), (
        f"Length mismatch: student={len(stud_df)}, correct={len(corr_df)}"
    )
    print(classification_report(corr_df["prediction"], stud_df["prediction"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predictions against ground truth")
    parser.add_argument("--student", type=str, required=True, help="Path to student predictions CSV")
    parser.add_argument(
        "--correct",
        type=str,
        default="correct_answers.csv",
        help="Path to correct answers CSV",
    )
    arguments = parser.parse_args()
    main(arguments)
