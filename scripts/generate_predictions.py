"""Generate predictions CSV (id, prediction) from test CSV using saved pipeline"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import HeartRiskPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate predictions CSV from test CSV")
    parser.add_argument("test_csv", type=Path, nargs="?", default=PROJECT_ROOT / "heart_test.csv")
    parser.add_argument("-o", "--output", type=Path, default=PROJECT_ROOT / "predictions.csv")
    args = parser.parse_args()
    pipeline = HeartRiskPipeline().load()
    pipeline.save_predictions_to_csv(args.test_csv, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()