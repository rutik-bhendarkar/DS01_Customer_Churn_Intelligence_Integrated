"""Run inference with the saved DS-01 churn pipeline."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from src.churn_model import predict_dataframe
from src.config import FEATURE_DATA_PATH, ROOT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict churn risk for a CSV using the saved DS-01 model.")
    parser.add_argument("--input", type=Path, default=FEATURE_DATA_PATH,
                        help="CSV containing all model feature columns (default: Part 1 feature CSV).")
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "reports" / "churn_predictions.csv",
                        help="Output CSV path (default: reports/churn_predictions.csv).")
    args = parser.parse_args()
    if not args.input.exists():
        parser.error(f"Input CSV not found: {args.input}. Run Part 1 or specify --input.")
    customers = pd.read_csv(args.input)
    predictions = predict_dataframe(customers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(f"Predicted {len(predictions):,} customer rows.")
    print(f"Predicted churn rate: {predictions['predicted_churn'].mean():.2%}")
    print(f"Saved predictions: {args.output.resolve()}")
    print("Note: these are model outputs for investigation, not verified future outcomes.")


if __name__ == "__main__":
    main()
