"""Train, compare, and evaluate customer churn models for DS-01 Part 2."""
from __future__ import annotations
from src.churn_model import read_feature_data, train_and_evaluate


def main() -> None:
    df = read_feature_data()
    train_and_evaluate(df)


if __name__ == "__main__":
    main()
