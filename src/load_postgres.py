"""Load the prepared feature table into PostgreSQL using DATABASE_URL from .env."""
from __future__ import annotations
import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from src.config import FEATURE_DATA_PATH


def main() -> None:
    load_dotenv()
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise SystemExit("DATABASE_URL is not set. Copy .env.example to .env and configure PostgreSQL first.")
    if not FEATURE_DATA_PATH.exists():
        raise SystemExit("Feature data not found. Run `python -m src.prepare_data` first.")
    df = pd.read_csv(FEATURE_DATA_PATH)
    engine = create_engine(database_url, pool_pre_ping=True)
    # Verify the database is reachable before writing.
    with engine.begin() as connection:
        connection.execute(text("SELECT 1"))
    # This replaces only this project-owned table on each run for reproducibility.
    # Do not point DATABASE_URL at a shared/production database.
    df.to_sql("telco_customer_features", engine, if_exists="replace", index=False, chunksize=1000, method="multi")
    with engine.connect() as connection:
        count = connection.execute(text("SELECT COUNT(*) FROM telco_customer_features")).scalar_one()
    print(f"Loaded {count:,} rows into PostgreSQL table public.telco_customer_features")
    print("WARNING: if_exists='replace' recreates this table. Use a dedicated local project database only.")
    engine.dispose()

if __name__ == "__main__":
    main()
