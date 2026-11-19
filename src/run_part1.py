"""Run all non-database Part 1 steps in the right order."""
from src.data_access import ensure_raw_dataset as ensure_data
from src.prepare_data import main as prepare_main
from src.run_eda import main as eda_main

if __name__ == "__main__":
    ensure_data()
    prepare_main()
    eda_main()
    print("\nPART 1 PIPELINE COMPLETE")
    print("To load into PostgreSQL, start PostgreSQL, configure .env, then run: python -m src.load_postgres")
