from src.data_access import ensure_raw_dataset

if __name__ == "__main__":
    path = ensure_raw_dataset()
    print(f"Dataset ready: {path.resolve()}")
