"""Download and load the public IBM Telco Customer Churn sample dataset."""
from __future__ import annotations
import hashlib
import urllib.request
from pathlib import Path
import pandas as pd
from src.config import DATA_URL, EXPECTED_SHA256, RAW_DATA_PATH, REQUIRED_COLUMNS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_raw_dataset(path: Path = RAW_DATA_PATH, force_download: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if force_download or not path.exists() or path.stat().st_size == 0:
        print(f"Downloading the IBM Telco dataset to: {path}")
        request = urllib.request.Request(
            DATA_URL,
            headers={"User-Agent": "DS01-Student-Project/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response, path.open("wb") as out:
                out.write(response.read())
        except Exception as exc:
            if path.exists() and path.stat().st_size == 0:
                path.unlink(missing_ok=True)
            raise RuntimeError(
                "Dataset download failed. Check your internet connection and try again. "
                f"Source: {DATA_URL}. Original error: {exc}"
            ) from exc
    checksum = sha256_file(path)
    print(f"Dataset SHA-256: {checksum}")
    if EXPECTED_SHA256 and checksum != EXPECTED_SHA256:
        print("WARNING: checksum differs from the documented upstream file; inspect the source/version.")
    return path


def load_raw_dataset(path: Path = RAW_DATA_PATH, download_if_missing: bool = True) -> pd.DataFrame:
    if not path.exists():
        if not download_if_missing:
            raise FileNotFoundError(f"Dataset not found at {path}. Run python -m src.download_data first.")
        ensure_raw_dataset(path)
    df = pd.read_csv(path, dtype={"customerID": "string"})
    df.columns = [str(c).strip() for c in df.columns]
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset does not match expected IBM Telco schema. Missing columns: {sorted(missing)}")
    return df
