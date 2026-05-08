"""Thin I/O helpers: load raw CSVs, save processed Parquet."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


_ROOT = Path(__file__).parents[2]
RAW_DIR       = _ROOT / "data_raw"
PROCESSED_DIR = _ROOT / "data_processed"


def load_raw(name: str) -> pd.DataFrame:
    """Load a raw CSV by stem name, e.g. 'process_foul'."""
    path = RAW_DIR / f"{name}.csv"
    return pd.read_csv(path, parse_dates=["timestamp"] if "process" in name else ["sample_time"])


def save_processed(df: pd.DataFrame, name: str) -> Path:
    """Write a processed DataFrame to Parquet and return the path."""
    PROCESSED_DIR.mkdir(exist_ok=True)
    path = PROCESSED_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path
