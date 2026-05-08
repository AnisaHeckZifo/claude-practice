"""
Utilities for temporally aligning low-cadence lab samples with
high-cadence process sensor time-series.
"""

from __future__ import annotations

import pandas as pd


def merge_process_lab(
    process_df: pd.DataFrame,
    lab_df: pd.DataFrame,
    process_time_col: str = "timestamp",
    lab_time_col: str = "sample_time",
    direction: str = "nearest",
    tolerance_minutes: int = 30,
) -> pd.DataFrame:
    """
    Merge lab rows onto the nearest process timestamp.

    Returns a copy of process_df with lab columns joined where a match
    falls within `tolerance_minutes`.
    """
    proc = process_df.copy()
    lab = lab_df.copy()

    proc[process_time_col] = pd.to_datetime(proc[process_time_col])
    lab[lab_time_col]      = pd.to_datetime(lab[lab_time_col])

    proc = proc.sort_values(process_time_col)
    lab  = lab.sort_values(lab_time_col).rename(columns={lab_time_col: process_time_col})

    tolerance = pd.Timedelta(minutes=tolerance_minutes)
    merged = pd.merge_asof(
        proc,
        lab,
        on=process_time_col,
        direction=direction,
        tolerance=tolerance,
        suffixes=("", "_lab"),
    )
    return merged
