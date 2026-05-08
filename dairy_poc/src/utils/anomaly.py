"""
Lightweight anomaly detection helpers.

Two approaches provided:
  - zscore_flag  : simple rolling z-score threshold (fast, interpretable)
  - isolation_flag : Isolation Forest wrapper (scikit-learn, better for multivariate)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def zscore_flag(
    series: pd.Series,
    window: int = 60,
    threshold: float = 3.0,
) -> pd.Series:
    """Return boolean Series where rolling z-score exceeds threshold."""
    roll = series.rolling(window, min_periods=1)
    z = (series - roll.mean()) / roll.std().replace(0, np.nan)
    return z.abs() > threshold


def isolation_flag(
    df: pd.DataFrame,
    feature_cols: list[str],
    contamination: float = 0.02,
    random_state: int = 42,
) -> pd.Series:
    """
    Fit an Isolation Forest on `feature_cols` and return a boolean
    anomaly flag Series (True = anomaly).
    """
    X = df[feature_cols].fillna(df[feature_cols].median())
    clf = IsolationForest(contamination=contamination, random_state=random_state)
    preds = clf.fit_predict(X)          # -1 = anomaly, 1 = normal
    return pd.Series(preds == -1, index=df.index, name="anomaly_if")
