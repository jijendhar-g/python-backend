"""
Data Preprocessing Module
─────────────────────────
Handles loading, cleaning, normalising, and windowing of
agricultural commodity price data for LSTM training.

Flow:
  raw CSV → clean → normalise (Min-Max) → sliding windows → (X, y) arrays
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional

from config import (
    DATA_DIR,
    SCALER_PATH,
    SEQUENCE_LENGTH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TEST_SPLIT,
)


# ──────────────────────────────────────────────
# 1. Load raw data
# ──────────────────────────────────────────────
def load_data(file_path: str) -> pd.DataFrame:
    """Read CSV and parse the date column."""
    df = pd.read_csv(file_path, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ──────────────────────────────────────────────
# 2. Clean data
# ──────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values and outliers.
    - Forward-fill then backward-fill NaNs.
    - Cap outliers at 1st / 99th percentiles (IQR-based).
    """
    # Fill missing values
    df = df.ffill().bfill()

    # Remove outliers using IQR on numeric feature columns
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.01)
        q99 = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=q1, upper=q99)

    return df


# ──────────────────────────────────────────────
# 3. Normalise (Min-Max scaling to [0, 1])
# ──────────────────────────────────────────────
def normalise_data(
    df: pd.DataFrame,
    fit: bool = True,
    scaler: Optional[MinMaxScaler] = None,
) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Apply Min-Max normalisation to the feature columns.
    If `fit=True`, a new scaler is fitted and saved to disk.
    Otherwise the provided scaler is used (inference mode).
    """
    features = df[FEATURE_COLUMNS].values.astype("float32")

    if fit:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(features)
        # Persist scaler for later inference
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided when fit=False")
        scaled = scaler.transform(features)

    return scaled, scaler


def load_scaler() -> MinMaxScaler:
    """Load a previously fitted scaler from disk."""
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────
# 4. Create sliding-window sequences
# ──────────────────────────────────────────────
def create_sequences(
    data: np.ndarray,
    seq_length: int = SEQUENCE_LENGTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding window technique:
    Given an array of shape (T, F), produce:
      X: (T - seq_length, seq_length, F)   — input windows
      y: (T - seq_length,)                  — next-step target (price)
    
    The target is always the first feature column (price)
    because FEATURE_COLUMNS[0] == TARGET_COLUMN.
    """
    X, y = [], []
    target_idx = FEATURE_COLUMNS.index(TARGET_COLUMN)

    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])        # window of seq_length days
        y.append(data[i + seq_length, target_idx]) # next day's price

    return np.array(X), np.array(y)


# ──────────────────────────────────────────────
# 5. Full pipeline: CSV → train/test arrays
# ──────────────────────────────────────────────
def prepare_dataset(
    file_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    End-to-end preprocessing pipeline.
    Returns X_train, X_test, y_train, y_test, scaler.
    """
    df = load_data(file_path)
    df = clean_data(df)
    scaled, scaler = normalise_data(df, fit=True)
    X, y = create_sequences(scaled)

    # Chronological train/test split (no shuffle for time-series)
    split_idx = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test, scaler
