"""
LSTM Model Module
─────────────────
Builds, trains, saves, and loads the multivariate LSTM model
for agricultural commodity price forecasting.

Architecture:
  Input → LSTM(64) → Dropout(0.2) → LSTM(128) → Dropout(0.2) → Dense(1)

Optimizer : Adam
Loss      : Mean Squared Error (MSE)
"""

import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore

from config import (
    LSTM_UNITS_1,
    LSTM_UNITS_2,
    DROPOUT_RATE,
    EPOCHS,
    BATCH_SIZE,
    VALIDATION_SPLIT,
    MODEL_PATH,
)


# ──────────────────────────────────────────────
# 1. Build the LSTM model
# ──────────────────────────────────────────────
def build_model(input_shape: tuple) -> Sequential:
    """
    Construct a two-layer stacked LSTM with dropout.

    Parameters
    ----------
    input_shape : (sequence_length, num_features)
        Shape of a single input sample.

    Returns
    -------
    Compiled Keras Sequential model.
    """
    model = Sequential([
        # First LSTM layer — returns sequences so the second LSTM can consume them
        Input(shape=input_shape),
        LSTM(LSTM_UNITS_1, return_sequences=True),
        Dropout(DROPOUT_RATE),

        # Second LSTM layer — returns only the last hidden state
        LSTM(LSTM_UNITS_2, return_sequences=False),
        Dropout(DROPOUT_RATE),

        # Dense output — single value (predicted price, normalised)
        Dense(1),
    ])

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.summary()
    return model


# ──────────────────────────────────────────────
# 2. Train
# ──────────────────────────────────────────────
def train_model(
    model: Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict:
    """
    Train the model with early stopping and learning-rate reduction.
    Returns the Keras history dictionary.
    """
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1,
    )

    # Persist trained model
    model.save(MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

    return history.history


# ──────────────────────────────────────────────
# 3. Load a previously trained model
# ──────────────────────────────────────────────
def load_trained_model() -> Sequential:
    """Load the saved Keras model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. "
            "Please train the model first via POST /train-model."
        )
    return load_model(MODEL_PATH)


# ──────────────────────────────────────────────
# 4. Predict
# ──────────────────────────────────────────────
def predict(model: Sequential, X: np.ndarray) -> np.ndarray:
    """
    Run inference on input sequences.

    Parameters
    ----------
    X : array of shape (n_samples, sequence_length, num_features)

    Returns
    -------
    Predicted values (normalised). Shape: (n_samples,)
    """
    predictions = model.predict(X, verbose=0)
    return predictions.flatten()
