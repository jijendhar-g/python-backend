"""
FastAPI Application — Agricultural Commodity Price Prediction
─────────────────────────────────────────────────────────────
Endpoints:
  GET  /health          → service health check
  POST /train-model     → train the LSTM on CSV data
  POST /predict-price   → get next-day price prediction
  GET  /model-metrics   → retrieve latest evaluation metrics
"""

import json
import os
import logging
import numpy as np
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import (
    API_HOST,
    API_PORT,
    CORS_ORIGINS,
    DATA_DIR,
    METRICS_PATH,
    SEQUENCE_LENGTH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from data_preprocessing import prepare_dataset, load_scaler, normalise_data, create_sequences, load_data, clean_data
from lstm_model import build_model, train_model, load_trained_model, predict
from evaluation import compute_metrics

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────
app = FastAPI(
    title="AgriPrice Prediction API",
    description="LSTM-based agricultural commodity price forecasting backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Request / Response schemas
# ──────────────────────────────────────────────
class TrainRequest(BaseModel):
    """Optional: specify a CSV filename inside data/. Defaults to sample_data.csv."""
    filename: str = Field(default="sample_data.csv", description="CSV file in the data/ directory")
    epochs: Optional[int] = Field(default=None, description="Override default epochs")


class TrainResponse(BaseModel):
    message: str
    metrics: dict
    epochs_run: int


class PredictRequest(BaseModel):
    """
    Provide the last `sequence_length` days of feature values.
    Each inner list should have the same number of elements as FEATURE_COLUMNS.
    Example for 3 features (price, demand, season):
      [[25.5, 120, 1], [26.0, 115, 1], ...]
    """
    sequence: List[List[float]] = Field(
        ...,
        description="Recent price history — list of [price, demand, season] per day",
    )
    commodity: str = Field(default="Tomato", description="Commodity name for context")


class PredictResponse(BaseModel):
    commodity: str
    predicted_price: float
    confidence_note: str


class MetricsResponse(BaseModel):
    mae: float
    rmse: float
    mape: float
    r2_score: float


# ──────────────────────────────────────────────
# Utility: inverse-transform predicted price
# ──────────────────────────────────────────────
def inverse_transform_price(scaled_value: float, scaler) -> float:
    """
    Convert a normalised prediction back to the original price scale.
    We create a dummy row with the prediction in the target column
    and zeros elsewhere, then inverse-transform.
    """
    target_idx = FEATURE_COLUMNS.index(TARGET_COLUMN)
    dummy = np.zeros((1, len(FEATURE_COLUMNS)))
    dummy[0, target_idx] = scaled_value
    inv = scaler.inverse_transform(dummy)
    return float(inv[0, target_idx])


# ──────────────────────────────────────────────
# GET /health
# ──────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """Simple liveness probe."""
    model_exists = os.path.exists(
        os.path.join(os.path.dirname(__file__), "models", "lstm_model.keras")
    )
    return {
        "status": "healthy",
        "model_loaded": model_exists,
        "version": "1.0.0",
    }


# ──────────────────────────────────────────────
# POST /train-model
# ──────────────────────────────────────────────
@app.post("/train-model", response_model=TrainResponse)
async def train_endpoint(req: TrainRequest):
    """
    Train (or retrain) the LSTM model on the specified CSV.
    Steps:
      1. Load & preprocess data
      2. Build LSTM architecture
      3. Train with early stopping
      4. Evaluate on held-out test set
      5. Return metrics
    """
    file_path = os.path.join(DATA_DIR, req.filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Data file '{req.filename}' not found in data/")

    try:
        logger.info(f"Starting training on {req.filename}")

        # 1 — Preprocess
        X_train, X_test, y_train, y_test, scaler = prepare_dataset(file_path)
        logger.info(
            f"Data ready — train samples: {len(X_train)}, test samples: {len(X_test)}"
        )

        # 2 — Build model
        input_shape = (X_train.shape[1], X_train.shape[2])  # (seq_len, features)
        model = build_model(input_shape)

        # 3 — Train
        from config import EPOCHS
        epochs = req.epochs or EPOCHS
        history = train_model(model, X_train, y_train)
        epochs_run = len(history["loss"])

        # 4 — Evaluate on test set
        y_pred_scaled = predict(model, X_test)

        # Inverse-transform to original price scale
        y_test_real = np.array([inverse_transform_price(v, scaler) for v in y_test])
        y_pred_real = np.array([inverse_transform_price(v, scaler) for v in y_pred_scaled])

        metrics = compute_metrics(y_test_real, y_pred_real)
        logger.info(f"Training complete — metrics: {metrics}")

        return TrainResponse(
            message="Model trained successfully",
            metrics=metrics,
            epochs_run=epochs_run,
        )

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────
# POST /predict-price
# ──────────────────────────────────────────────
@app.post("/predict-price", response_model=PredictResponse)
async def predict_endpoint(req: PredictRequest):
    """
    Predict next-day price for a commodity.
    The client sends the last `sequence_length` days of features.
    """
    try:
        model = load_trained_model()
        scaler = load_scaler()
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Validate input shape
    if len(req.sequence) != SEQUENCE_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {SEQUENCE_LENGTH} time steps, got {len(req.sequence)}",
        )
    if any(len(row) != len(FEATURE_COLUMNS) for row in req.sequence):
        raise HTTPException(
            status_code=422,
            detail=f"Each time step must have {len(FEATURE_COLUMNS)} features: {FEATURE_COLUMNS}",
        )

    try:
        # Normalise the input using the saved scaler
        raw = np.array(req.sequence, dtype="float32")
        scaled = scaler.transform(raw)

        # Reshape for model: (1, seq_len, features)
        X_input = scaled.reshape(1, SEQUENCE_LENGTH, len(FEATURE_COLUMNS))

        # Predict (normalised)
        pred_scaled = predict(model, X_input)[0]

        # Inverse-transform to original scale
        predicted_price = inverse_transform_price(pred_scaled, scaler)

        return PredictResponse(
            commodity=req.commodity,
            predicted_price=round(predicted_price, 2),
            confidence_note="Prediction based on LSTM model with historical trend analysis",
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────
# GET /model-metrics
# ──────────────────────────────────────────────
@app.get("/model-metrics", response_model=MetricsResponse)
async def metrics_endpoint():
    """Return the evaluation metrics from the last training run."""
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(
            status_code=404,
            detail="No metrics available. Train the model first via POST /train-model.",
        )

    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    return MetricsResponse(**metrics)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting AgriPrice API on {API_HOST}:{API_PORT}")
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)
