from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    DATA_DIR,
    HISTORY_MINUTES,
)
from .feature_pipeline import (
    generate_enhanced_features,
    create_model_window,
)
from .predictor import NetworkAttackPredictor


# =============================================================================
# APPLICATION
# =============================================================================

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
)


# =============================================================================
# MODEL
# =============================================================================

predictor = NetworkAttackPredictor()


# =============================================================================
# REQUEST MODELS
# =============================================================================

class ForecastRequest(BaseModel):
    """
    Request containing a CSV file path for network-state data.

    The CSV must contain enough chronological network-state rows to construct
    the required 10-minute history window.
    """

    file_path: str = Field(
        ...,
        description=(
            "Path to a network_states.csv-compatible file."
        ),
    )


# =============================================================================
# RESPONSE MODEL
# =============================================================================

class ForecastResponse(BaseModel):
    forecast_horizon_minutes: int
    attack_probability: float
    predicted_attack: bool
    decision_threshold: float
    risk_level: str


# =============================================================================
# ROOT ENDPOINT
# =============================================================================

@app.get("/")
def root() -> Dict[str, Any]:
    """
    Basic API information.
    """

    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "status": "online",
        "forecast_horizon_minutes": 15,
        "history_minutes": HISTORY_MINUTES,
        "model_feature_count": 161,
    }


# =============================================================================
# HEALTH ENDPOINT
# =============================================================================

@app.get("/health")
def health() -> Dict[str, Any]:
    """
    Check whether the forecasting model is loaded correctly.
    """

    return {
        "status": "healthy",
        "model": predictor.check_model(),
    }


# =============================================================================
# FORECAST FROM CSV
# =============================================================================

@app.post(
    "/forecast",
    response_model=ForecastResponse,
)
def forecast(
    request: ForecastRequest,
) -> ForecastResponse:
    """
    Generate a network attack forecast from a CSV file.

    Processing pipeline:

        CSV
          ↓
        network states
          ↓
        enhanced temporal features
          ↓
        latest 10-minute window
          ↓
        XGBoost
          ↓
        15-minute attack forecast
    """

    try:

        # ---------------------------------------------------------------------
        # Validate input path
        # ---------------------------------------------------------------------

        input_file = request.file_path

        if not input_file:
            raise HTTPException(
                status_code=400,
                detail="file_path cannot be empty.",
            )

        # ---------------------------------------------------------------------
        # Load CSV
        # ---------------------------------------------------------------------

        try:
            df = pd.read_csv(input_file)

        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"Input file not found: {input_file}",
            )

        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Could not read CSV file: {exc}",
            )

        # ---------------------------------------------------------------------
        # Validate number of rows
        # ---------------------------------------------------------------------

        if len(df) < HISTORY_MINUTES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"At least {HISTORY_MINUTES} rows are required "
                    f"to create a forecast window. "
                    f"Received {len(df)} rows."
                ),
            )

        # ---------------------------------------------------------------------
        # Generate enhanced features
        # ---------------------------------------------------------------------

        enhanced_df = generate_enhanced_features(df)

        # ---------------------------------------------------------------------
        # Create latest history window
        # ---------------------------------------------------------------------

        window = create_model_window(
            enhanced_df
        )

        # ---------------------------------------------------------------------
        # Generate prediction
        # ---------------------------------------------------------------------

        result = predictor.predict(
            window
        )

        # ---------------------------------------------------------------------
        # Return prediction
        # ---------------------------------------------------------------------

        return ForecastResponse(
            **result
        )

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Forecast generation failed: {exc}",
        )


# =============================================================================
# MODEL INFORMATION
# =============================================================================

@app.get("/model")
def model_information() -> Dict[str, Any]:
    """
    Return information about the currently loaded forecasting model.
    """

    return predictor.check_model()