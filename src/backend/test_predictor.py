from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from .config import (
    DATA_DIR,
    FEATURE_COUNT,
    FORECAST_HORIZON_MINUTES,
    HISTORY_MINUTES,
    MODEL_FILE,
    MODEL_THRESHOLD,
)
from .feature_pipeline import (
    create_model_window,
    generate_enhanced_features,
)
from .main import app
from .predictor import NetworkAttackPredictor


# ============================================================
# TEST INPUT
# ============================================================

INPUT_FILE = (
    DATA_DIR
    / "temporal"
    / "network_states.csv"
)


# ============================================================
# API TEST CLIENT
# ============================================================

client = TestClient(app)


# ============================================================
# TEST HELPERS
# ============================================================

def load_test_window() -> np.ndarray:
    """
    Load the canonical temporal network-state dataset,
    generate the enhanced features, select one source,
    and create the final inference window.
    """

    assert INPUT_FILE.exists(), (
        f"Network-state file not found: {INPUT_FILE}"
    )

    df = pd.read_csv(INPUT_FILE)

    assert len(df) >= HISTORY_MINUTES, (
        f"Expected at least {HISTORY_MINUTES} rows, "
        f"got {len(df)}"
    )

    enhanced_df = generate_enhanced_features(df)

    assert "Source_File" in enhanced_df.columns
    assert "Minute" in enhanced_df.columns

    sources = (
        enhanced_df["Source_File"]
        .dropna()
        .unique()
    )

    assert len(sources) > 0, (
        "No Source_File values found."
    )

    source_df = (
        enhanced_df[
            enhanced_df["Source_File"] == sources[0]
        ]
        .sort_values("Minute")
        .reset_index(drop=True)
    )

    assert len(source_df) >= HISTORY_MINUTES, (
        "Not enough states for a model window."
    )

    window = create_model_window(source_df)

    return window


# ============================================================
# MODEL FILE TESTS
# ============================================================

def test_model_file_exists():
    """The configured XGBoost model file must exist."""

    assert MODEL_FILE.exists(), (
        f"Model file not found: {MODEL_FILE}"
    )


def test_predictor_loads_model():
    """The forecasting predictor must load successfully."""

    predictor = NetworkAttackPredictor()

    info = predictor.check_model()

    assert info["loaded"] is True
    assert info["model_type"] == "XGBoost"
    assert info["feature_count"] == FEATURE_COUNT
    assert info["flattened_feature_count"] == (
        HISTORY_MINUTES * FEATURE_COUNT
    )
    assert info["history_minutes"] == HISTORY_MINUTES
    assert info["forecast_horizon_minutes"] == (
        FORECAST_HORIZON_MINUTES
    )
    assert info["decision_threshold"] == MODEL_THRESHOLD


# ============================================================
# FEATURE / WINDOW TESTS
# ============================================================

def test_model_window_shape():
    """The inference window must have the expected shape."""

    window = load_test_window()

    assert isinstance(window, np.ndarray)

    assert window.shape == (
        HISTORY_MINUTES,
        FEATURE_COUNT,
    )


def test_model_window_is_finite():
    """The final inference window must contain finite values."""

    window = load_test_window()

    assert np.isfinite(window).all()


# ============================================================
# PREDICTION TESTS
# ============================================================

def test_prediction_returns_expected_fields():
    """The predictor must return all required forecast fields."""

    predictor = NetworkAttackPredictor()
    window = load_test_window()

    result = predictor.predict(window)

    expected_fields = {
        "forecast_horizon_minutes",
        "attack_probability",
        "predicted_attack",
        "decision_threshold",
        "risk_level",
    }

    assert expected_fields.issubset(result.keys())


def test_prediction_probability_is_valid():
    """Attack probability must be between 0 and 1."""

    predictor = NetworkAttackPredictor()
    window = load_test_window()

    result = predictor.predict(window)

    probability = result["attack_probability"]

    assert isinstance(probability, float)
    assert 0.0 <= probability <= 1.0


def test_prediction_threshold_is_correct():
    """The configured decision threshold must be returned."""

    predictor = NetworkAttackPredictor()
    window = load_test_window()

    result = predictor.predict(window)

    assert result["decision_threshold"] == MODEL_THRESHOLD


def test_prediction_horizon_is_correct():
    """The configured forecast horizon must be returned."""

    predictor = NetworkAttackPredictor()
    window = load_test_window()

    result = predictor.predict(window)

    assert result["forecast_horizon_minutes"] == (
        FORECAST_HORIZON_MINUTES
    )


def test_prediction_is_boolean():
    """The attack prediction must be a boolean."""

    predictor = NetworkAttackPredictor()
    window = load_test_window()

    result = predictor.predict(window)

    assert isinstance(
        result["predicted_attack"],
        bool,
    )


def test_risk_level_is_valid():
    """The predictor must return a recognized risk level."""

    predictor = NetworkAttackPredictor()
    window = load_test_window()

    result = predictor.predict(window)

    valid_levels = {
        "MINIMAL",
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL",
    }

    assert result["risk_level"] in valid_levels


# ============================================================
# FASTAPI ROOT TEST
# ============================================================

def test_api_root():
    """The root endpoint must report the API as online."""

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["name"] == "Network Attack Forecasting API"
    assert data["version"] == "1.0.0"
    assert data["status"] == "online"
    assert data["history_minutes"] == HISTORY_MINUTES
    assert data["forecast_horizon_minutes"] == (
        FORECAST_HORIZON_MINUTES
    )
    assert data["model_feature_count"] == FEATURE_COUNT


# ============================================================
# FASTAPI HEALTH TEST
# ============================================================

def test_api_health():
    """The health endpoint must confirm the model is loaded."""

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"

    model = data["model"]

    assert model["loaded"] is True
    assert model["model_type"] == "XGBoost"
    assert model["feature_count"] == FEATURE_COUNT
    assert model["flattened_feature_count"] == (
        HISTORY_MINUTES * FEATURE_COUNT
    )


# ============================================================
# FASTAPI MODEL INFORMATION TEST
# ============================================================

def test_api_model_information():
    """The model endpoint must expose the expected configuration."""

    response = client.get("/model")

    assert response.status_code == 200

    data = response.json()

    assert data["loaded"] is True
    assert data["model_type"] == "XGBoost"
    assert data["feature_count"] == FEATURE_COUNT
    assert data["flattened_feature_count"] == (
        HISTORY_MINUTES * FEATURE_COUNT
    )
    assert data["history_minutes"] == HISTORY_MINUTES
    assert data["forecast_horizon_minutes"] == (
        FORECAST_HORIZON_MINUTES
    )
    assert data["decision_threshold"] == MODEL_THRESHOLD


# ============================================================
# FASTAPI FORECAST TEST
# ============================================================

def test_api_forecast():
    """
    The complete forecast endpoint must process the
    network-state CSV and return a valid prediction.
    """

    response = client.post(
        "/forecast",
        json={
            "file_path": str(INPUT_FILE),
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["forecast_horizon_minutes"] == (
        FORECAST_HORIZON_MINUTES
    )

    assert 0.0 <= data["attack_probability"] <= 1.0

    assert isinstance(
        data["predicted_attack"],
        bool,
    )

    assert data["decision_threshold"] == MODEL_THRESHOLD

    assert data["risk_level"] in {
        "MINIMAL",
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL",
    }


# ============================================================
# FASTAPI ERROR TEST
# ============================================================

def test_api_forecast_missing_file():
    """The forecast endpoint must reject a missing input file."""

    missing_file = (
        DATA_DIR
        / "temporal"
        / "does_not_exist.csv"
    )

    response = client.post(
        "/forecast",
        json={
            "file_path": str(missing_file),
        },
    )

    assert response.status_code == 404

    data = response.json()

    assert "Input file not found" in data["detail"]