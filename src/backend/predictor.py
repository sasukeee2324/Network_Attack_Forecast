from __future__ import annotations

from pathlib import Path

import numpy as np
import xgboost as xgb

from .config import (
    MODEL_FILE,
    MODEL_THRESHOLD,
    HISTORY_MINUTES,
    FORECAST_HORIZON_MINUTES,
    FEATURE_COUNT,
    FLATTENED_FEATURE_COUNT,
)


class NetworkAttackPredictor:
    def __init__(
        self,
        model_file: Path = MODEL_FILE,
        threshold: float = MODEL_THRESHOLD,
    ):
        self.model_file = Path(model_file)
        self.threshold = float(threshold)

        if not self.model_file.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_file}"
            )

        self.model = xgb.XGBClassifier()
        self.model.load_model(str(self.model_file))

    def _validate_window(self, window: np.ndarray) -> np.ndarray:
        """
        Validate the incoming 10-minute x 161-feature model window.
        """

        array = np.asarray(window, dtype=np.float32)

        expected_shape = (HISTORY_MINUTES, FEATURE_COUNT)

        if array.shape != expected_shape:
            raise ValueError(
                f"Invalid window shape: {array.shape}. "
                f"Expected {expected_shape}."
            )

        if not np.isfinite(array).all():
            raise ValueError(
                "Input window contains NaN or infinite values."
            )

        return array

    def predict(self, window: np.ndarray) -> dict:
        """
        Generate a 15-minute attack forecast.
        """

        array = self._validate_window(window)

        flattened = array.reshape(1, FLATTENED_FEATURE_COUNT)

        probability = float(
            self.model.predict_proba(flattened)[0, 1]
        )

        predicted_attack = probability >= self.threshold

        risk_level = self._risk_level(probability)

        return {
            "forecast_horizon_minutes": FORECAST_HORIZON_MINUTES,
            "attack_probability": probability,
            "predicted_attack": predicted_attack,
            "decision_threshold": self.threshold,
            "risk_level": risk_level,
        }

    @staticmethod
    def _risk_level(probability: float) -> str:
        """
        Convert attack probability into a human-readable risk level.
        """

        if probability >= 0.80:
            return "CRITICAL"

        if probability >= 0.60:
            return "HIGH"

        if probability >= 0.40:
            return "MEDIUM"

        if probability >= 0.20:
            return "LOW"

        return "MINIMAL"

    def check_model(self) -> dict:
        """
        Return model health and configuration information.
        """

        try:
            booster = self.model.get_booster()

            return {
                "loaded": True,
                "model_file": str(self.model_file),
                "model_type": "XGBoost",
                "feature_count": FEATURE_COUNT,
                "flattened_feature_count": FLATTENED_FEATURE_COUNT,
                "history_minutes": HISTORY_MINUTES,
                "forecast_horizon_minutes": FORECAST_HORIZON_MINUTES,
                "decision_threshold": self.threshold,
                "booster_features": booster.num_features(),
            }

        except Exception as exc:
            return {
                "loaded": False,
                "model_file": str(self.model_file),
                "error": str(exc),
            }