from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_FILE = MODEL_DIR / "xgboost_enhanced_forecast15_v2.json"


# ============================================================
# FORECAST CONFIGURATION
# ============================================================

HISTORY_MINUTES = 10
FORECAST_HORIZON_MINUTES = 15

FEATURE_COUNT = 161
FLATTENED_FEATURE_COUNT = HISTORY_MINUTES * FEATURE_COUNT


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_THRESHOLD = 0.20


# ============================================================
# API CONFIGURATION
# ============================================================

API_TITLE = "Network Attack Forecasting API"
API_DESCRIPTION = (
    "Backend inference API for temporal network attack forecasting."
)
API_VERSION = "1.0.0"