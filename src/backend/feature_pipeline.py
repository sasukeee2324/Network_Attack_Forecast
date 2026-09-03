from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

HISTORY_MINUTES = 10
EXPECTED_FEATURE_COUNT = 161

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

TEMPORAL_DIR = PROJECT_ROOT / "data" / "processed" / "temporal"

NETWORK_STATES_FILE = TEMPORAL_DIR / "network_states.csv"

# Exact feature schema used when the enhanced model was trained.
ENHANCED_FEATURE_FILE = TEMPORAL_DIR / "network_states_enhanced.csv"


# =============================================================================
# ORIGINAL BASE FEATURES
# =============================================================================

BASE_FEATURES = [
    "Flow_Count",
    "Avg_Flow_Duration",
    "Avg_Fwd_Packets",
    "Avg_Bwd_Packets",
    "Avg_Fwd_Bytes",
    "Avg_Bwd_Bytes",
    "Avg_Bytes_Per_Second",
    "Avg_Packets_Per_Second",
    "Avg_Packet_Length",
    "Avg_Packet_Length_Std",
    "Avg_Fwd_Packets_Per_Second",
    "Avg_Bwd_Packets_Per_Second",
    "SYN_Count",
    "RST_Count",
    "PSH_Count",
    "ACK_Count",
    "Unique_Destination_Ports",
    "Unique_Protocols",
]

TREND_FEATURES = BASE_FEATURES.copy()


# =============================================================================
# LABEL / NON-MODEL COLUMNS
# =============================================================================

NON_MODEL_COLUMNS = {
    "Minute",
    "Source_File",
    "Attack_Flow_Count",
    "Attack_Ratio",
    "Attack_State",
    "Attack_Type",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def safe_pct_change(
    series: pd.Series,
    periods: int = 1,
) -> pd.Series:
    """
    Calculate percentage change safely.

    Extremely small denominators are clipped to avoid division by zero.
    Infinite values are converted to NaN.
    """

    previous = series.shift(periods)

    denominator = previous.abs().clip(lower=1e-6)

    result = (series - previous) / denominator

    return result.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def rolling_zscore(
    series: pd.Series,
    window: int = 5,
) -> pd.Series:
    """
    Calculate rolling z-score.
    """

    rolling_mean = series.rolling(
        window=window,
        min_periods=window,
    ).mean()

    rolling_std = series.rolling(
        window=window,
        min_periods=window,
    ).std()

    rolling_std = rolling_std.replace(
        0,
        np.nan,
    )

    result = (series - rolling_mean) / rolling_std

    return result.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def rolling_slope(
    series: pd.Series,
    window: int = 5,
) -> pd.Series:
    """
    Calculate rolling linear regression slope.
    """

    x = np.arange(
        window,
        dtype=float,
    )

    def calculate_slope(values):

        values = np.asarray(
            values,
            dtype=float,
        )

        valid = np.isfinite(values)

        if valid.sum() < 2:
            return np.nan

        x_valid = x[-len(values):][valid]
        y_valid = values[valid]

        if len(x_valid) < 2:
            return np.nan

        x_mean = x_valid.mean()
        y_mean = y_valid.mean()

        numerator = np.sum(
            (x_valid - x_mean)
            * (y_valid - y_mean)
        )

        denominator = np.sum(
            (x_valid - x_mean) ** 2
        )

        if denominator == 0:
            return np.nan

        return numerator / denominator

    return series.rolling(
        window=window,
        min_periods=2,
    ).apply(
        calculate_slope,
        raw=True,
    )


# =============================================================================
# FEATURE SCHEMA
# =============================================================================

def get_model_feature_columns(
    df: pd.DataFrame | None = None,
) -> List[str]:
    """
    Return the exact 161 feature columns expected by the trained model.

    The canonical schema is read from network_states_enhanced.csv.
    """

    if not ENHANCED_FEATURE_FILE.exists():
        raise FileNotFoundError(
            "Canonical enhanced feature file not found:\n"
            f"{ENHANCED_FEATURE_FILE}"
        )

    canonical_df = pd.read_csv(
        ENHANCED_FEATURE_FILE,
        nrows=1,
    )

    feature_columns = [
        column
        for column in canonical_df.columns
        if column not in NON_MODEL_COLUMNS
    ]

    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            "Canonical feature schema mismatch!\n"
            f"Expected: {EXPECTED_FEATURE_COUNT}\n"
            f"Found:    {len(feature_columns)}\n"
            f"File:     {ENHANCED_FEATURE_FILE}"
        )

    return feature_columns


# =============================================================================
# FEATURE GENERATION FOR ONE SOURCE
# =============================================================================

def generate_features_for_source(
    group: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate enhanced temporal features for one capture/source.

    Temporal calculations are performed separately for each source file so
    rolling windows do not cross capture boundaries.
    """

    group = group.copy()

    # -------------------------------------------------------------------------
    # Sort chronologically without modifying the original Minute values.
    #
    # IMPORTANT:
    # Minute contains timestamp strings. We must preserve those strings because
    # Source_File + Minute uniquely identifies a network-state row.
    # -------------------------------------------------------------------------

    group["_Minute_Sort"] = pd.to_datetime(
        group["Minute"],
        errors="coerce",
    )

    invalid_minutes = group["_Minute_Sort"].isna().sum()

    if invalid_minutes:
        raise ValueError(
            "Invalid Minute timestamps detected.\n"
            f"Invalid values: {invalid_minutes}"
        )

    group = (
        group
        .sort_values("_Minute_Sort")
        .reset_index(drop=True)
        .drop(columns="_Minute_Sort")
    )

    # -------------------------------------------------------------------------
    # Ensure base features are numeric
    # -------------------------------------------------------------------------

    for feature in BASE_FEATURES:
        group[feature] = pd.to_numeric(
            group[feature],
            errors="coerce",
        )

    # -------------------------------------------------------------------------
    # Temporal features
    # -------------------------------------------------------------------------

    generated_features = {}

    for feature in TREND_FEATURES:

        series = group[feature]

        generated_features[
            f"{feature}_Diff1"
        ] = series - series.shift(1)

        generated_features[
            f"{feature}_Diff5"
        ] = series - series.shift(5)

        generated_features[
            f"{feature}_PctChange1"
        ] = safe_pct_change(
            series,
            periods=1,
        )

        generated_features[
            f"{feature}_PctChange5"
        ] = safe_pct_change(
            series,
            periods=5,
        )

        generated_features[
            f"{feature}_RollingMean5"
        ] = (
            series
            .rolling(
                window=5,
                min_periods=1,
            )
            .mean()
        )

        generated_features[
            f"{feature}_RollingStd5"
        ] = (
            series
            .rolling(
                window=5,
                min_periods=2,
            )
            .std()
        )

        generated_features[
            f"{feature}_ZScore5"
        ] = rolling_zscore(
            series,
            window=5,
        )

        generated_features[
            f"{feature}_Slope5"
        ] = rolling_slope(
            series,
            window=5,
        )

    # -------------------------------------------------------------------------
    # Add all temporal features at once.
    #
    # This avoids the DataFrame fragmentation warning caused by repeatedly
    # inserting columns one at a time.
    # -------------------------------------------------------------------------

    temporal_df = pd.DataFrame(
        generated_features,
        index=group.index,
    )

    group = pd.concat(
        [
            group,
            temporal_df,
        ],
        axis=1,
    )

    # -------------------------------------------------------------------------
    # Behaviour ratios
    # -------------------------------------------------------------------------

    flow_count = (
        group["Flow_Count"]
        .abs()
        .clip(lower=1e-6)
    )

    ratio_features = pd.DataFrame(
        {
            "SYN_Per_Flow": (
                group["SYN_Count"]
                / flow_count
            ),

            "RST_Per_Flow": (
                group["RST_Count"]
                / flow_count
            ),

            "PSH_Per_Flow": (
                group["PSH_Count"]
                / flow_count
            ),

            "ACK_Per_Flow": (
                group["ACK_Count"]
                / flow_count
            ),

            "Bwd_Fwd_Packet_Ratio": (
                group["Avg_Bwd_Packets"]
                / group["Avg_Fwd_Packets"]
                .abs()
                .clip(lower=1e-6)
            ),

            "Bwd_Fwd_Byte_Ratio": (
                group["Avg_Bwd_Bytes"]
                / group["Avg_Fwd_Bytes"]
                .abs()
                .clip(lower=1e-6)
            ),

            "Ports_Per_Flow": (
                group["Unique_Destination_Ports"]
                / flow_count
            ),
        },
        index=group.index,
    )

    group = pd.concat(
        [
            group,
            ratio_features,
        ],
        axis=1,
    )

    # -------------------------------------------------------------------------
    # Clean numerical problems
    # -------------------------------------------------------------------------

    group = group.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return group


# =============================================================================
# COMPLETE ENHANCED FEATURE PIPELINE
# =============================================================================

def generate_enhanced_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate enhanced temporal features and align them to the exact
    161-feature schema used by the trained model.

    Minute and Source_File are preserved as metadata.
    """

    df = df.copy()

    # -------------------------------------------------------------------------
    # Validate required columns
    # -------------------------------------------------------------------------

    required_columns = [
        "Minute",
        "Source_File",
        *BASE_FEATURES,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_columns
            )
        )

    # -------------------------------------------------------------------------
    # IMPORTANT:
    # Preserve Minute as a timestamp/string.
    #
    # The previous implementation converted Minute with pd.to_numeric(),
    # which transformed timestamp strings into NaN and destroyed row identity.
    # -------------------------------------------------------------------------

    df["Minute"] = df["Minute"].astype(str)

    # -------------------------------------------------------------------------
    # Validate timestamps without replacing the original values.
    # -------------------------------------------------------------------------

    minute_datetime = pd.to_datetime(
        df["Minute"],
        errors="coerce",
    )

    invalid_minutes = minute_datetime.isna().sum()

    if invalid_minutes:
        raise ValueError(
            "Invalid Minute timestamps detected.\n"
            f"Invalid values: {invalid_minutes}"
        )

    # -------------------------------------------------------------------------
    # Temporary chronological sort.
    #
    # We deliberately do NOT overwrite Minute.
    # -------------------------------------------------------------------------

    df["_Minute_Sort"] = minute_datetime

    df = (
        df
        .sort_values(
            ["Source_File", "_Minute_Sort"]
        )
        .reset_index(drop=True)
        .drop(columns="_Minute_Sort")
    )

    # -------------------------------------------------------------------------
    # Generate features separately for each source.
    # -------------------------------------------------------------------------

    processed_groups = []

    for source_file, group in df.groupby(
        "Source_File",
        sort=False,
        dropna=False,
    ):

        processed_group = generate_features_for_source(
            group
        )

        processed_groups.append(
            processed_group
        )

    if not processed_groups:
        raise RuntimeError(
            "No source groups were available for feature generation."
        )

    enhanced_df = pd.concat(
        processed_groups,
        ignore_index=True,
    )

    # -------------------------------------------------------------------------
    # Final chronological ordering.
    # -------------------------------------------------------------------------

    enhanced_df["_Minute_Sort"] = pd.to_datetime(
        enhanced_df["Minute"],
        errors="coerce",
    )

    enhanced_df = (
        enhanced_df
        .sort_values(
            ["Source_File", "_Minute_Sort"]
        )
        .reset_index(drop=True)
        .drop(columns="_Minute_Sort")
    )

    # -------------------------------------------------------------------------
    # Get canonical trained-model schema.
    # -------------------------------------------------------------------------

    feature_columns = get_model_feature_columns(
        enhanced_df
    )

    # -------------------------------------------------------------------------
    # Check that every canonical feature exists.
    # -------------------------------------------------------------------------

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in enhanced_df.columns
    ]

    if missing_features:

        print()
        print("=" * 80)
        print("MISSING CANONICAL FEATURES")
        print("=" * 80)

        for feature in missing_features:
            print(feature)

        print("=" * 80)

        raise RuntimeError(
            "The backend could not reproduce all canonical "
            "training features.\n"
            f"Missing: {len(missing_features)}"
        )

    # -------------------------------------------------------------------------
    # Select exact canonical feature order.
    # -------------------------------------------------------------------------

    ordered_features = enhanced_df[
        feature_columns
    ].copy()

    # -------------------------------------------------------------------------
    # Convert model features to numeric.
    # -------------------------------------------------------------------------

    ordered_features = ordered_features.apply(
        pd.to_numeric,
        errors="coerce",
    )

    # -------------------------------------------------------------------------
    # Replace infinite values.
    # -------------------------------------------------------------------------

    ordered_features = ordered_features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # -------------------------------------------------------------------------
    # Preserve metadata / labels.
    # -------------------------------------------------------------------------

    output_columns = []

    for column in [
        "Minute",
        "Source_File",
        "Attack_Flow_Count",
        "Attack_Ratio",
        "Attack_State",
        "Attack_Type",
    ]:

        if column in enhanced_df.columns:
            output_columns.append(column)

    metadata_df = enhanced_df[
        output_columns
    ].copy()

    # -------------------------------------------------------------------------
    # Combine metadata + exact model features.
    # -------------------------------------------------------------------------

    result = pd.concat(
        [
            metadata_df.reset_index(drop=True),
            ordered_features.reset_index(drop=True),
        ],
        axis=1,
    )

    # -------------------------------------------------------------------------
    # Final validation.
    # -------------------------------------------------------------------------

    actual_feature_count = len(
        get_model_feature_columns(result)
    )

    if actual_feature_count != EXPECTED_FEATURE_COUNT:

        raise RuntimeError(
            "\n"
            "Feature count mismatch!\n"
            f"Expected: {EXPECTED_FEATURE_COUNT}\n"
            f"Actual:   {actual_feature_count}\n"
        )

    # -------------------------------------------------------------------------
    # Verify row identity was preserved.
    # -------------------------------------------------------------------------

    identity_keys = (
        result["Source_File"].astype(str)
        + "||"
        + result["Minute"].astype(str)
    )

    duplicate_count = identity_keys.duplicated().sum()

    if duplicate_count:
        raise RuntimeError(
            "Feature pipeline produced duplicate Source_File + Minute keys.\n"
            f"Duplicates: {duplicate_count}"
        )

    # -------------------------------------------------------------------------
    # Verify row count was preserved.
    # -------------------------------------------------------------------------

    if len(result) != len(df):
        raise RuntimeError(
            "Feature pipeline changed the number of rows.\n"
            f"Input rows:  {len(df)}\n"
            f"Output rows: {len(result)}"
        )

    print(
        f"Canonical model feature count: "
        f"{actual_feature_count}"
    )

    return result


# =============================================================================
# CREATE MODEL WINDOW
# =============================================================================

def create_model_window(
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Extract the latest 10 minutes of network history.

    Returns:
        numpy array with shape (10, 161)
    """

    if len(df) < HISTORY_MINUTES:
        raise ValueError(
            f"At least {HISTORY_MINUTES} rows are required "
            f"to create a model window. "
            f"Received: {len(df)}"
        )

    # -------------------------------------------------------------------------
    # Get exact trained feature ordering.
    # -------------------------------------------------------------------------

    feature_columns = get_model_feature_columns(
        df
    )

    # -------------------------------------------------------------------------
    # Select latest history window.
    #
    # generate_enhanced_features() already sorts chronologically by source.
    # For normal API input, the last 10 rows represent the latest window.
    # -------------------------------------------------------------------------

    window_df = df.tail(
        HISTORY_MINUTES
    )

    window = window_df[
        feature_columns
    ].to_numpy(
        dtype=np.float32
    )

    # -------------------------------------------------------------------------
    # Validate shape.
    # -------------------------------------------------------------------------

    expected_shape = (
        HISTORY_MINUTES,
        EXPECTED_FEATURE_COUNT,
    )

    if window.shape != expected_shape:

        raise RuntimeError(
            "\n"
            "Model window shape mismatch!\n"
            f"Expected: {expected_shape}\n"
            f"Actual:   {window.shape}\n"
        )

    # -------------------------------------------------------------------------
    # Validate numerical values.
    # -------------------------------------------------------------------------

    if not np.isfinite(window).all():

        bad_count = np.sum(
            ~np.isfinite(window)
        )

        raise ValueError(
            "\n"
            "Model window contains invalid values!\n"
            f"Invalid values: {bad_count}\n"
        )

    return window


# =============================================================================
# SIMPLE PIPELINE SELF-TEST
# =============================================================================

def validate_feature_pipeline() -> dict:
    """
    Run a basic validation of the backend feature pipeline.
    """

    if not NETWORK_STATES_FILE.exists():
        raise FileNotFoundError(
            f"Network states file not found:\n"
            f"{NETWORK_STATES_FILE}"
        )

    if not ENHANCED_FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Enhanced feature file not found:\n"
            f"{ENHANCED_FEATURE_FILE}"
        )

    df = pd.read_csv(
        NETWORK_STATES_FILE
    )

    enhanced_df = generate_enhanced_features(
        df
    )

    feature_columns = get_model_feature_columns(
        enhanced_df
    )

    window = create_model_window(
        enhanced_df
    )

    return {
        "input_rows": len(df),
        "enhanced_rows": len(enhanced_df),
        "feature_count": len(feature_columns),
        "window_shape": window.shape,
        "finite_window": bool(
            np.isfinite(window).all()
        ),
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 80)
    print("BACKEND FEATURE PIPELINE TEST")
    print("=" * 80)

    result = validate_feature_pipeline()

    print()
    print("Pipeline validation successful.")
    print()
    print(f"Input rows:       {result['input_rows']}")
    print(f"Enhanced rows:    {result['enhanced_rows']}")
    print(f"Feature count:    {result['feature_count']}")
    print(f"Window shape:     {result['window_shape']}")
    print(f"Finite window:    {result['finite_window']}")
    print()
    print("=" * 80)