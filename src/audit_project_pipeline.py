"""
Network Attack Forecasting
Final Project Pipeline Audit

Project root:
    D:\\Network_Attack_Forecasting

Script location:
    D:\\Network_Attack_Forecasting\\src\\audit_project_pipeline.py

This audit validates:
    1. Required project files
    2. Network-state dataset
    3. Enhanced temporal features
    4. Enhanced sequences
    5. Train/validation/test split
    6. Enhanced unseen-attack dataset
    7. Trained models
    8. Python environment
    9. Source scripts
   10. Project artifact summary

Temporal assumptions:
    History window      = 10 minutes
    Forecast horizon    = 15 minutes
    Split start embargo = 25 minutes

Important:
    The actual enhanced sequence metadata contains:
        Source_File
        Forecast_Start

    It does NOT contain:
        History_End
        Forecast_End

    The unseen-attack metadata uses a different schema, so it is
    validated separately rather than being forced to contain
    Forecast_Start.
"""

from pathlib import Path
import sys
import importlib

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

# audit_project_pipeline.py is inside:
#
# D:\Network_Attack_Forecasting\src
#
# Therefore the project root is two levels up from this file.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

TEMPORAL_DIR = PROCESSED_DIR / "temporal"

# Enhanced pipeline
ENHANCED_SEQ_DIR = PROCESSED_DIR / "enhanced_sequences"
ENHANCED_TRAIN_DIR = PROCESSED_DIR / "enhanced_training"
ENHANCED_UNSEEN_DIR = PROCESSED_DIR / "enhanced_unseen_attack"

# Legacy pipeline
LEGACY_SEQ_DIR = PROCESSED_DIR / "sequences"
LEGACY_TRAIN_DIR = PROCESSED_DIR / "training"
LEGACY_UNSEEN_DIR = PROCESSED_DIR / "unseen_attack"

MODELS_DIR = PROJECT_ROOT / "models"

# Temporal assumptions
HISTORY_WINDOW = 10
FORECAST_HORIZON_MINUTES = 15
EMBARGO_MINUTES = 25

# Enhanced model input
EXPECTED_ENHANCED_FEATURES = 161

# Expected main dataset
EXPECTED_NETWORK_STATE_ROWS = 1638
EXPECTED_NETWORK_STATE_COLUMNS = 24

# Expected enhanced feature dataset
EXPECTED_ENHANCED_ROWS = 1638
EXPECTED_ENHANCED_COLUMNS = 167

# Expected enhanced sequence dataset
EXPECTED_ENHANCED_SEQUENCE_COUNT = 1481


# ============================================================
# AUDIT COUNTERS
# ============================================================

PASS_COUNT = 0
FAIL_COUNT = 0
WARN_COUNT = 0


# ============================================================
# OUTPUT HELPERS
# ============================================================

def section(title):
    print()
    print("=" * 75)
    print(title)
    print("=" * 75)


def subsection(title):
    print()
    print(title)
    print("-" * 40)


def info(message):
    print(f"[INFO] {message}")


def check_pass(message):
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"[PASS] {message}")


def check_fail(message):
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"[FAIL] {message}")


def check_warn(message):
    global WARN_COUNT
    WARN_COUNT += 1
    print(f"[WARNING] {message}")


# ============================================================
# FILE HELPERS
# ============================================================

def check_file(path, label, optional=False):

    if path.exists() and path.is_file():
        check_pass(label)
        return True

    if optional:
        check_warn(
            f"Optional/legacy artifact not found: {path}"
        )
    else:
        check_fail(label)

    return False


def load_npy(path, label):

    try:
        return np.load(
            path,
            allow_pickle=True
        )
    except Exception as exc:
        check_fail(
            f"Could not load {label}: {exc}"
        )
        return None


def load_csv(path, label):

    try:
        return pd.read_csv(path)
    except Exception as exc:
        check_fail(
            f"Could not load {label}: {exc}"
        )
        return None


# ============================================================
# TEMPORAL HELPERS
# ============================================================

def prepare_forecast_metadata(metadata):
    """
    Prepare metadata that contains Forecast_Start.

    Forecast_End is derived only when needed:

        Forecast_End =
            Forecast_Start + 15 minutes

    This function is used for the main enhanced
    sequence/training pipeline.
    """

    required = [
        "Source_File",
        "Forecast_Start",
    ]

    missing = [
        column
        for column in required
        if column not in metadata.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required metadata columns: {missing}"
        )

    result = metadata.copy()

    result["_Forecast_Start"] = pd.to_datetime(
        result["Forecast_Start"],
        errors="coerce"
    )

    if result["_Forecast_Start"].isna().any():
        raise ValueError(
            "Invalid Forecast_Start timestamps detected"
        )

    result["_Forecast_End"] = (
        result["_Forecast_Start"]
        + pd.Timedelta(
            minutes=FORECAST_HORIZON_MINUTES
        )
    )

    return result


def check_source_chronology(
    metadata,
    label
):
    """
    Check chronology separately for every Source_File.
    """

    try:
        prepared = prepare_forecast_metadata(
            metadata
        )
    except Exception as exc:
        check_fail(
            f"{label}: temporal metadata invalid: {exc}"
        )
        return None

    all_valid = True

    for source, group in prepared.groupby(
        "Source_File"
    ):

        times = group["_Forecast_Start"]

        if times.is_monotonic_increasing:

            check_pass(
                f"{label}: chronology valid for {source}"
            )

        else:

            check_fail(
                f"{label}: chronology invalid for {source}"
            )

            all_valid = False

    if all_valid:

        check_pass(
            f"{label}: all source timelines chronological"
        )

    return prepared


def check_exact_timestamp_overlap(
    metadata_a,
    metadata_b,
    name_a,
    name_b
):
    """
    Check whether two splits contain identical Forecast_Start
    timestamps for the same source capture.
    """

    try:
        a = prepare_forecast_metadata(
            metadata_a
        )

        b = prepare_forecast_metadata(
            metadata_b
        )

    except Exception as exc:

        check_fail(
            f"Could not check timestamp overlap "
            f"{name_a}/{name_b}: {exc}"
        )

        return False

    overlap_found = False

    common_sources = sorted(
        set(a["Source_File"])
        & set(b["Source_File"])
    )

    for source in common_sources:

        times_a = set(
            a.loc[
                a["Source_File"] == source,
                "_Forecast_Start"
            ]
        )

        times_b = set(
            b.loc[
                b["Source_File"] == source,
                "_Forecast_Start"
            ]
        )

        overlap = times_a & times_b

        if overlap:

            check_fail(
                f"Exact Forecast_Start overlap: "
                f"{name_a}/{name_b}, "
                f"{source}: {len(overlap)} timestamps"
            )

            overlap_found = True

    if not overlap_found:

        check_pass(
            f"No exact Forecast_Start overlap: "
            f"{name_a}/{name_b}"
        )

    return not overlap_found


def check_start_time_embargo(
    earlier_metadata,
    later_metadata,
    earlier_name,
    later_name
):
    """
    Check the split embargo using Forecast_Start timestamps.

    The dataset split was constructed with a 25-minute
    start-to-start separation.

    Therefore:

        later Forecast_Start
        -
        earlier Forecast_Start

    must be >= 25 minutes.

    We intentionally DO NOT subtract the 15-minute forecast
    horizon here.

    This is different from checking the gap after the forecast
    interval itself.
    """

    try:

        earlier = prepare_forecast_metadata(
            earlier_metadata
        )

        later = prepare_forecast_metadata(
            later_metadata
        )

    except Exception as exc:

        check_fail(
            f"Could not prepare embargo metadata "
            f"for {earlier_name} -> {later_name}: {exc}"
        )

        return False

    violations = []

    common_sources = sorted(
        set(earlier["Source_File"])
        & set(later["Source_File"])
    )

    for source in common_sources:

        e = earlier[
            earlier["Source_File"] == source
        ].sort_values(
            "_Forecast_Start"
        )

        l = later[
            later["Source_File"] == source
        ].sort_values(
            "_Forecast_Start"
        )

        if e.empty or l.empty:
            continue

        # IMPORTANT:
        # Use start-to-start separation.
        earlier_last_start = e[
            "_Forecast_Start"
        ].max()

        later_first_start = l[
            "_Forecast_Start"
        ].min()

        gap_minutes = (
            later_first_start
            - earlier_last_start
        ).total_seconds() / 60.0

        if gap_minutes < EMBARGO_MINUTES:

            violations.append(
                (
                    source,
                    gap_minutes,
                    earlier_last_start,
                    later_first_start,
                )
            )

    if not violations:

        check_pass(
            f"{earlier_name} -> {later_name}: "
            f"{EMBARGO_MINUTES}-minute start-time embargo satisfied"
        )

        return True

    for (
        source,
        gap_minutes,
        earlier_last_start,
        later_first_start,
    ) in violations:

        check_fail(
            f"Embargo violation: {source} | "
            f"start-to-start gap={gap_minutes:.2f} min | "
            f"earlier_start={earlier_last_start} | "
            f"later_start={later_first_start}"
        )

    return False


# ============================================================
# 1. REQUIRED PROJECT FILES
# ============================================================

def audit_required_files():

    section("1. REQUIRED PROJECT FILES")

    required_files = [

        # Main temporal datasets
        (
            TEMPORAL_DIR / "network_states.csv",
            "network_states.csv"
        ),

        (
            TEMPORAL_DIR / "network_states_enhanced.csv",
            "network_states_enhanced.csv"
        ),

        # Enhanced sequences
        (
            ENHANCED_SEQ_DIR / "X_enhanced.npy",
            "enhanced_sequences/X_enhanced.npy"
        ),

        (
            ENHANCED_SEQ_DIR / "y_forecast15_enhanced.npy",
            "enhanced_sequences/y_forecast15_enhanced.npy"
        ),

        (
            ENHANCED_SEQ_DIR / "y_presence5_enhanced.npy",
            "enhanced_sequences/y_presence5_enhanced.npy"
        ),

        (
            ENHANCED_SEQ_DIR / "y_onset5_enhanced.npy",
            "enhanced_sequences/y_onset5_enhanced.npy"
        ),

        (
            ENHANCED_SEQ_DIR / "metadata_enhanced.csv",
            "enhanced_sequences/metadata_enhanced.csv"
        ),

        # Enhanced training
        (
            ENHANCED_TRAIN_DIR / "X_train.npy",
            "enhanced_training/X_train.npy"
        ),

        (
            ENHANCED_TRAIN_DIR / "X_validation.npy",
            "enhanced_training/X_validation.npy"
        ),

        (
            ENHANCED_TRAIN_DIR / "X_test.npy",
            "enhanced_training/X_test.npy"
        ),

        (
            ENHANCED_TRAIN_DIR / "y_train.npy",
            "enhanced_training/y_train.npy"
        ),

        (
            ENHANCED_TRAIN_DIR / "y_validation.npy",
            "enhanced_training/y_validation.npy"
        ),

        (
            ENHANCED_TRAIN_DIR / "y_test.npy",
            "enhanced_training/y_test.npy"
        ),

        (
            ENHANCED_TRAIN_DIR / "metadata_train.csv",
            "enhanced_training/metadata_train.csv"
        ),

        (
            ENHANCED_TRAIN_DIR / "metadata_validation.csv",
            "enhanced_training/metadata_validation.csv"
        ),

        (
            ENHANCED_TRAIN_DIR / "metadata_test.csv",
            "enhanced_training/metadata_test.csv"
        ),

        # Enhanced unseen attack
        (
            ENHANCED_UNSEEN_DIR / "X_train.npy",
            "enhanced_unseen_attack/X_train.npy"
        ),

        (
            ENHANCED_UNSEEN_DIR / "y_train.npy",
            "enhanced_unseen_attack/y_train.npy"
        ),

        (
            ENHANCED_UNSEEN_DIR / "metadata_train.csv",
            "enhanced_unseen_attack/metadata_train.csv"
        ),

        (
            ENHANCED_UNSEEN_DIR / "X_unseen.npy",
            "enhanced_unseen_attack/X_unseen.npy"
        ),

        (
            ENHANCED_UNSEEN_DIR / "y_unseen.npy",
            "enhanced_unseen_attack/y_unseen.npy"
        ),

        (
            ENHANCED_UNSEEN_DIR / "metadata_unseen.csv",
            "enhanced_unseen_attack/metadata_unseen.csv"
        ),
    ]

    for path, label in required_files:

        check_file(
            path,
            label
        )

    # --------------------------------------------------------
    # Legacy/original artifacts
    # --------------------------------------------------------

    print()
    print("Legacy/original datasets:")

    legacy_files = [

        LEGACY_SEQ_DIR / "y_forecast15.npy",
        LEGACY_SEQ_DIR / "y_presence5.npy",
        LEGACY_SEQ_DIR / "y_onset5.npy",

        LEGACY_TRAIN_DIR / "X_train.npy",
        LEGACY_TRAIN_DIR / "X_validation.npy",
        LEGACY_TRAIN_DIR / "X_test.npy",

        LEGACY_TRAIN_DIR / "y_forecast15_train.npy",
        LEGACY_TRAIN_DIR / "y_forecast15_validation.npy",
        LEGACY_TRAIN_DIR / "y_forecast15_test.npy",

        LEGACY_UNSEEN_DIR / "X_train.npy",
        LEGACY_UNSEEN_DIR / "X_unseen.npy",
        LEGACY_UNSEEN_DIR / "y_train.npy",
        LEGACY_UNSEEN_DIR / "y_unseen.npy",
    ]

    for path in legacy_files:

        relative = path.relative_to(
            PROJECT_ROOT
        )

        check_file(
            path,
            f"Optional/legacy artifact: {relative}",
            optional=True
        )


# ============================================================
# 2. NETWORK STATES
# ============================================================

def audit_network_states():

    section("2. NETWORK STATES")

    path = TEMPORAL_DIR / "network_states.csv"

    if not path.exists():

        check_fail(
            "network_states.csv"
        )

        return

    df = load_csv(
        path,
        "network_states.csv"
    )

    if df is None:
        return

    print(
        f"Rows: {df.shape[0]}, columns: {df.shape[1]}"
    )

    if df.shape == (
        EXPECTED_NETWORK_STATE_ROWS,
        EXPECTED_NETWORK_STATE_COLUMNS
    ):

        check_pass(
            "Rows: 1638, columns: 24"
        )

    else:

        check_warn(
            f"Unexpected shape: {df.shape}; "
            f"expected "
            f"({EXPECTED_NETWORK_STATE_ROWS}, "
            f"{EXPECTED_NETWORK_STATE_COLUMNS})"
        )

    expected_columns = [
        "Minute",
        "Flow_Count",
        "Attack_Flow_Count",
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
        "Attack_Ratio",
        "Attack_Type",
        "Attack_State",
        "Source_File",
    ]

    missing = [
        column
        for column in expected_columns
        if column not in df.columns
    ]

    if not missing:

        check_pass(
            "All required network-state columns present"
        )

    else:

        check_fail(
            f"Missing network-state columns: {missing}"
        )

    # --------------------------------------------------------
    # Timestamp validation
    # --------------------------------------------------------

    if "Minute" in df.columns:

        times = pd.to_datetime(
            df["Minute"],
            errors="coerce"
        )

        if times.isna().any():

            check_fail(
                "Invalid Minute timestamps detected"
            )

        else:

            check_pass(
                "All Minute timestamps valid"
            )

            if times.is_monotonic_increasing:

                check_pass(
                    "network_states.csv is globally chronological"
                )

            else:

                check_warn(
                    "network_states.csv is not globally chronological"
                )

    # --------------------------------------------------------
    # Source-specific chronology
    # --------------------------------------------------------

    if (
        "Source_File" in df.columns
        and "Minute" in df.columns
    ):

        all_valid = True

        for source, group in df.groupby(
            "Source_File"
        ):

            source_times = pd.to_datetime(
                group["Minute"],
                errors="coerce"
            )

            if source_times.is_monotonic_increasing:

                check_pass(
                    f"Chronology valid: {source}"
                )

            else:

                check_fail(
                    f"Chronology invalid: {source}"
                )

                all_valid = False

        if all_valid:

            check_pass(
                "All Source_File timelines are chronological"
            )

        print()
        print("Source distribution:")

        for source, count in (
            df["Source_File"]
            .value_counts()
            .items()
        ):

            print(
                f"  {source}: {count}"
            )


# ============================================================
# 3. ENHANCED FEATURE DATASET
# ============================================================

def audit_enhanced_features():

    section("3. ENHANCED FEATURE DATASET")

    path = (
        TEMPORAL_DIR
        / "network_states_enhanced.csv"
    )

    if not path.exists():

        check_fail(
            "network_states_enhanced.csv"
        )

        return

    df = load_csv(
        path,
        "network_states_enhanced.csv"
    )

    if df is None:
        return

    print(
        f"Rows: {df.shape[0]}, columns: {df.shape[1]}"
    )

    if df.shape[0] == EXPECTED_ENHANCED_ROWS:

        check_pass(
            "Rows: 1638"
        )

    else:

        check_warn(
            f"Expected 1638 rows, found {df.shape[0]}"
        )

    if df.shape[1] == EXPECTED_ENHANCED_COLUMNS:

        check_pass(
            "Rows: 1638, columns: 167"
        )

    else:

        check_warn(
            f"Expected 167 columns, found {df.shape[1]}"
        )

    # --------------------------------------------------------
    # Identify model features
    # --------------------------------------------------------

    label_columns = {
        "Attack_Flow_Count",
        "Attack_Ratio",
        "Attack_State",
        "Attack_Type",
    }

    metadata_columns = {
        "Minute",
        "Source_File",
    }

    model_features = [
        column
        for column in df.columns
        if column not in label_columns
        and column not in metadata_columns
    ]

    print(
        f"Model feature count: {len(model_features)}"
    )

    if len(model_features) == EXPECTED_ENHANCED_FEATURES:

        check_pass(
            "Enhanced dataset contains exactly "
            "161 model features"
        )

    else:

        check_fail(
            f"Expected 161 model features, "
            f"found {len(model_features)}"
        )

    # --------------------------------------------------------
    # Leakage check
    # --------------------------------------------------------

    leakage_patterns = [
        "attack_type",
        "attack_state",
        "attack_ratio",
        "attack_flow_count",
    ]

    leakage_features = []

    for feature in model_features:

        lower = feature.lower()

        if any(
            pattern in lower
            for pattern in leakage_patterns
        ):

            leakage_features.append(
                feature
            )

    if not leakage_features:

        check_pass(
            "Enhanced feature leakage check passed"
        )

    else:

        check_fail(
            f"Potential label leakage: "
            f"{leakage_features}"
        )

    # --------------------------------------------------------
    # Numeric validation
    # --------------------------------------------------------

    non_numeric = [
        column
        for column in model_features
        if not pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    if not non_numeric:

        check_pass(
            "All model features are numeric"
        )

    else:

        check_fail(
            f"Non-numeric model features: "
            f"{non_numeric}"
        )

    # --------------------------------------------------------
    # NaN / Inf
    # --------------------------------------------------------

    model_data = df[model_features]

    nan_count = int(
        model_data.isna()
        .sum()
        .sum()
    )

    values = model_data.to_numpy(
        dtype=float
    )

    inf_count = int(
        np.isinf(values).sum()
    )

    print(
        f"Feature NaNs: {nan_count}"
    )

    print(
        f"Feature Inf: {inf_count}"
    )

    if inf_count == 0:

        check_pass(
            "No infinite feature values"
        )

    else:

        check_fail(
            f"{inf_count} infinite feature values detected"
        )

    if nan_count == 0:

        check_pass(
            "No NaN feature values"
        )

    else:

        check_warn(
            f"{nan_count} NaNs detected; "
            "expected from rolling/temporal feature initialization"
        )

    # --------------------------------------------------------
    # Chronology
    # --------------------------------------------------------

    if (
        "Minute" in df.columns
        and "Source_File" in df.columns
    ):

        all_valid = True

        for source, group in df.groupby(
            "Source_File"
        ):

            times = pd.to_datetime(
                group["Minute"],
                errors="coerce"
            )

            if times.is_monotonic_increasing:

                check_pass(
                    f"Enhanced chronology valid: {source}"
                )

            else:

                check_fail(
                    f"Enhanced chronology invalid: {source}"
                )

                all_valid = False

        if all_valid:

            check_pass(
                "Enhanced dataset chronological within every Source_File"
            )


# ============================================================
# 4. ENHANCED SEQUENCES
# ============================================================

def audit_enhanced_sequences():

    section("4. ENHANCED SEQUENCES")

    paths = {
        "X": (
            ENHANCED_SEQ_DIR
            / "X_enhanced.npy"
        ),

        "forecast15": (
            ENHANCED_SEQ_DIR
            / "y_forecast15_enhanced.npy"
        ),

        "presence5": (
            ENHANCED_SEQ_DIR
            / "y_presence5_enhanced.npy"
        ),

        "onset5": (
            ENHANCED_SEQ_DIR
            / "y_onset5_enhanced.npy"
        ),

        "metadata": (
            ENHANCED_SEQ_DIR
            / "metadata_enhanced.csv"
        ),
    }

    for name, path in paths.items():

        check_file(
            path,
            f"enhanced_sequences/{path.name}"
        )

    if not all(
        path.exists()
        for path in paths.values()
    ):

        return

    X = load_npy(
        paths["X"],
        "Enhanced sequence X"
    )

    y_forecast = load_npy(
        paths["forecast15"],
        "Enhanced forecast15 target"
    )

    y_presence = load_npy(
        paths["presence5"],
        "Enhanced presence5 target"
    )

    y_onset = load_npy(
        paths["onset5"],
        "Enhanced onset5 target"
    )

    metadata = load_csv(
        paths["metadata"],
        "Enhanced sequence metadata"
    )

    if any(
        item is None
        for item in [
            X,
            y_forecast,
            y_presence,
            y_onset,
            metadata,
        ]
    ):

        return

    print(
        f"X shape: {X.shape}"
    )

    print(
        f"Forecast15 shape: {y_forecast.shape}"
    )

    print(
        f"Presence5 shape: {y_presence.shape}"
    )

    print(
        f"Onset5 shape: {y_onset.shape}"
    )

    print(
        f"Metadata shape: {metadata.shape}"
    )

    # --------------------------------------------------------
    # Shape
    # --------------------------------------------------------

    expected_shape = (
        EXPECTED_ENHANCED_SEQUENCE_COUNT,
        HISTORY_WINDOW,
        EXPECTED_ENHANCED_FEATURES,
    )

    if X.shape == expected_shape:

        check_pass(
            f"Enhanced X shape valid: {X.shape}"
        )

    else:

        check_fail(
            f"Enhanced X shape {X.shape}; "
            f"expected {expected_shape}"
        )

    # --------------------------------------------------------
    # Targets
    # --------------------------------------------------------

    targets = [
        ("forecast15", y_forecast),
        ("presence5", y_presence),
        ("onset5", y_onset),
    ]

    for name, target in targets:

        if len(target) == len(X):

            check_pass(
                f"{name} target aligned"
            )

        else:

            check_fail(
                f"{name} target not aligned"
            )

        unique_values = np.unique(
            target
        )

        if np.all(
            np.isin(
                unique_values,
                [0, 1]
            )
        ):

            check_pass(
                f"{name} target is binary"
            )

        else:

            check_fail(
                f"{name} target is not binary: "
                f"{unique_values}"
            )

    # --------------------------------------------------------
    # Tensor finite
    # --------------------------------------------------------

    if np.isfinite(X).all():

        check_pass(
            "Enhanced sequence tensor is finite"
        )

    else:

        check_fail(
            "Enhanced sequence tensor contains NaN/Inf"
        )

    # --------------------------------------------------------
    # Metadata
    #
    # Actual schema:
    #   Source_File
    #   Forecast_Start
    #
    # Do NOT require:
    #   History_End
    #   Forecast_End
    # --------------------------------------------------------

    required_metadata = [
        "Source_File",
        "Forecast_Start",
    ]

    missing = [
        column
        for column in required_metadata
        if column not in metadata.columns
    ]

    if not missing:

        check_pass(
            "Required enhanced sequence metadata columns present"
        )

        check_source_chronology(
            metadata,
            "Sequence"
        )

    else:

        check_fail(
            f"Missing required metadata columns: {missing}"
        )


# ============================================================
# 5. ENHANCED TRAIN / VALIDATION / TEST
# ============================================================

def audit_enhanced_training():

    section(
        "5. ENHANCED TRAIN / VALIDATION / TEST SPLIT"
    )

    split_files = {

        "train": {
            "X": (
                ENHANCED_TRAIN_DIR
                / "X_train.npy"
            ),
            "y": (
                ENHANCED_TRAIN_DIR
                / "y_train.npy"
            ),
            "metadata": (
                ENHANCED_TRAIN_DIR
                / "metadata_train.csv"
            ),
        },

        "validation": {
            "X": (
                ENHANCED_TRAIN_DIR
                / "X_validation.npy"
            ),
            "y": (
                ENHANCED_TRAIN_DIR
                / "y_validation.npy"
            ),
            "metadata": (
                ENHANCED_TRAIN_DIR
                / "metadata_validation.csv"
            ),
        },

        "test": {
            "X": (
                ENHANCED_TRAIN_DIR
                / "X_test.npy"
            ),
            "y": (
                ENHANCED_TRAIN_DIR
                / "y_test.npy"
            ),
            "metadata": (
                ENHANCED_TRAIN_DIR
                / "metadata_test.csv"
            ),
        },
    }

    loaded = {}

    # --------------------------------------------------------
    # Load splits
    # --------------------------------------------------------

    for split_name, files in split_files.items():

        subsection(
            split_name.upper()
        )

        print(
            f"X: {files['X'].name}"
        )

        print(
            f"y: {files['y'].name}"
        )

        print(
            f"metadata: {files['metadata'].name}"
        )

        if not all(
            path.exists()
            for path in files.values()
        ):

            check_fail(
                f"{split_name}: one or more files missing"
            )

            continue

        X = load_npy(
            files["X"],
            f"{split_name} X"
        )

        y = load_npy(
            files["y"],
            f"{split_name} y"
        )

        metadata = load_csv(
            files["metadata"],
            f"{split_name} metadata"
        )

        if any(
            item is None
            for item in [
                X,
                y,
                metadata,
            ]
        ):

            continue

        loaded[split_name] = {
            "X": X,
            "y": y,
            "metadata": metadata,
        }

        print(
            f"X shape: {X.shape}"
        )

        print(
            f"y shape: {y.shape}"
        )

        print(
            f"metadata shape: {metadata.shape}"
        )

        # ----------------------------------------------------
        # Alignment
        # ----------------------------------------------------

        if (
            len(X)
            == len(y)
            == len(metadata)
        ):

            check_pass(
                f"{split_name}: X/y/metadata aligned"
            )

        else:

            check_fail(
                f"{split_name}: X/y/metadata not aligned"
            )

        # ----------------------------------------------------
        # Shape
        # ----------------------------------------------------

        if (
            X.ndim == 3
            and X.shape[1] == HISTORY_WINDOW
            and X.shape[2] == EXPECTED_ENHANCED_FEATURES
        ):

            check_pass(
                f"{split_name}: X shape valid"
            )

        else:

            check_fail(
                f"{split_name}: X shape invalid: {X.shape}"
            )

        # ----------------------------------------------------
        # Finite
        # ----------------------------------------------------

        if np.isfinite(X).all():

            check_pass(
                f"{split_name}: X contains finite values only"
            )

        else:

            check_fail(
                f"{split_name}: X contains NaN/Inf"
            )

        # ----------------------------------------------------
        # Binary target
        # ----------------------------------------------------

        unique_values = np.unique(y)

        if np.all(
            np.isin(
                unique_values,
                [0, 1]
            )
        ):

            check_pass(
                f"{split_name}: binary target"
            )

        else:

            check_fail(
                f"{split_name}: target not binary: "
                f"{unique_values}"
            )

        # ----------------------------------------------------
        # Distribution
        # ----------------------------------------------------

        unique, counts = np.unique(
            y,
            return_counts=True
        )

        distribution = {
            int(value): int(count)
            for value, count
            in zip(
                unique,
                counts
            )
        }

        print(
            "Target distribution: "
            + ", ".join(
                f"{key}={value}"
                for key, value
                in distribution.items()
            )
        )

        # ----------------------------------------------------
        # Chronology
        # ----------------------------------------------------

        check_source_chronology(
            metadata,
            split_name
        )

    # --------------------------------------------------------
    # Cross-split checks
    # --------------------------------------------------------

    required_splits = [
        "train",
        "validation",
        "test",
    ]

    if not all(
        split in loaded
        for split in required_splits
    ):

        check_fail(
            "Cross-split temporal audit could not run"
        )

        return

    train_meta = loaded[
        "train"
    ]["metadata"]

    val_meta = loaded[
        "validation"
    ]["metadata"]

    test_meta = loaded[
        "test"
    ]["metadata"]

    # --------------------------------------------------------
    # Exact timestamp overlap
    # --------------------------------------------------------

    print()
    print(
        "SOURCE-SPECIFIC TIMESTAMP OVERLAP"
    )
    print(
        "-" * 40
    )

    check_exact_timestamp_overlap(
        train_meta,
        val_meta,
        "train",
        "validation"
    )

    check_exact_timestamp_overlap(
        train_meta,
        test_meta,
        "train",
        "test"
    )

    check_exact_timestamp_overlap(
        val_meta,
        test_meta,
        "validation",
        "test"
    )

    # --------------------------------------------------------
    # Start-time embargo
    # --------------------------------------------------------

    print()
    print(
        "SOURCE-SPECIFIC TEMPORAL EMBARGO"
    )
    print(
        "-" * 40
    )

    check_start_time_embargo(
        train_meta,
        val_meta,
        "train",
        "validation"
    )

    check_start_time_embargo(
        train_meta,
        test_meta,
        "train",
        "test"
    )

    check_start_time_embargo(
        val_meta,
        test_meta,
        "validation",
        "test"
    )


# ============================================================
# 6. ENHANCED UNSEEN ATTACK
# ============================================================

def audit_enhanced_unseen():

    section(
        "6. ENHANCED UNSEEN-ATTACK DATASET"
    )

    files = {

        "training_X": (
            ENHANCED_UNSEEN_DIR
            / "X_train.npy"
        ),

        "training_y": (
            ENHANCED_UNSEEN_DIR
            / "y_train.npy"
        ),

        "training_metadata": (
            ENHANCED_UNSEEN_DIR
            / "metadata_train.csv"
        ),

        "unseen_X": (
            ENHANCED_UNSEEN_DIR
            / "X_unseen.npy"
        ),

        "unseen_y": (
            ENHANCED_UNSEEN_DIR
            / "y_unseen.npy"
        ),

        "unseen_metadata": (
            ENHANCED_UNSEEN_DIR
            / "metadata_unseen.csv"
        ),
    }

    all_present = True

    for name, path in files.items():

        if not check_file(
            path,
            f"enhanced_unseen_attack/{path.name}"
        ):

            all_present = False

    if not all_present:
        return

    X_train = load_npy(
        files["training_X"],
        "Unseen training X"
    )

    y_train = load_npy(
        files["training_y"],
        "Unseen training y"
    )

    metadata_train = load_csv(
        files["training_metadata"],
        "Unseen training metadata"
    )

    X_unseen = load_npy(
        files["unseen_X"],
        "Unseen test X"
    )

    y_unseen = load_npy(
        files["unseen_y"],
        "Unseen test y"
    )

    metadata_unseen = load_csv(
        files["unseen_metadata"],
        "Unseen test metadata"
    )

    if any(
        item is None
        for item in [
            X_train,
            y_train,
            metadata_train,
            X_unseen,
            y_unseen,
            metadata_unseen,
        ]
    ):

        return

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    subsection(
        "UNSEEN-ATTACK TRAINING"
    )

    print(
        f"X shape: {X_train.shape}"
    )

    print(
        f"y shape: {y_train.shape}"
    )

    print(
        f"metadata shape: {metadata_train.shape}"
    )

    if (
        len(X_train)
        == len(y_train)
        == len(metadata_train)
    ):

        check_pass(
            "Unseen training X/y/metadata aligned"
        )

    else:

        check_fail(
            "Unseen training X/y/metadata not aligned"
        )

    if (
        X_train.ndim == 3
        and X_train.shape[1:] == (
            HISTORY_WINDOW,
            EXPECTED_ENHANCED_FEATURES
        )
    ):

        check_pass(
            "Unseen training X shape valid"
        )

    else:

        check_fail(
            f"Unseen training X shape invalid: "
            f"{X_train.shape}"
        )

    if np.isfinite(X_train).all():

        check_pass(
            "Unseen training X contains finite values only"
        )

    else:

        check_fail(
            "Unseen training X contains NaN/Inf"
        )

    if np.all(
        np.isin(
            np.unique(y_train),
            [0, 1]
        )
    ):

        check_pass(
            "Unseen training target is binary"
        )

    else:

        check_fail(
            "Unseen training target is not binary"
        )

    unique, counts = np.unique(
        y_train,
        return_counts=True
    )

    print(
        "Training target distribution: "
        + ", ".join(
            f"{int(v)}={int(c)}"
            for v, c
            in zip(
                unique,
                counts
            )
        )
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # The unseen-attack metadata has a different schema.
    # Do not require Forecast_Start here.
    #
    # Instead validate that:
    #   - metadata has rows matching X/y
    #   - Source_File exists
    #   - source values are valid
    # --------------------------------------------------------

    if "Source_File" in metadata_train.columns:

        check_pass(
            "Unseen training metadata contains Source_File"
        )

        source_counts = (
            metadata_train["Source_File"]
            .value_counts()
        )

        print(
            "Training sources:"
        )

        for source, count in source_counts.items():

            print(
                f"  {source}: {count}"
            )

    else:

        check_fail(
            "Unseen training metadata missing Source_File"
        )

    # --------------------------------------------------------
    # Unseen test
    # --------------------------------------------------------

    subsection(
        "UNSEEN ATTACK TEST"
    )

    print(
        f"X shape: {X_unseen.shape}"
    )

    print(
        f"y shape: {y_unseen.shape}"
    )

    print(
        f"metadata shape: {metadata_unseen.shape}"
    )

    if (
        len(X_unseen)
        == len(y_unseen)
        == len(metadata_unseen)
    ):

        check_pass(
            "Unseen test X/y/metadata aligned"
        )

    else:

        check_fail(
            "Unseen test X/y/metadata not aligned"
        )

    if (
        X_unseen.ndim == 3
        and X_unseen.shape[1:] == (
            HISTORY_WINDOW,
            EXPECTED_ENHANCED_FEATURES
        )
    ):

        check_pass(
            "Unseen test X shape valid"
        )

    else:

        check_fail(
            f"Unseen test X shape invalid: "
            f"{X_unseen.shape}"
        )

    if np.isfinite(X_unseen).all():

        check_pass(
            "Unseen test X contains finite values only"
        )

    else:

        check_fail(
            "Unseen test X contains NaN/Inf"
        )

    if np.all(
        np.isin(
            np.unique(y_unseen),
            [0, 1]
        )
    ):

        check_pass(
            "Unseen test target is binary"
        )

    else:

        check_fail(
            "Unseen test target is not binary"
        )

    unique, counts = np.unique(
        y_unseen,
        return_counts=True
    )

    print(
        "Unseen target distribution: "
        + ", ".join(
            f"{int(v)}={int(c)}"
            for v, c
            in zip(
                unique,
                counts
            )
        )
    )

    if "Source_File" in metadata_unseen.columns:

        check_pass(
            "Unseen test metadata contains Source_File"
        )

        source_counts = (
            metadata_unseen["Source_File"]
            .value_counts()
        )

        print(
            "Unseen sources:"
        )

        for source, count in source_counts.items():

            print(
                f"  {source}: {count}"
            )

    else:

        check_fail(
            "Unseen test metadata missing Source_File"
        )

    # --------------------------------------------------------
    # Source separation
    # --------------------------------------------------------

    print()
    print(
        "SOURCE SEPARATION"
    )
    print(
        "-" * 40
    )

    if (
        "Source_File" in metadata_train.columns
        and "Source_File" in metadata_unseen.columns
    ):

        train_sources = set(
            metadata_train["Source_File"]
        )

        unseen_sources = set(
            metadata_unseen["Source_File"]
        )

        overlap = (
            train_sources
            & unseen_sources
        )

        if overlap:

            check_fail(
                "Unseen dataset shares source captures "
                f"with training: {sorted(overlap)}"
            )

        else:

            check_pass(
                "Unseen attack uses source captures "
                "not used for training"
            )

        info(
            "Training sources: "
            + ", ".join(
                sorted(train_sources)
            )
        )

        info(
            "Unseen sources: "
            + ", ".join(
                sorted(unseen_sources)
            )
        )


# ============================================================
# 7. TRAINED MODELS
# ============================================================

def audit_models():

    section("7. TRAINED MODELS")

    expected_models = [

        "xgboost_forecast15.json",

        "xgboost_forecast15_v2.json",

        "xgboost_enhanced_forecast15_v2.json",

        "xgboost_unseen_infilteration.json",

        "xgboost_enhanced_unseen_infilteration.json",

        "pytorch_gru_forecast15.pt",

        "pytorch_gru_unseen_infilteration.pt",
    ]

    for filename in expected_models:

        path = MODELS_DIR / filename

        if not path.exists():

            check_fail(
                f"Missing model: {filename}"
            )

            continue

        size_kb = (
            path.stat().st_size
            / 1024
        )

        if size_kb > 0:

            check_pass(
                f"{filename} ({size_kb:.1f} KB)"
            )

        else:

            check_fail(
                f"{filename} exists but is empty"
            )


# ============================================================
# 8. PYTHON ENVIRONMENT
# ============================================================

def audit_environment():

    section("8. PYTHON ENVIRONMENT")

    info(
        f"Python executable: {sys.executable}"
    )

    info(
        f"Python version: "
        f"{sys.version.split()[0]}"
    )

    required_packages = [
        "numpy",
        "pandas",
        "sklearn",
        "xgboost",
        "torch",
    ]

    for package_name in required_packages:

        try:

            module = importlib.import_module(
                package_name
            )

            version = getattr(
                module,
                "__version__",
                "unknown"
            )

            check_pass(
                f"{package_name}: {version}"
            )

        except Exception as exc:

            check_fail(
                f"{package_name}: import failed: {exc}"
            )


# ============================================================
# 9. SOURCE SCRIPT INVENTORY
# ============================================================

def audit_scripts():

    section("9. SOURCE SCRIPT INVENTORY")

    expected_scripts = [

        "analyze_attack_generalization.py",
        "analyze_dataset.py",
        "analyze_enhanced_feature_importance.py",
        "analyze_enhanced_split.py",
        "analyze_enhanced_unseen_predictions.py",
        "analyze_feature_generalization.py",
        "analyze_unseen_predictions.py",
        "audit_project_pipeline.py",

        "create_enhanced_sequences.py",
        "create_enhanced_training_data.py",
        "create_enhanced_unseen_attack_dataset.py",
        "create_sequences.py",
        "create_temporal_dataset.py",
        "create_temporal_features.py",
        "create_unseen_attack_dataset.py",

        "explore_dataset.py",
        "inspect_dataset.py",
        "prepare_training_data.py",
        "preprocess_dataset.py",

        "train_enhanced_unseen_attack_xgboost.py",
        "train_enhanced_xgboost.py",
        "train_pytorch.py",
        "train_unseen_attack_pytorch.py",
        "train_unseen_attack_xgboost.py",
        "train_xgboost.py",
        "train_xgboost_v2.py",
    ]

    missing = []

    for filename in expected_scripts:

        path = (
            PROJECT_ROOT
            / "src"
            / filename
        )

        if path.exists():

            check_pass(
                f"Script exists: {filename}"
            )

        else:

            missing.append(
                filename
            )

    if not missing:

        check_pass(
            f"All {len(expected_scripts)} "
            "expected project scripts are present"
        )

    else:

        check_fail(
            f"Missing {len(missing)} scripts: "
            f"{missing}"
        )


# ============================================================
# 10. ARTIFACT SUMMARY
# ============================================================

def audit_artifact_summary():

    section(
        "10. PROJECT ARTIFACT SUMMARY"
    )

    directories = [

        (
            "Temporal",
            TEMPORAL_DIR
        ),

        (
            "Enhanced sequences",
            ENHANCED_SEQ_DIR
        ),

        (
            "Enhanced training",
            ENHANCED_TRAIN_DIR
        ),

        (
            "Enhanced unseen",
            ENHANCED_UNSEEN_DIR
        ),

        (
            "Models",
            MODELS_DIR
        ),

        (
            "Source scripts",
            PROJECT_ROOT / "src"
        ),
    ]

    for label, directory in directories:

        if directory.exists():

            count = sum(
                1
                for item in directory.iterdir()
                if item.is_file()
            )

            info(
                f"{label} files: {count}"
            )

        else:

            check_warn(
                f"{label} directory missing: "
                f"{directory}"
            )


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_summary():

    section(
        "AUDIT SUMMARY"
    )

    print(
        f"PASS : {PASS_COUNT}"
    )

    print(
        f"WARN : {WARN_COUNT}"
    )

    print(
        f"FAIL : {FAIL_COUNT}"
    )

    print()

    if FAIL_COUNT == 0:

        if WARN_COUNT == 0:

            print(
                "[SUCCESS] COMPLETE PIPELINE AUDIT PASSED."
            )

        else:

            print(
                "[SUCCESS] PIPELINE AUDIT PASSED "
                "WITH WARNINGS."
            )

    else:

        print(
            "[ACTION REQUIRED] "
            "Pipeline audit contains failures."
        )

    print()

    print(
        "Project root:"
    )

    print(
        PROJECT_ROOT
    )

    print()

    print(
        "Key temporal assumptions:"
    )

    print(
        f"- History window: "
        f"{HISTORY_WINDOW} minutes"
    )

    print(
        f"- Forecast horizon: "
        f"{FORECAST_HORIZON_MINUTES} minutes"
    )

    print(
        f"- Required split start-to-start embargo: "
        f"{EMBARGO_MINUTES} minutes"
    )

    print()

    print(
        "Metadata handling:"
    )

    print(
        "- Main enhanced metadata requires "
        "Source_File + Forecast_Start"
    )

    print(
        "- Forecast_End is derived only when needed"
    )

    print(
        "- Unseen-attack metadata is validated "
        "using its actual schema"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 75
    )

    print(
        "NETWORK ATTACK FORECASTING PROJECT AUDIT"
    )

    print(
        "=" * 75
    )

    print()

    info(
        f"Project root: {PROJECT_ROOT}"
    )

    info(
        f"Processed data: {PROCESSED_DIR}"
    )

    info(
        f"Enhanced sequences: {ENHANCED_SEQ_DIR}"
    )

    info(
        f"Enhanced training: {ENHANCED_TRAIN_DIR}"
    )

    info(
        f"Enhanced unseen: {ENHANCED_UNSEEN_DIR}"
    )

    info(
        f"Models: {MODELS_DIR}"
    )

    info(
        f"Python: {sys.executable}"
    )

    # --------------------------------------------------------
    # Run audits
    # --------------------------------------------------------

    audit_required_files()

    audit_network_states()

    audit_enhanced_features()

    audit_enhanced_sequences()

    audit_enhanced_training()

    audit_enhanced_unseen()

    audit_models()

    audit_environment()

    audit_scripts()

    audit_artifact_summary()

    print_summary()

    # Exit code:
    #   0 = no failures
    #   1 = one or more failures
    return 1 if FAIL_COUNT > 0 else 0


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    raise SystemExit(
        main()
    )