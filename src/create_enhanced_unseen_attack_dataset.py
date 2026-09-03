"""
Create enhanced unseen-attack dataset.

Training attacks:
    Wednesday-14-02-2018:
        SSH-Bruteforce
        FTP-BruteForce

    Friday-02-03-2018:
        Bot

Completely unseen test attack:
    Thursday-01-03-2018:
        Infilteration

Uses:
    161 enhanced temporal/behaviour features
    10-minute history
    15-minute forecast horizon

Important:
    Infilteration is NEVER included in training.
"""

import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed/temporal/network_states_enhanced.csv"

OUTPUT_DIR = "data/processed/enhanced_unseen_attack"

HISTORY_MINUTES = 10
FORECAST_MINUTES = 15

# Source files
UNSEEN_SOURCE = "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"

TRAIN_SOURCES = [
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
]


# ============================================================
# HELPERS
# ============================================================

def source_name(path):
    """Return only the filename portion."""
    return os.path.basename(str(path))


def create_sequences_for_source(df, feature_columns):
    """
    Create 10-minute history -> 15-minute future attack
    sequences for one source capture.

    Each sequence:
        X = previous/current 10 minutes
        y = whether an attack occurs within next 15 minutes
    """

    df = df.sort_values("Minute").reset_index(drop=True)

    # Convert time column
    df["Minute"] = pd.to_datetime(df["Minute"], errors="coerce")

    # Remove rows with invalid timestamps
    df = df.dropna(subset=["Minute"]).reset_index(drop=True)

    X_list = []
    y_list = []
    metadata = []

    skipped_nan = 0
    skipped_gap = 0
    skipped_future = 0

    for i in range(HISTORY_MINUTES - 1, len(df)):

        # ----------------------------------------------------
        # History window
        # ----------------------------------------------------

        history_start = i - HISTORY_MINUTES + 1
        history_end = i + 1

        history = df.iloc[history_start:history_end]

        if len(history) != HISTORY_MINUTES:
            continue

        # ----------------------------------------------------
        # Check history timestamps are continuous
        # ----------------------------------------------------

        history_times = history["Minute"].values

        history_diffs = np.diff(
            history["Minute"].astype("int64") // 10**9
        )

        # Network states should normally be one minute apart.
        # Allow a small tolerance but reject major gaps.
        if len(history_diffs) > 0:
            if np.any(history_diffs > 90):
                skipped_gap += 1
                continue

        # ----------------------------------------------------
        # Future 15-minute window
        # ----------------------------------------------------

        current_time = df.iloc[i]["Minute"]

        future_end = current_time + pd.Timedelta(
            minutes=FORECAST_MINUTES
        )

        future = df[
            (df["Minute"] > current_time)
            & (df["Minute"] <= future_end)
        ]

        if len(future) == 0:
            skipped_future += 1
            continue

        # Require the future window to actually reach
        # approximately the requested horizon.
        latest_future_time = future["Minute"].max()

        if latest_future_time < (
            current_time
            + pd.Timedelta(minutes=FORECAST_MINUTES - 1)
        ):
            skipped_future += 1
            continue

        # ----------------------------------------------------
        # Extract features
        # ----------------------------------------------------

        X_window = history[feature_columns].values.astype(np.float32)

        # Reject NaN / inf
        if not np.isfinite(X_window).all():
            skipped_nan += 1
            continue

        # ----------------------------------------------------
        # Future attack target
        # ----------------------------------------------------

        # Attack_Flow_Count is label-derived and is NOT an input
        # feature. It is used only to construct the future target.
        attack_count = future["Attack_Flow_Count"].fillna(0)

        y = int((attack_count > 0).any())

        X_list.append(X_window)
        y_list.append(y)

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        future_attack_types = future.loc[
            future["Attack_Flow_Count"] > 0,
            "Attack_Type"
        ].dropna().astype(str).unique()

        if len(future_attack_types) == 0:
            future_attack_type = "Benign"
        else:
            future_attack_type = "|".join(
                sorted(future_attack_types)
            )

        metadata.append({
            "Source_File": source_name(
                df.iloc[i]["Source_File"]
            ),
            "History_End": current_time,
            "Forecast_End": future_end,
            "Future_Attack": y,
            "Future_Attack_Type": future_attack_type,
        })

    print(
        f"    Created: {len(X_list)} sequences"
    )
    print(
        f"    Skipped NaN/inf: {skipped_nan}"
    )
    print(
        f"    Skipped time gaps: {skipped_gap}"
    )
    print(
        f"    Skipped incomplete future windows: "
        f"{skipped_future}"
    )

    if len(X_list) == 0:
        return None, None, None

    return (
        np.stack(X_list),
        np.array(y_list, dtype=np.int64),
        pd.DataFrame(metadata),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("ENHANCED UNSEEN ATTACK DATASET")
    print("=" * 80)

    print("\nLoading enhanced temporal dataset:")
    print(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows loaded: {len(df)}")
    print(f"Columns loaded: {len(df.columns)}")

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    required_columns = [
        "Minute",
        "Attack_Flow_Count",
        "Attack_Type",
        "Attack_State",
        "Source_File",
    ]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # --------------------------------------------------------
    # Identify model features
    # --------------------------------------------------------

    excluded_columns = {
        "Minute",
        "Attack_Flow_Count",
        "Attack_Ratio",
        "Attack_State",
        "Attack_Type",
        "Source_File",
    }

    feature_columns = [
        c for c in df.columns
        if c not in excluded_columns
    ]

    print("\nFeature configuration:")
    print(f"Model features: {len(feature_columns)}")

    if len(feature_columns) != 161:
        raise ValueError(
            f"Expected 161 enhanced features, "
            f"found {len(feature_columns)}"
        )

    print("✓ Exactly 161 model features detected")

    # --------------------------------------------------------
    # Show source distribution
    # --------------------------------------------------------

    print("\nSource files:")

    for source in sorted(df["Source_File"].unique()):
        count = (
            df["Source_File"]
            .astype(str)
            .map(source_name)
            .eq(source_name(source))
            .sum()
        )

        print(f"  {source_name(source)}: {count} rows")

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --------------------------------------------------------
    # TRAINING DATA
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("CREATING TRAINING DATA")
    print("=" * 80)

    train_X = []
    train_y = []
    train_metadata = []

    for source in TRAIN_SOURCES:

        print(f"\nProcessing:")
        print(f"  {source}")

        mask = (
            df["Source_File"]
            .astype(str)
            .map(source_name)
            == source_name(source)
        )

        source_df = df.loc[mask].copy()

        if len(source_df) == 0:
            raise ValueError(
                f"Training source not found: {source}"
            )

        X, y, metadata = create_sequences_for_source(
            source_df,
            feature_columns
        )

        if X is None:
            raise ValueError(
                f"No sequences created for {source}"
            )

        train_X.append(X)
        train_y.append(y)
        train_metadata.append(metadata)

    X_train = np.concatenate(train_X, axis=0)
    y_train = np.concatenate(train_y, axis=0)
    metadata_train = pd.concat(
        train_metadata,
        ignore_index=True
    )

    # --------------------------------------------------------
    # UNSEEN TEST DATA
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("CREATING UNSEEN INFILTERATION TEST DATA")
    print("=" * 80)

    print(f"\nProcessing:")
    print(f"  {UNSEEN_SOURCE}")

    unseen_mask = (
        df["Source_File"]
        .astype(str)
        .map(source_name)
        == source_name(UNSEEN_SOURCE)
    )

    unseen_df = df.loc[unseen_mask].copy()

    if len(unseen_df) == 0:
        raise ValueError(
            f"Unseen source not found: {UNSEEN_SOURCE}"
        )

    X_unseen, y_unseen, metadata_unseen = (
        create_sequences_for_source(
            unseen_df,
            feature_columns
        )
    )

    if X_unseen is None:
        raise ValueError(
            "No unseen sequences were created."
        )

    # --------------------------------------------------------
    # SAVE DATA
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("SAVING DATASET")
    print("=" * 80)

    np.save(
        os.path.join(OUTPUT_DIR, "X_train.npy"),
        X_train
    )

    np.save(
        os.path.join(OUTPUT_DIR, "y_train.npy"),
        y_train
    )

    metadata_train.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "metadata_train.csv"
        ),
        index=False
    )

    np.save(
        os.path.join(OUTPUT_DIR, "X_unseen.npy"),
        X_unseen
    )

    np.save(
        os.path.join(OUTPUT_DIR, "y_unseen.npy"),
        y_unseen
    )

    metadata_unseen.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "metadata_unseen.csv"
        ),
        index=False
    )

    # Save feature names so the experiment is reproducible
    pd.DataFrame({
        "Feature": feature_columns
    }).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "feature_names.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # VALIDATION / SANITY CHECKS
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)

    print("\nTraining:")
    print(f"  X shape: {X_train.shape}")
    print(f"  y shape: {y_train.shape}")

    print("\nUnseen:")
    print(f"  X shape: {X_unseen.shape}")
    print(f"  y shape: {y_unseen.shape}")

    # Feature dimension
    assert X_train.shape[2] == 161
    assert X_unseen.shape[2] == 161

    # History length
    assert X_train.shape[1] == HISTORY_MINUTES
    assert X_unseen.shape[1] == HISTORY_MINUTES

    # Finite values
    assert np.isfinite(X_train).all()
    assert np.isfinite(X_unseen).all()

    print("\n✓ Feature count check passed")
    print("✓ History length check passed")
    print("✓ Finite-value check passed")

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    print("\nTraining target distribution:")
    train_counts = np.bincount(y_train)

    for cls, count in enumerate(train_counts):
        print(f"  {cls}: {count}")

    print("\nUnseen target distribution:")
    unseen_counts = np.bincount(y_unseen)

    for cls, count in enumerate(unseen_counts):
        print(f"  {cls}: {count}")

    # --------------------------------------------------------
    # ATTACK TYPE CONTAMINATION CHECK
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("UNSEEN ATTACK CONTAMINATION CHECK")
    print("=" * 80)

    train_types = set()

    for value in metadata_train["Future_Attack_Type"]:
        for attack_type in str(value).split("|"):
            if attack_type != "Benign":
                train_types.add(attack_type)

    unseen_types = set()

    for value in metadata_unseen["Future_Attack_Type"]:
        for attack_type in str(value).split("|"):
            if attack_type != "Benign":
                unseen_types.add(attack_type)

    print("\nTraining future attack types:")
    for attack_type in sorted(train_types):
        print(f"  {attack_type}")

    print("\nUnseen future attack types:")
    for attack_type in sorted(unseen_types):
        print(f"  {attack_type}")

    if UNSEEN_SOURCE in TRAIN_SOURCES:
        raise AssertionError(
            "Unseen source accidentally included in training!"
        )

    if "Infilteration" in train_types:
        raise AssertionError(
            "CRITICAL: Infilteration contamination "
            "detected in training labels!"
        )

    print("\n✓ Infilteration is completely unseen")
    print("✓ No attack-label contamination detected")

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nFiles created:")
    print("  X_train.npy")
    print("  y_train.npy")
    print("  metadata_train.csv")
    print("  X_unseen.npy")
    print("  y_unseen.npy")
    print("  metadata_unseen.csv")
    print("  feature_names.csv")

    print("\nTraining attacks:")
    print("  Bot")
    print("  FTP-BruteForce")
    print("  SSH-Bruteforce")

    print("\nCompletely unseen attack:")
    print("  Infilteration")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()