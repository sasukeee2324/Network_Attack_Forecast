from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/processed/temporal/network_states_enhanced.csv"
)

OUTPUT_DIR = Path(
    "data/processed/enhanced_sequences"
)

HISTORY_MINUTES = 10
FORECAST_HORIZON = 15
PRESENCE_HORIZON = 5


# ============================================================
# COLUMNS THAT MUST NEVER ENTER THE MODEL
# ============================================================

NON_FEATURE_COLUMNS = [
    "Minute",
    "Source_File",
    "Attack_Flow_Count",
    "Attack_Ratio",
    "Attack_Type",
    "Attack_State",
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("CREATING ENHANCED TEMPORAL FORECASTING SEQUENCES")
    print("=" * 80)

    print(f"\nHistory window: {HISTORY_MINUTES} minutes")
    print(f"Main forecast horizon: {FORECAST_HORIZON} minutes")
    print(f"Presence horizon: {PRESENCE_HORIZON} minutes")

    # ========================================================
    # LOAD DATA
    # ========================================================

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nEnhanced dataset not found:\n{INPUT_FILE}\n"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"\nEnhanced states loaded: {len(df):,}")
    print(f"Input columns: {len(df.columns)}")

    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    required_columns = [
        "Minute",
        "Source_File",
        "Attack_State",
        "Attack_Type",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "\nMissing required columns:\n"
            + "\n".join(missing)
        )

    # ========================================================
    # TIMESTAMP
    # ========================================================

    df["Minute"] = pd.to_datetime(
        df["Minute"],
        errors="coerce"
    )

    if df["Minute"].isna().any():
        raise ValueError(
            "Invalid Minute values detected."
        )

    # ========================================================
    # SORT
    # ========================================================

    df = df.sort_values(
        ["Source_File", "Minute"]
    ).reset_index(drop=True)

    # ========================================================
    # MODEL FEATURES
    # ========================================================

    feature_columns = [
        column
        for column in df.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    print(f"\nModel features: {len(feature_columns)}")

    # ========================================================
    # LEAKAGE CHECK
    # ========================================================

    leakage_columns = [
        "Attack_Flow_Count",
        "Attack_Ratio",
        "Attack_State",
        "Attack_Type",
    ]

    leakage_found = [
        column
        for column in leakage_columns
        if column in feature_columns
    ]

    if leakage_found:
        raise RuntimeError(
            "\nLEAKAGE DETECTED!\n"
            + "\n".join(leakage_found)
        )

    print("Leakage check: PASSED")

    # ========================================================
    # NUMERIC FEATURES
    # ========================================================

    for feature in feature_columns:
        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    df[feature_columns] = df[
        feature_columns
    ].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # ========================================================
    # SEQUENCE STORAGE
    # ========================================================

    X_sequences = []

    y_forecast15 = []
    y_presence5 = []
    y_onset5 = []

    metadata = []

    total_possible = 0
    total_nan_skipped = 0
    total_time_skipped = 0

    # ========================================================
    # PROCESS EACH CAPTURE SEPARATELY
    # ========================================================

    for source, group in df.groupby(
        "Source_File",
        sort=False
    ):

        group = group.sort_values(
            "Minute"
        ).reset_index(drop=True)

        print("\n" + "-" * 80)
        print(f"Source: {source}")
        print(f"States: {len(group):,}")

        minutes = group["Minute"].to_numpy()

        states = group[
            feature_columns
        ].to_numpy(dtype=np.float32)

        attack_state = group[
            "Attack_State"
        ].to_numpy(dtype=np.int8)

        attack_type = (
            group["Attack_Type"]
            .astype(str)
            .to_numpy()
        )

        # ====================================================
        # MAXIMUM VALID START
        # ====================================================

        max_start = (
            len(group)
            - HISTORY_MINUTES
            - FORECAST_HORIZON
            + 1
        )

        if max_start <= 0:
            print("Not enough states for sequences.")
            continue

        source_sequences = 0
        source_nan_skipped = 0
        source_time_skipped = 0

        # ====================================================
        # CREATE SEQUENCES
        # ====================================================

        for start in range(max_start):

            total_possible += 1

            history_start = start
            history_end = (
                start + HISTORY_MINUTES
            )

            forecast_end = (
                history_end
                + FORECAST_HORIZON
            )

            # ------------------------------------------------
            # HISTORY
            # ------------------------------------------------

            X = states[
                history_start:history_end
            ]

            history_times = minutes[
                history_start:history_end
            ]

            # ------------------------------------------------
            # HISTORY TIME CONTINUITY
            # ------------------------------------------------

            history_time_values = (
                history_times
                .astype("datetime64[m]")
                .astype(np.int64)
            )

            history_diffs = np.diff(
                history_time_values
            )

            if not np.all(
                history_diffs == 1
            ):
                total_time_skipped += 1
                source_time_skipped += 1
                continue

            # ------------------------------------------------
            # FORECAST WINDOW
            # ------------------------------------------------

            forecast_times = minutes[
                history_end:forecast_end
            ]

            if len(forecast_times) != FORECAST_HORIZON:
                continue

            forecast_time_values = (
                forecast_times
                .astype("datetime64[m]")
                .astype(np.int64)
            )

            forecast_diffs = np.diff(
                forecast_time_values
            )

            if not np.all(
                forecast_diffs == 1
            ):
                total_time_skipped += 1
                source_time_skipped += 1
                continue

            # ------------------------------------------------
            # FEATURE VALIDITY
            # ------------------------------------------------

            if not np.isfinite(X).all():
                total_nan_skipped += 1
                source_nan_skipped += 1
                continue

            # ------------------------------------------------
            # 15-MINUTE FORECAST TARGET
            # ------------------------------------------------

            future_attack = attack_state[
                history_end:forecast_end
            ]

            target_forecast15 = int(
                np.any(
                    future_attack == 1
                )
            )

            # ------------------------------------------------
            # 5-MINUTE PRESENCE TARGET
            # ------------------------------------------------

            presence_end = (
                history_end
                + PRESENCE_HORIZON
            )

            future_presence = attack_state[
                history_end:presence_end
            ]

            target_presence5 = int(
                np.any(
                    future_presence == 1
                )
            )

            # ------------------------------------------------
            # 5-MINUTE ONSET TARGET
            # ------------------------------------------------

            target_onset5 = 0

            for j in range(
                history_end,
                presence_end
            ):

                current_attack = attack_state[j]

                previous_attack = (
                    attack_state[j - 1]
                    if j > 0
                    else 0
                )

                if (
                    current_attack == 1
                    and previous_attack == 0
                ):
                    target_onset5 = 1
                    break

            # ------------------------------------------------
            # FUTURE ATTACK TYPES
            # ------------------------------------------------

            future_types = attack_type[
                history_end:forecast_end
            ]

            future_attack_types = []

            for attack in future_types:

                attack_lower = attack.lower()

                if attack_lower in {
                    "benign",
                    "nan",
                    "",
                }:
                    continue

                if attack not in future_attack_types:
                    future_attack_types.append(
                        attack
                    )

            future_attack_types.sort()

            if future_attack_types:
                future_attack_types_text = "|".join(
                    future_attack_types
                )
            else:
                future_attack_types_text = "Benign"

            # ------------------------------------------------
            # STORE
            # ------------------------------------------------

            X_sequences.append(X)

            y_forecast15.append(
                target_forecast15
            )

            y_presence5.append(
                target_presence5
            )

            y_onset5.append(
                target_onset5
            )

            metadata.append(
                {
                    "Forecast_Start": minutes[
                        history_end
                    ],
                    "Source_File": source,
                    "Target_Forecast15": (
                        target_forecast15
                    ),
                    "Target_Presence5": (
                        target_presence5
                    ),
                    "Target_Onset5": (
                        target_onset5
                    ),
                    "Future_Attack_Types": (
                        future_attack_types_text
                    ),
                }
            )

            source_sequences += 1

        # ====================================================
        # SOURCE SUMMARY
        # ====================================================

        print(
            f"Sequences created: "
            f"{source_sequences}"
        )

        if source_nan_skipped:
            print(
                f"Skipped due to NaN/inf: "
                f"{source_nan_skipped}"
            )

        if source_time_skipped:
            print(
                f"Skipped due to time gaps: "
                f"{source_time_skipped}"
            )

    # ========================================================
    # CHECK SEQUENCES
    # ========================================================

    if not X_sequences:
        raise RuntimeError(
            "\nNo valid sequences were created."
        )

    # ========================================================
    # CONVERT ARRAYS
    # ========================================================

    X = np.stack(
        X_sequences
    ).astype(np.float32)

    y_forecast15 = np.asarray(
        y_forecast15,
        dtype=np.int64
    )

    y_presence5 = np.asarray(
        y_presence5,
        dtype=np.int64
    )

    y_onset5 = np.asarray(
        y_onset5,
        dtype=np.int64
    )

    metadata_df = pd.DataFrame(
        metadata
    )

    metadata_df["Forecast_Start"] = pd.to_datetime(
        metadata_df["Forecast_Start"]
    )

    # ========================================================
    # ALIGNMENT CHECK
    # ========================================================

    sample_count = len(X)

    if not (
        len(y_forecast15)
        == len(y_presence5)
        == len(y_onset5)
        == len(metadata_df)
        == sample_count
    ):
        raise RuntimeError(
            "Sequence alignment check failed."
        )

    print(
        "\nSequence alignment check: PASSED"
    )

    # ========================================================
    # TARGET CONSISTENCY
    # ========================================================

    if not np.array_equal(
        y_forecast15,
        metadata_df[
            "Target_Forecast15"
        ].to_numpy()
    ):
        raise RuntimeError(
            "15-minute target mismatch."
        )

    if not np.array_equal(
        y_presence5,
        metadata_df[
            "Target_Presence5"
        ].to_numpy()
    ):
        raise RuntimeError(
            "5-minute presence target mismatch."
        )

    if not np.array_equal(
        y_onset5,
        metadata_df[
            "Target_Onset5"
        ].to_numpy()
    ):
        raise RuntimeError(
            "5-minute onset target mismatch."
        )

    print(
        "Target consistency check: PASSED"
    )

    # ========================================================
    # FINAL FEATURE SHAPE CHECK
    # ========================================================

    if X.shape[2] != 161:

        raise RuntimeError(
            "\nUnexpected feature count!\n"
            f"Expected: 161\n"
            f"Actual:   {X.shape[2]}"
        )

    print(
        "Feature count check: PASSED"
    )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        OUTPUT_DIR / "X_enhanced.npy",
        X
    )

    np.save(
        OUTPUT_DIR / "y_forecast15_enhanced.npy",
        y_forecast15
    )

    np.save(
        OUTPUT_DIR / "y_presence5_enhanced.npy",
        y_presence5
    )

    np.save(
        OUTPUT_DIR / "y_onset5_enhanced.npy",
        y_onset5
    )

    metadata_df.to_csv(
        OUTPUT_DIR / "metadata_enhanced.csv",
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 80)
    print("ENHANCED SEQUENCE DATASET CREATED")
    print("=" * 80)

    print(
        f"\nTotal possible sequences: "
        f"{total_possible:,}"
    )

    print(
        f"Skipped due to NaN/inf: "
        f"{total_nan_skipped:,}"
    )

    print(
        f"Skipped due to time gaps: "
        f"{total_time_skipped:,}"
    )

    print(
        f"\nFinal sequences: "
        f"{sample_count:,}"
    )

    print(
        f"\nX shape: {X.shape}"
    )

    print(
        f"15-minute target shape: "
        f"{y_forecast15.shape}"
    )

    print(
        f"5-minute presence shape: "
        f"{y_presence5.shape}"
    )

    print(
        f"5-minute onset shape: "
        f"{y_onset5.shape}"
    )

    # ========================================================
    # TARGET DISTRIBUTIONS
    # ========================================================

    print(
        "\n15-MINUTE FORECAST TARGET:"
    )

    forecast_counts = np.bincount(
        y_forecast15,
        minlength=2
    )

    print(
        f"  No attack: "
        f"{forecast_counts[0]:,} "
        f"({forecast_counts[0] / sample_count * 100:.2f}%)"
    )

    print(
        f"  Attack:    "
        f"{forecast_counts[1]:,} "
        f"({forecast_counts[1] / sample_count * 100:.2f}%)"
    )

    print(
        "\n5-MINUTE PRESENCE TARGET:"
    )

    presence_counts = np.bincount(
        y_presence5,
        minlength=2
    )

    print(
        f"  No attack: "
        f"{presence_counts[0]:,} "
        f"({presence_counts[0] / sample_count * 100:.2f}%)"
    )

    print(
        f"  Attack:    "
        f"{presence_counts[1]:,} "
        f"({presence_counts[1] / sample_count * 100:.2f}%)"
    )

    print(
        "\n5-MINUTE ONSET TARGET:"
    )

    onset_counts = np.bincount(
        y_onset5,
        minlength=2
    )

    print(
        f"  No onset: "
        f"{onset_counts[0]:,} "
        f"({onset_counts[0] / sample_count * 100:.2f}%)"
    )

    print(
        f"  Attack onset: "
        f"{onset_counts[1]:,} "
        f"({onset_counts[1] / sample_count * 100:.2f}%)"
    )

    # ========================================================
    # SOURCE DISTRIBUTION
    # ========================================================

    print(
        "\nSequences per source:"
    )

    print(
        metadata_df[
            "Source_File"
        ]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # ATTACK TYPE DISTRIBUTION
    # ========================================================

    print(
        "\nFuture attack types:"
    )

    attack_type_counts = {}

    for value in metadata_df[
        "Future_Attack_Types"
    ]:

        for attack in str(value).split("|"):

            if attack == "Benign":
                continue

            attack_type_counts[attack] = (
                attack_type_counts.get(
                    attack,
                    0
                ) + 1
            )

    if attack_type_counts:

        for attack, count in sorted(
            attack_type_counts.items(),
            key=lambda item: item[1],
            reverse=True
        ):

            print(
                f"  {attack}: {count:,}"
            )

    else:
        print("  None")

    # ========================================================
    # FILES
    # ========================================================

    print(
        "\nFiles saved:"
    )

    print(
        f"  {OUTPUT_DIR / 'X_enhanced.npy'}"
    )

    print(
        f"  {OUTPUT_DIR / 'y_forecast15_enhanced.npy'}"
    )

    print(
        f"  {OUTPUT_DIR / 'y_presence5_enhanced.npy'}"
    )

    print(
        f"  {OUTPUT_DIR / 'y_onset5_enhanced.npy'}"
    )

    print(
        f"  {OUTPUT_DIR / 'metadata_enhanced.csv'}"
    )

    print("\n" + "=" * 80)
    print("ENHANCED SEQUENCE CREATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()