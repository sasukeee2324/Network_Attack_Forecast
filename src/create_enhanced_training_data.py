from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path(
    "data/processed/enhanced_sequences"
)

OUTPUT_DIR = Path(
    "data/processed/enhanced_training"
)

X_FILE = INPUT_DIR / "X_enhanced.npy"
Y_FILE = INPUT_DIR / "y_forecast15_enhanced.npy"
PRESENCE_FILE = (
    INPUT_DIR / "y_presence5_enhanced.npy"
)
ONSET_FILE = (
    INPUT_DIR / "y_onset5_enhanced.npy"
)
METADATA_FILE = (
    INPUT_DIR / "metadata_enhanced.csv"
)


# ------------------------------------------------------------
# Split configuration
# ------------------------------------------------------------

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# The history is 10 minutes and the forecast is 15 minutes.
# We keep a 25-minute temporal embargo between partitions.
EMBARGO_MINUTES = 25


# ============================================================
# HELPERS
# ============================================================

def print_distribution(name, y):

    total = len(y)

    benign = int(
        np.sum(y == 0)
    )

    attack = int(
        np.sum(y == 1)
    )

    print(f"\n{name}:")
    print(
        f"  Total:  {total}"
    )

    print(
        f"  Benign: {benign} "
        f"({benign / total * 100:.2f}%)"
        if total > 0
        else "  Benign: 0"
    )

    print(
        f"  Attack: {attack} "
        f"({attack / total * 100:.2f}%)"
        if total > 0
        else "  Attack: 0"
    )


def describe_source_split(
    source,
    train,
    validation,
    test,
):

    print("\n" + "-" * 80)
    print(f"Source: {source}")
    print("-" * 80)

    print(
        f"Original sequences: "
        f"{len(train) + len(validation) + len(test)}"
    )

    for name, data in [
        ("Train", train),
        ("Validation", validation),
        ("Test", test),
    ]:

        if len(data) == 0:

            print(
                f"{name}: 0 samples"
            )

            continue

        benign = int(
            np.sum(
                data[
                    "Target_Forecast15"
                ].to_numpy()
                == 0
            )
        )

        attack = int(
            np.sum(
                data[
                    "Target_Forecast15"
                ].to_numpy()
                == 1
            )
        )

        print(
            f"{name}: {len(data)} "
            f"(Benign={benign}, Attack={attack})"
        )

        print(
            f"  Range: "
            f"{data['Forecast_Start'].iloc[0]}"
            f" → "
            f"{data['Forecast_Start'].iloc[-1]}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("CREATING ENHANCED TRAINING DATA")
    print("=" * 80)

    print(
        "\nStrategy:"
    )

    print(
        "  Chronological split"
    )

    print(
        "  Per-capture splitting"
    )

    print(
        "  25-minute embargo"
    )

    print(
        "  No random shuffling"
    )

    print(
        "\nRatios:"
    )

    print(
        f"  Train:      {TRAIN_RATIO:.0%}"
    )

    print(
        f"  Validation: {VALIDATION_RATIO:.0%}"
    )

    print(
        f"  Test:       {TEST_RATIO:.0%}"
    )

    # ========================================================
    # LOAD
    # ========================================================

    for file in [
        X_FILE,
        Y_FILE,
        PRESENCE_FILE,
        ONSET_FILE,
        METADATA_FILE,
    ]:

        if not file.exists():

            raise FileNotFoundError(
                f"\nRequired file not found:\n{file}"
            )

    X = np.load(
        X_FILE
    )

    y = np.load(
        Y_FILE
    )

    y_presence = np.load(
        PRESENCE_FILE
    )

    y_onset = np.load(
        ONSET_FILE
    )

    metadata = pd.read_csv(
        METADATA_FILE,
        parse_dates=[
            "Forecast_Start"
        ]
    )

    # ========================================================
    # BASIC CHECKS
    # ========================================================

    print(
        "\nInput shapes:"
    )

    print(
        f"X:        {X.shape}"
    )

    print(
        f"Forecast: {y.shape}"
    )

    print(
        f"Presence: {y_presence.shape}"
    )

    print(
        f"Onset:    {y_onset.shape}"
    )

    print(
        f"Metadata: {metadata.shape}"
    )

    if X.ndim != 3:

        raise ValueError(
            "X must be 3-dimensional."
        )

    sample_count = X.shape[0]

    if len(y) != sample_count:

        raise ValueError(
            "X/y sample mismatch."
        )

    if len(y_presence) != sample_count:

        raise ValueError(
            "X/presence sample mismatch."
        )

    if len(y_onset) != sample_count:

        raise ValueError(
            "X/onset sample mismatch."
        )

    if len(metadata) != sample_count:

        raise ValueError(
            "X/metadata sample mismatch."
        )

    if X.shape[2] != 161:

        raise ValueError(
            f"Expected 161 features, "
            f"got {X.shape[2]}."
        )

    # ========================================================
    # TARGET CONSISTENCY
    # ========================================================

    if not np.array_equal(
        y,
        metadata[
            "Target_Forecast15"
        ].to_numpy()
    ):

        raise ValueError(
            "15-minute target mismatch."
        )

    if not np.array_equal(
        y_presence,
        metadata[
            "Target_Presence5"
        ].to_numpy()
    ):

        raise ValueError(
            "5-minute presence target mismatch."
        )

    if not np.array_equal(
        y_onset,
        metadata[
            "Target_Onset5"
        ].to_numpy()
    ):

        raise ValueError(
            "5-minute onset target mismatch."
        )

    print(
        "\nTarget consistency check: PASSED"
    )

    # ========================================================
    # FEATURE VALIDITY
    # ========================================================

    if not np.isfinite(X).all():

        raise ValueError(
            "X contains NaN or infinity."
        )

    print(
        "Feature validity check: PASSED"
    )

    # ========================================================
    # GLOBAL ORDER
    # ========================================================

    metadata = metadata.copy()

    metadata["_Original_Index"] = np.arange(
        len(metadata)
    )

    # ========================================================
    # CREATE SPLITS
    # ========================================================

    train_indices = []
    validation_indices = []
    test_indices = []

    split_rows = []

    print(
        "\n" + "=" * 80
    )

    print(
        "CREATING CHRONOLOGICAL SPLITS"
    )

    print(
        "=" * 80
    )

    for source, group in metadata.groupby(
        "Source_File",
        sort=False
    ):

        group = group.sort_values(
            "Forecast_Start"
        ).reset_index(drop=False)

        n = len(group)

        # ----------------------------------------------------
        # Raw chronological boundaries
        # ----------------------------------------------------

        train_count = int(
            n * TRAIN_RATIO
        )

        validation_count = int(
            n * VALIDATION_RATIO
        )

        train_raw = group.iloc[
            :train_count
        ]

        validation_raw = group.iloc[
            train_count:
            train_count + validation_count
        ]

        test_raw = group.iloc[
            train_count + validation_count:
        ]

        if len(train_raw) == 0:
            raise RuntimeError(
                f"Empty training split for {source}"
            )

        if len(validation_raw) == 0:
            raise RuntimeError(
                f"Empty validation split for {source}"
            )

        if len(test_raw) == 0:
            raise RuntimeError(
                f"Empty test split for {source}"
            )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_last_time = (
            train_raw[
                "Forecast_Start"
            ].iloc[-1]
        )

        # ----------------------------------------------------
        # VALIDATION EMBARGO
        # ----------------------------------------------------

        validation_allowed_time = (
            train_last_time
            + pd.Timedelta(
                minutes=EMBARGO_MINUTES
            )
        )

        validation = validation_raw[
            validation_raw[
                "Forecast_Start"
            ]
            >= validation_allowed_time
        ].copy()

        if len(validation) == 0:

            raise RuntimeError(
                f"No validation samples remain "
                f"after embargo for {source}."
            )

        # ----------------------------------------------------
        # TEST EMBARGO
        # ----------------------------------------------------

        validation_last_time = (
            validation[
                "Forecast_Start"
            ].iloc[-1]
        )

        test_allowed_time = (
            validation_last_time
            + pd.Timedelta(
                minutes=EMBARGO_MINUTES
            )
        )

        test = test_raw[
            test_raw[
                "Forecast_Start"
            ]
            >= test_allowed_time
        ].copy()

        if len(test) == 0:

            raise RuntimeError(
                f"No test samples remain "
                f"after embargo for {source}."
            )

        # ----------------------------------------------------
        # ORIGINAL INDICES
        # ----------------------------------------------------

        train_original = (
            train_raw[
                "_Original_Index"
            ].to_numpy()
        )

        validation_original = (
            validation[
                "_Original_Index"
            ].to_numpy()
        )

        test_original = (
            test[
                "_Original_Index"
            ].to_numpy()
        )

        train_indices.extend(
            train_original.tolist()
        )

        validation_indices.extend(
            validation_original.tolist()
        )

        test_indices.extend(
            test_original.tolist()
        )

        # ----------------------------------------------------
        # REPORT
        # ----------------------------------------------------

        describe_source_split(
            source,
            train_raw,
            validation,
            test
        )

        train_val_gap = (
            validation[
                "Forecast_Start"
            ].iloc[0]
            - train_raw[
                "Forecast_Start"
            ].iloc[-1]
        )

        val_test_gap = (
            test[
                "Forecast_Start"
            ].iloc[0]
            - validation[
                "Forecast_Start"
            ].iloc[-1]
        )

        print(
            f"  Train → Validation gap: "
            f"{train_val_gap}"
        )

        print(
            f"  Validation → Test gap: "
            f"{val_test_gap}"
        )

        # ----------------------------------------------------
        # SAVE SPLIT METADATA
        # ----------------------------------------------------

        for idx in train_original:

            split_rows.append(
                {
                    "Original_Index": int(idx),
                    "Split": "Train",
                    "Source_File": source,
                }
            )

        for idx in validation_original:

            split_rows.append(
                {
                    "Original_Index": int(idx),
                    "Split": "Validation",
                    "Source_File": source,
                }
            )

        for idx in test_original:

            split_rows.append(
                {
                    "Original_Index": int(idx),
                    "Split": "Test",
                    "Source_File": source,
                }
            )

    # ========================================================
    # ARRAYS
    # ========================================================

    train_indices = np.asarray(
        train_indices,
        dtype=int
    )

    validation_indices = np.asarray(
        validation_indices,
        dtype=int
    )

    test_indices = np.asarray(
        test_indices,
        dtype=int
    )

    # ========================================================
    # OVERLAP CHECK
    # ========================================================

    train_set = set(
        train_indices.tolist()
    )

    validation_set = set(
        validation_indices.tolist()
    )

    test_set = set(
        test_indices.tolist()
    )

    if train_set & validation_set:

        raise RuntimeError(
            "Train/validation overlap detected."
        )

    if train_set & test_set:

        raise RuntimeError(
            "Train/test overlap detected."
        )

    if validation_set & test_set:

        raise RuntimeError(
            "Validation/test overlap detected."
        )

    print(
        "\nSplit overlap check: PASSED"
    )

    # ========================================================
    # EXTRACT DATA
    # ========================================================

    X_train = X[
        train_indices
    ]

    X_validation = X[
        validation_indices
    ]

    X_test = X[
        test_indices
    ]

    y_train = y[
        train_indices
    ]

    y_validation = y[
        validation_indices
    ]

    y_test = y[
        test_indices
    ]

    presence_train = y_presence[
        train_indices
    ]

    presence_validation = y_presence[
        validation_indices
    ]

    presence_test = y_presence[
        test_indices
    ]

    onset_train = y_onset[
        train_indices
    ]

    onset_validation = y_onset[
        validation_indices
    ]

    onset_test = y_onset[
        test_indices
    ]

    # ========================================================
    # FINAL SIZE
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "FINAL SPLIT SIZES"
    )

    print(
        "=" * 80
    )

    print(
        f"\nX_train:       {X_train.shape}"
    )

    print(
        f"X_validation:  {X_validation.shape}"
    )

    print(
        f"X_test:        {X_test.shape}"
    )

    # ========================================================
    # DISTRIBUTIONS
    # ========================================================

    print_distribution(
        "TRAIN - 15 minute forecast",
        y_train
    )

    print_distribution(
        "VALIDATION - 15 minute forecast",
        y_validation
    )

    print_distribution(
        "TEST - 15 minute forecast",
        y_test
    )

    print_distribution(
        "TRAIN - 5 minute presence",
        presence_train
    )

    print_distribution(
        "VALIDATION - 5 minute presence",
        presence_validation
    )

    print_distribution(
        "TEST - 5 minute presence",
        presence_test
    )

    print_distribution(
        "TRAIN - 5 minute onset",
        onset_train
    )

    print_distribution(
        "VALIDATION - 5 minute onset",
        onset_validation
    )

    print_distribution(
        "TEST - 5 minute onset",
        onset_test
    )

    # ========================================================
    # REPRESENTATIVENESS WARNING
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "REPRESENTATIVENESS CHECK"
    )

    print(
        "=" * 80
    )

    for name, target in [
        ("Train", y_train),
        ("Validation", y_validation),
        ("Test", y_test),
    ]:

        unique = np.unique(
            target
        )

        if len(unique) < 2:

            print(
                f"WARNING: {name} contains "
                f"only one target class."
            )

        else:

            print(
                f"{name}: contains both classes."
            )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        OUTPUT_DIR / "X_train.npy",
        X_train
    )

    np.save(
        OUTPUT_DIR / "X_validation.npy",
        X_validation
    )

    np.save(
        OUTPUT_DIR / "X_test.npy",
        X_test
    )

    np.save(
        OUTPUT_DIR / "y_train.npy",
        y_train
    )

    np.save(
        OUTPUT_DIR / "y_validation.npy",
        y_validation
    )

    np.save(
        OUTPUT_DIR / "y_test.npy",
        y_test
    )

    np.save(
        OUTPUT_DIR / "y_presence_train.npy",
        presence_train
    )

    np.save(
        OUTPUT_DIR / "y_presence_validation.npy",
        presence_validation
    )

    np.save(
        OUTPUT_DIR / "y_presence_test.npy",
        presence_test
    )

    np.save(
        OUTPUT_DIR / "y_onset_train.npy",
        onset_train
    )

    np.save(
        OUTPUT_DIR / "y_onset_validation.npy",
        onset_validation
    )

    np.save(
        OUTPUT_DIR / "y_onset_test.npy",
        onset_test
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata_clean = metadata.drop(
        columns=["_Original_Index"]
    )

    metadata_train = metadata_clean.iloc[
        train_indices
    ].copy()

    metadata_validation = metadata_clean.iloc[
        validation_indices
    ].copy()

    metadata_test = metadata_clean.iloc[
        test_indices
    ].copy()

    metadata_train.to_csv(
        OUTPUT_DIR / "metadata_train.csv",
        index=False
    )

    metadata_validation.to_csv(
        OUTPUT_DIR / "metadata_validation.csv",
        index=False
    )

    metadata_test.to_csv(
        OUTPUT_DIR / "metadata_test.csv",
        index=False
    )

    split_report = pd.DataFrame(
        split_rows
    )

    split_report.to_csv(
        OUTPUT_DIR / "split_indices.csv",
        index=False
    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "FINAL VALIDATION"
    )

    print(
        "=" * 80
    )

    checks = {

        "Train sample count":
            len(X_train)
            == len(y_train),

        "Validation sample count":
            len(X_validation)
            == len(y_validation),

        "Test sample count":
            len(X_test)
            == len(y_test),

        "Train features":
            X_train.shape[2] == 161,

        "Validation features":
            X_validation.shape[2] == 161,

        "Test features":
            X_test.shape[2] == 161,

        "Train finite":
            np.isfinite(X_train).all(),

        "Validation finite":
            np.isfinite(X_validation).all(),

        "Test finite":
            np.isfinite(X_test).all(),
    }

    all_passed = True

    for name, result in checks.items():

        if result:

            print(
                f"{name}: PASSED"
            )

        else:

            print(
                f"{name}: FAILED"
            )

            all_passed = False

    if not all_passed:

        raise RuntimeError(
            "One or more final checks failed."
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "ENHANCED TRAINING DATA CREATED"
    )

    print(
        "=" * 80
    )

    print(
        "\nSaved to:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    print(
        "\nThis directory is now the canonical"
        " enhanced training split."
    )

    print(
        "\nNext:"
    )

    print(
        "Train XGBoost using these exact splits."
    )

    print(
        "Do not create another independent split."
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()