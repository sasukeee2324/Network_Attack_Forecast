from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


SEQUENCE_DIR = Path("data/processed/sequences")
OUTPUT_DIR = Path("data/processed/training")


def chronological_split(
    metadata,
    train_ratio=0.70,
    validation_ratio=0.15,
    embargo_minutes=25
):
    """
    Chronological split performed independently for each
    capture day, with a 25-minute embargo between splits.

    Each sequence contains:

        10 minutes of history
        +
        15 minutes of forecast

    Therefore, neighboring sequences overlap in time.

    The embargo prevents temporal overlap between
    train, validation, and test samples.

    Split:

        70% -> Train
        25-minute embargo
        15% -> Validation
        25-minute embargo
        15% -> Test
    """

    train_indices = []
    validation_indices = []
    test_indices = []

    print(
        f"\nEmbargo between splits: "
        f"{embargo_minutes} minutes"
    )

    for source_file, group in metadata.groupby(
        "Source_File",
        sort=False
    ):

        # --------------------------------------------------------
        # Sort by actual forecast timestamp
        # --------------------------------------------------------

        group = group.sort_values(
            "Forecast_Start"
        )

        indices = group.index.to_numpy()

        n = len(indices)

        # --------------------------------------------------------
        # Calculate chronological boundaries
        # --------------------------------------------------------

        train_end = int(
            n * train_ratio
        )

        validation_end = int(
            n * (train_ratio + validation_ratio)
        )

        # --------------------------------------------------------
        # TRAIN
        # --------------------------------------------------------

        train_part = indices[
            :train_end
        ]

        # --------------------------------------------------------
        # VALIDATION
        #
        # Skip embargo after training.
        # --------------------------------------------------------

        validation_start = min(
            train_end + embargo_minutes,
            validation_end
        )

        validation_part = indices[
            validation_start:validation_end
        ]

        # --------------------------------------------------------
        # TEST
        #
        # Skip embargo after validation.
        # --------------------------------------------------------

        test_start = min(
            validation_end + embargo_minutes,
            n
        )

        test_part = indices[
            test_start:
        ]

        # --------------------------------------------------------
        # Store indices
        # --------------------------------------------------------

        train_indices.extend(
            train_part
        )

        validation_indices.extend(
            validation_part
        )

        test_indices.extend(
            test_part
        )

        # --------------------------------------------------------
        # Report split sizes
        # --------------------------------------------------------

        print("\n" + "-" * 80)

        print(
            f"Source: {source_file}"
        )

        print(
            f"Original sequences: {n:,}"
        )

        print(
            f"Train: {len(train_part):,}"
        )

        print(
            f"Validation: {len(validation_part):,}"
        )

        print(
            f"Test: {len(test_part):,}"
        )

        # --------------------------------------------------------
        # Report actual time ranges
        # --------------------------------------------------------

        if len(train_part) > 0:

            train_start = metadata.loc[
                train_part,
                "Forecast_Start"
            ].min()

            train_end_time = metadata.loc[
                train_part,
                "Forecast_Start"
            ].max()

            print(
                f"Train range: "
                f"{train_start} "
                f"→ "
                f"{train_end_time}"
            )

        if len(validation_part) > 0:

            validation_start_time = metadata.loc[
                validation_part,
                "Forecast_Start"
            ].min()

            validation_end_time = metadata.loc[
                validation_part,
                "Forecast_Start"
            ].max()

            print(
                f"Validation range: "
                f"{validation_start_time} "
                f"→ "
                f"{validation_end_time}"
            )

        if len(test_part) > 0:

            test_start_time = metadata.loc[
                test_part,
                "Forecast_Start"
            ].min()

            test_end_time = metadata.loc[
                test_part,
                "Forecast_Start"
            ].max()

            print(
                f"Test range: "
                f"{test_start_time} "
                f"→ "
                f"{test_end_time}"
            )

        # --------------------------------------------------------
        # Verify the actual gaps
        # --------------------------------------------------------

        if (
            len(train_part) > 0
            and len(validation_part) > 0
        ):

            train_last = metadata.loc[
                train_part,
                "Forecast_Start"
            ].max()

            validation_first = metadata.loc[
                validation_part,
                "Forecast_Start"
            ].min()

            gap = (
                validation_first
                - train_last
            ).total_seconds() / 60

            print(
                f"Train → Validation gap: "
                f"{gap:.0f} minutes"
            )

        if (
            len(validation_part) > 0
            and len(test_part) > 0
        ):

            validation_last = metadata.loc[
                validation_part,
                "Forecast_Start"
            ].max()

            test_first = metadata.loc[
                test_part,
                "Forecast_Start"
            ].min()

            gap = (
                test_first
                - validation_last
            ).total_seconds() / 60

            print(
                f"Validation → Test gap: "
                f"{gap:.0f} minutes"
            )

    return (
        np.asarray(
            train_indices,
            dtype=int
        ),

        np.asarray(
            validation_indices,
            dtype=int
        ),

        np.asarray(
            test_indices,
            dtype=int
        ),
    )


def print_distribution(
    name,
    targets
):
    """
    Print binary target distribution.
    """

    negative = np.sum(
        targets == 0
    )

    positive = np.sum(
        targets == 1
    )

    total = len(targets)

    print(f"\n{name}:")

    print(
        f"  No attack: "
        f"{negative:,} "
        f"({negative / total * 100:.2f}%)"
    )

    print(
        f"  Attack: "
        f"{positive:,} "
        f"({positive / total * 100:.2f}%)"
    )


def scale_sequences(
    X_train,
    X_validation,
    X_test
):
    """
    Fit scaler ONLY on training data.

    Then apply the same scaler to validation/test.
    """

    n_features = X_train.shape[2]

    scaler = StandardScaler()

    # ------------------------------------------------------------
    # Flatten temporal dimensions
    #
    # samples × time × features
    #
    # becomes:
    #
    # (samples*time) × features
    # ------------------------------------------------------------

    X_train_flat = X_train.reshape(
        -1,
        n_features
    )

    X_validation_flat = (
        X_validation.reshape(
            -1,
            n_features
        )
    )

    X_test_flat = X_test.reshape(
        -1,
        n_features
    )

    # ------------------------------------------------------------
    # IMPORTANT:
    # Fit ONLY on training data.
    # ------------------------------------------------------------

    scaler.fit(
        X_train_flat
    )

    X_train_flat = scaler.transform(
        X_train_flat
    )

    X_validation_flat = scaler.transform(
        X_validation_flat
    )

    X_test_flat = scaler.transform(
        X_test_flat
    )

    # ------------------------------------------------------------
    # Restore original shapes
    # ------------------------------------------------------------

    X_train = X_train_flat.reshape(
        X_train.shape
    ).astype(np.float32)

    X_validation = (
        X_validation_flat.reshape(
            X_validation.shape
        ).astype(np.float32)
    )

    X_test = X_test_flat.reshape(
        X_test.shape
    ).astype(np.float32)

    return (
        X_train,
        X_validation,
        X_test,
        scaler
    )


def main():

    print("=" * 80)
    print("PREPARING TRAINING DATA")
    print("=" * 80)

    # ------------------------------------------------------------
    # Required files
    # ------------------------------------------------------------

    required_files = [
        SEQUENCE_DIR / "X.npy",
        SEQUENCE_DIR / "y_forecast15.npy",
        SEQUENCE_DIR / "y_presence5.npy",
        SEQUENCE_DIR / "y_onset5.npy",
        SEQUENCE_DIR / "metadata.csv",
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n"
                f"{file_path}"
            )

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------

    X = np.load(
        SEQUENCE_DIR / "X.npy"
    )

    y_forecast15 = np.load(
        SEQUENCE_DIR / "y_forecast15.npy"
    )

    y_presence5 = np.load(
        SEQUENCE_DIR / "y_presence5.npy"
    )

    y_onset5 = np.load(
        SEQUENCE_DIR / "y_onset5.npy"
    )

    metadata = pd.read_csv(
        SEQUENCE_DIR / "metadata.csv",
        parse_dates=["Forecast_Start"]
    )

    print(
        f"\nX shape: "
        f"{X.shape}"
    )

    print(
        f"15-minute target: "
        f"{y_forecast15.shape}"
    )

    print(
        f"5-minute presence target: "
        f"{y_presence5.shape}"
    )

    print(
        f"5-minute onset target: "
        f"{y_onset5.shape}"
    )

    print(
        f"Metadata rows: "
        f"{len(metadata):,}"
    )

    # ------------------------------------------------------------
    # Validate lengths
    # ------------------------------------------------------------

    if len(X) != len(metadata):

        raise ValueError(
            "X and metadata have different lengths."
        )

    if len(y_forecast15) != len(X):

        raise ValueError(
            "15-minute target length does not match X."
        )

    if len(y_presence5) != len(X):

        raise ValueError(
            "5-minute presence target length does not match X."
        )

    if len(y_onset5) != len(X):

        raise ValueError(
            "5-minute onset target length does not match X."
        )

    # ------------------------------------------------------------
    # Verify target consistency with metadata
    # ------------------------------------------------------------

    metadata_forecast = (
        metadata[
            "Target_Forecast15"
        ].to_numpy()
    )

    if not np.array_equal(
        metadata_forecast,
        y_forecast15
    ):

        raise ValueError(
            "Target_Forecast15 in metadata "
            "does not match y_forecast15.npy."
        )

    print(
        "\nTarget consistency check: PASSED"
    )

    # ------------------------------------------------------------
    # Chronological split
    # ------------------------------------------------------------

    print(
        "\nCreating chronological splits..."
    )

    (
        train_indices,
        validation_indices,
        test_indices
    ) = chronological_split(
        metadata
    )

    # ------------------------------------------------------------
    # Check that splits do not overlap
    # ------------------------------------------------------------

    train_set = set(
        train_indices
    )

    validation_set = set(
        validation_indices
    )

    test_set = set(
        test_indices
    )

    if train_set & validation_set:

        raise ValueError(
            "Train and validation indices overlap."
        )

    if train_set & test_set:

        raise ValueError(
            "Train and test indices overlap."
        )

    if validation_set & test_set:

        raise ValueError(
            "Validation and test indices overlap."
        )

    print(
        "\nSplit overlap check: PASSED"
    )

    # ------------------------------------------------------------
    # Extract X
    # ------------------------------------------------------------

    X_train = X[
        train_indices
    ]

    X_validation = X[
        validation_indices
    ]

    X_test = X[
        test_indices
    ]

    # ------------------------------------------------------------
    # Main target
    #
    # 15-minute attack forecast
    # ------------------------------------------------------------

    y_train = y_forecast15[
        train_indices
    ]

    y_validation = y_forecast15[
        validation_indices
    ]

    y_test = y_forecast15[
        test_indices
    ]

    # ------------------------------------------------------------
    # Secondary targets
    # ------------------------------------------------------------

    presence5_train = y_presence5[
        train_indices
    ]

    presence5_validation = y_presence5[
        validation_indices
    ]

    presence5_test = y_presence5[
        test_indices
    ]

    onset5_train = y_onset5[
        train_indices
    ]

    onset5_validation = y_onset5[
        validation_indices
    ]

    onset5_test = y_onset5[
        test_indices
    ]

    # ------------------------------------------------------------
    # Scale
    # ------------------------------------------------------------

    print(
        "\nScaling features..."
    )

    (
        X_train,
        X_validation,
        X_test,
        scaler
    ) = scale_sequences(
        X_train,
        X_validation,
        X_test
    )

    # ------------------------------------------------------------
    # Create output directory
    # ------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # Save scaler
    # ------------------------------------------------------------

    joblib.dump(
        scaler,
        OUTPUT_DIR / "scaler.joblib"
    )

    # ------------------------------------------------------------
    # Save feature sequences
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Save main targets
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Save explicit 15-minute targets
    # ------------------------------------------------------------

    np.save(
        OUTPUT_DIR / "y_forecast15_train.npy",
        y_train
    )

    np.save(
        OUTPUT_DIR / "y_forecast15_validation.npy",
        y_validation
    )

    np.save(
        OUTPUT_DIR / "y_forecast15_test.npy",
        y_test
    )

    # ------------------------------------------------------------
    # Save 5-minute presence targets
    # ------------------------------------------------------------

    np.save(
        OUTPUT_DIR / "y_presence5_train.npy",
        presence5_train
    )

    np.save(
        OUTPUT_DIR / "y_presence5_validation.npy",
        presence5_validation
    )

    np.save(
        OUTPUT_DIR / "y_presence5_test.npy",
        presence5_test
    )

    # ------------------------------------------------------------
    # Save 5-minute onset targets
    # ------------------------------------------------------------

    np.save(
        OUTPUT_DIR / "y_onset5_train.npy",
        onset5_train
    )

    np.save(
        OUTPUT_DIR / "y_onset5_validation.npy",
        onset5_validation
    )

    np.save(
        OUTPUT_DIR / "y_onset5_test.npy",
        onset5_test
    )

    # ------------------------------------------------------------
    # Save metadata
    # ------------------------------------------------------------

    metadata.iloc[
        train_indices
    ].to_csv(
        OUTPUT_DIR / "train_metadata.csv",
        index=False
    )

    metadata.iloc[
        validation_indices
    ].to_csv(
        OUTPUT_DIR / "validation_metadata.csv",
        index=False
    )

    metadata.iloc[
        test_indices
    ].to_csv(
        OUTPUT_DIR / "test_metadata.csv",
        index=False
    )

    # ------------------------------------------------------------
    # Main target statistics
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("15-MINUTE FORECAST TARGET")
    print("=" * 80)

    print_distribution(
        "Training",
        y_train
    )

    print_distribution(
        "Validation",
        y_validation
    )

    print_distribution(
        "Test",
        y_test
    )

    # ------------------------------------------------------------
    # Secondary target statistics
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("5-MINUTE PRESENCE TARGET")
    print("=" * 80)

    print_distribution(
        "Training",
        presence5_train
    )

    print_distribution(
        "Validation",
        presence5_validation
    )

    print_distribution(
        "Test",
        presence5_test
    )

    print("\n" + "=" * 80)
    print("5-MINUTE ONSET TARGET")
    print("=" * 80)

    print_distribution(
        "Training",
        onset5_train
    )

    print_distribution(
        "Validation",
        onset5_validation
    )

    print_distribution(
        "Test",
        onset5_test
    )

    # ------------------------------------------------------------
    # Final shapes
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("TRAINING DATA PREPARATION COMPLETE")
    print("=" * 80)

    print("\nMain model shapes:")

    print(
        f"X_train: "
        f"{X_train.shape}"
    )

    print(
        f"X_validation: "
        f"{X_validation.shape}"
    )

    print(
        f"X_test: "
        f"{X_test.shape}"
    )

    print(
        f"y_train: "
        f"{y_train.shape}"
    )

    print(
        f"y_validation: "
        f"{y_validation.shape}"
    )

    print(
        f"y_test: "
        f"{y_test.shape}"
    )

    print("\nMain target:")

    print(
        "Attack occurrence "
        "within the next 15 minutes."
    )

    print("\nSaved to:")

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()