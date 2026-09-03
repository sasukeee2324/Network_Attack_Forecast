from pathlib import Path
import pandas as pd
import numpy as np


INPUT_FILE = Path("data/processed/temporal/network_states.csv")
OUTPUT_DIR = Path("data/processed/sequences")

HISTORY_MINUTES = 10

# Main forecasting horizon
FORECAST_MINUTES = 15

# Secondary targets
PRESENCE_MINUTES = 5
ONSET_MINUTES = 5


# Only observable network measurements.
# NO attack labels or label-derived features.
FEATURE_COLUMNS = [
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


def is_consecutive(times):
    """Check that timestamps are exactly one minute apart."""

    if len(times) <= 1:
        return True

    differences = (
        times.diff()
        .dropna()
        .dt.total_seconds()
        / 60
    )

    return (differences == 1).all()


def create_sequences_for_day(day_df):
    """
    Create forecasting samples for one continuous capture day.

    Input:
        1-minute network states.

    Output:
        X:
            Previous 10 minutes of observable network behaviour.

        y_forecast15:
            Whether an attack occurs during the next 15 minutes.

        y_presence5:
            Whether an attack occurs during the next 5 minutes.

        y_onset5:
            Whether an attack starts during the next 5 minutes.
    """

    day_df = (
        day_df
        .sort_values("Minute")
        .reset_index(drop=True)
    )

    X = []
    y_forecast15 = []
    y_presence5 = []
    y_onset5 = []

    forecast_start_times = []
    source_files = []

    # We need:
    #
    # 10 minutes history
    # +
    # 15 minutes future
    #
    required_length = (
        HISTORY_MINUTES
        + FORECAST_MINUTES
    )

    if len(day_df) < required_length:
        return (
            X,
            y_forecast15,
            y_presence5,
            y_onset5,
            forecast_start_times,
            source_files,
        )

    for i in range(
        len(day_df) - required_length + 1
    ):

        # --------------------------------------------------------
        # History window
        # --------------------------------------------------------

        history = day_df.iloc[
            i:i + HISTORY_MINUTES
        ]

        # --------------------------------------------------------
        # Future window
        # --------------------------------------------------------

        future = day_df.iloc[
            i + HISTORY_MINUTES:
            i + HISTORY_MINUTES + FORECAST_MINUTES
        ]

        # --------------------------------------------------------
        # Verify history continuity
        # --------------------------------------------------------

        if not is_consecutive(
            history["Minute"]
        ):
            continue

        # --------------------------------------------------------
        # Verify future continuity
        # --------------------------------------------------------

        if not is_consecutive(
            future["Minute"]
        ):
            continue

        # --------------------------------------------------------
        # Verify history → future connection
        # --------------------------------------------------------

        connection = (
            future["Minute"].iloc[0]
            - history["Minute"].iloc[-1]
        ).total_seconds() / 60

        if connection != 1:
            continue

        # --------------------------------------------------------
        # Extract observable features
        # --------------------------------------------------------

        history_features = (
            history[FEATURE_COLUMNS]
            .to_numpy(dtype=np.float32)
        )

        # --------------------------------------------------------
        # Current attack state
        # --------------------------------------------------------

        current_state = int(
            history["Attack_State"].iloc[-1]
        )

        # --------------------------------------------------------
        # Future attack states
        # --------------------------------------------------------

        future_states = (
            future["Attack_State"]
            .to_numpy()
        )

        # --------------------------------------------------------
        # Target 1
        #
        # MAIN:
        #
        # Will an attack occur during
        # the next 15 minutes?
        # --------------------------------------------------------

        forecast15_target = int(
            future_states.max() > 0
        )

        # --------------------------------------------------------
        # Target 2
        #
        # SECONDARY:
        #
        # Will an attack occur during
        # the next 5 minutes?
        # --------------------------------------------------------

        presence5_states = future_states[
            :PRESENCE_MINUTES
        ]

        presence5_target = int(
            presence5_states.max() > 0
        )

        # --------------------------------------------------------
        # Target 3
        #
        # RESEARCH:
        #
        # Is the network currently benign,
        # but an attack begins during the
        # next 5 minutes?
        # --------------------------------------------------------

        onset5_target = int(
            current_state == 0
            and presence5_target == 1
        )

        # --------------------------------------------------------
        # Store sample
        # --------------------------------------------------------

        X.append(history_features)

        y_forecast15.append(
            forecast15_target
        )

        y_presence5.append(
            presence5_target
        )

        y_onset5.append(
            onset5_target
        )

        forecast_start_times.append(
            future["Minute"].iloc[0]
        )

        source_files.append(
            day_df["Source_File"].iloc[0]
        )

    return (
        X,
        y_forecast15,
        y_presence5,
        y_onset5,
        forecast_start_times,
        source_files,
    )


def print_distribution(
    name,
    targets,
    negative_label,
    positive_label
):
    """Print binary target distribution."""

    unique, counts = np.unique(
        targets,
        return_counts=True
    )

    print(f"\n{name}:")

    for value, count in zip(
        unique,
        counts
    ):

        percentage = (
            count / len(targets) * 100
        )

        label = (
            positive_label
            if value == 1
            else negative_label
        )

        print(
            f"  {label}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )


def main():

    print("=" * 80)
    print("CREATING TEMPORAL FORECASTING SEQUENCES")
    print("=" * 80)

    print(
        f"\nHistory window: "
        f"{HISTORY_MINUTES} minutes"
    )

    print(
        f"Main forecast horizon: "
        f"{FORECAST_MINUTES} minutes"
    )

    if not INPUT_FILE.exists():

        print(
            f"\nInput file not found:"
        )

        print(INPUT_FILE)

        return

    # ------------------------------------------------------------
    # Load temporal dataset
    # ------------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Minute"],
        low_memory=False
    )

    print(
        f"\nTotal network states: "
        f"{len(df):,}"
    )

    # ------------------------------------------------------------
    # Verify columns
    # ------------------------------------------------------------

    required_columns = (
        FEATURE_COLUMNS
        + [
            "Minute",
            "Attack_State",
            "Attack_Type",
            "Source_File",
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        print("\nMissing columns:")

        for column in missing_columns:
            print(f"  {column}")

        return

    # ------------------------------------------------------------
    # Process each capture day independently
    # ------------------------------------------------------------

    all_X = []

    all_y_forecast15 = []
    all_y_presence5 = []
    all_y_onset5 = []

    all_forecast_times = []
    all_sources = []

    for source_file, day_df in df.groupby(
        "Source_File",
        sort=False
    ):

        print("\n" + "-" * 80)

        print(
            f"Source: {source_file}"
        )

        print(
            f"States: {len(day_df):,}"
        )

        result = create_sequences_for_day(
            day_df
        )

        (
            X,
            y_forecast15,
            y_presence5,
            y_onset5,
            forecast_times,
            sources,
        ) = result

        all_X.extend(X)

        all_y_forecast15.extend(
            y_forecast15
        )

        all_y_presence5.extend(
            y_presence5
        )

        all_y_onset5.extend(
            y_onset5
        )

        all_forecast_times.extend(
            forecast_times
        )

        all_sources.extend(
            sources
        )

        print(
            f"Sequences created: "
            f"{len(y_forecast15):,}"
        )

    if not all_X:

        print(
            "\nNo sequences were created."
        )

        return

    # ------------------------------------------------------------
    # Convert to NumPy
    # ------------------------------------------------------------

    X = np.asarray(
        all_X,
        dtype=np.float32
    )

    y_forecast15 = np.asarray(
        all_y_forecast15,
        dtype=np.int64
    )

    y_presence5 = np.asarray(
        all_y_presence5,
        dtype=np.int64
    )

    y_onset5 = np.asarray(
        all_y_onset5,
        dtype=np.int64
    )

    # ------------------------------------------------------------
    # Create output directory
    # ------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # Save feature sequences
    # ------------------------------------------------------------

    np.save(
        OUTPUT_DIR / "X.npy",
        X
    )

    # ------------------------------------------------------------
    # Save main target
    # ------------------------------------------------------------

    np.save(
        OUTPUT_DIR / "y_forecast15.npy",
        y_forecast15
    )

    # Keep y.npy pointing to the main target.
    np.save(
        OUTPUT_DIR / "y.npy",
        y_forecast15
    )

    # ------------------------------------------------------------
    # Save secondary targets
    # ------------------------------------------------------------

    np.save(
        OUTPUT_DIR / "y_presence5.npy",
        y_presence5
    )

    np.save(
        OUTPUT_DIR / "y_onset5.npy",
        y_onset5
    )

    # ------------------------------------------------------------
    # Save metadata
    # ------------------------------------------------------------

    metadata = pd.DataFrame({
        "Forecast_Start": all_forecast_times,
        "Source_File": all_sources,
        "Target_Forecast15": y_forecast15,
        "Target_Presence5": y_presence5,
        "Target_Onset5": y_onset5,
    })

    metadata.to_csv(
        OUTPUT_DIR / "metadata.csv",
        index=False
    )

    # ------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("SEQUENCE DATASET CREATED")
    print("=" * 80)

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

    print(
        "\nExpected X structure:"
    )

    print(
        "(samples, history_minutes, features)"
    )

    # ------------------------------------------------------------
    # Distributions
    # ------------------------------------------------------------

    print_distribution(
        "15-MINUTE FORECAST TARGET",
        y_forecast15,
        "No attack in next 15 min",
        "Attack in next 15 min"
    )

    print_distribution(
        "5-MINUTE PRESENCE TARGET",
        y_presence5,
        "No attack in next 5 min",
        "Attack in next 5 min"
    )

    print_distribution(
        "5-MINUTE ONSET TARGET",
        y_onset5,
        "No attack onset",
        "Attack onset"
    )

    # ------------------------------------------------------------
    # Files
    # ------------------------------------------------------------

    print("\nFiles saved:")

    print(
        OUTPUT_DIR / "X.npy"
    )

    print(
        OUTPUT_DIR / "y.npy"
    )

    print(
        OUTPUT_DIR / "y_forecast15.npy"
    )

    print(
        OUTPUT_DIR / "y_presence5.npy"
    )

    print(
        OUTPUT_DIR / "y_onset5.npy"
    )

    print(
        OUTPUT_DIR / "metadata.csv"
    )


if __name__ == "__main__":
    main()