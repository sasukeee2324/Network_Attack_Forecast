from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/processed/CIC-IDS2018")
OUTPUT_DIR = Path("data/processed/unseen_attack")

UNSEEN_ATTACK = "Infilteration"

HISTORY_MINUTES = 10
FORECAST_MINUTES = 15


# ============================================================
# FEATURES
# ============================================================

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


# ============================================================
# BUILD NETWORK STATES
# ============================================================

def create_network_states(df, source_file):

    df = df.copy()

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Timestamp"]
    )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    numeric_columns = [
        "Flow Duration",
        "Tot Fwd Pkts",
        "Tot Bwd Pkts",
        "TotLen Fwd Pkts",
        "TotLen Bwd Pkts",
        "Flow Byts/s",
        "Flow Pkts/s",
        "Pkt Len Mean",
        "Pkt Len Std",
        "Fwd Pkts/s",
        "Bwd Pkts/s",
        "SYN Flag Cnt",
        "RST Flag Cnt",
        "PSH Flag Cnt",
        "ACK Flag Cnt",
        "Dst Port",
        "Protocol",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # One-minute buckets
    # --------------------------------------------------------

    df["Minute"] = (
        df["Timestamp"]
        .dt.floor("min")
    )

    states = []

    for minute, group in df.groupby(
        "Minute",
        sort=True
    ):

        state = {

            "Timestamp": minute,

            "Source_File": source_file,

            "Flow_Count":
                len(group),

            "Avg_Flow_Duration":
                group["Flow Duration"].mean(),

            "Avg_Fwd_Packets":
                group["Tot Fwd Pkts"].mean(),

            "Avg_Bwd_Packets":
                group["Tot Bwd Pkts"].mean(),

            "Avg_Fwd_Bytes":
                group["TotLen Fwd Pkts"].mean(),

            "Avg_Bwd_Bytes":
                group["TotLen Bwd Pkts"].mean(),

            "Avg_Bytes_Per_Second":
                group["Flow Byts/s"].mean(),

            "Avg_Packets_Per_Second":
                group["Flow Pkts/s"].mean(),

            "Avg_Packet_Length":
                group["Pkt Len Mean"].mean(),

            "Avg_Packet_Length_Std":
                group["Pkt Len Std"].mean(),

            "Avg_Fwd_Packets_Per_Second":
                group["Fwd Pkts/s"].mean(),

            "Avg_Bwd_Packets_Per_Second":
                group["Bwd Pkts/s"].mean(),

            "SYN_Count":
                group["SYN Flag Cnt"].sum(),

            "RST_Count":
                group["RST Flag Cnt"].sum(),

            "PSH_Count":
                group["PSH Flag Cnt"].sum(),

            "ACK_Count":
                group["ACK Flag Cnt"].sum(),

            "Unique_Destination_Ports":
                group["Dst Port"].nunique(),

            "Unique_Protocols":
                group["Protocol"].nunique(),
        }

        # ----------------------------------------------------
        # Attack state
        # ----------------------------------------------------

        attack_labels = group.loc[
            group["Label"].str.lower() != "benign",
            "Label"
        ]

        if attack_labels.empty:

            state["Attack_State"] = 0
            state["Attack_Type"] = "Benign"

        else:

            state["Attack_State"] = 1

            state["Attack_Type"] = (
                attack_labels
                .value_counts()
                .index[0]
            )

        states.append(state)

    states_df = pd.DataFrame(states)

    # --------------------------------------------------------
    # Clean features
    # --------------------------------------------------------

    states_df[FEATURE_COLUMNS] = (
        states_df[FEATURE_COLUMNS]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    states_df = states_df.sort_values(
        "Timestamp"
    ).reset_index(
        drop=True
    )

    return states_df


# ============================================================
# CREATE SEQUENCES
# ============================================================

def create_sequences(states):

    X = []
    y = []
    metadata = []

    states = states.sort_values(
        "Timestamp"
    ).reset_index(
        drop=True
    )

    total_window = (
        HISTORY_MINUTES
        + FORECAST_MINUTES
    )

    # --------------------------------------------------------
    # Need at least 25 states
    # --------------------------------------------------------

    if len(states) < total_window:

        return (
            np.empty(
                (
                    0,
                    HISTORY_MINUTES,
                    len(FEATURE_COLUMNS)
                ),
                dtype=np.float32
            ),
            np.empty(
                (0,),
                dtype=np.int64
            ),
            pd.DataFrame()
        )

    # --------------------------------------------------------
    # Sliding window
    # --------------------------------------------------------

    for start in range(
        0,
        len(states) - total_window + 1
    ):

        history_start = start

        history_end = (
            start + HISTORY_MINUTES
        )

        forecast_end = (
            start
            + HISTORY_MINUTES
            + FORECAST_MINUTES
        )

        history = states.iloc[
            history_start:history_end
        ]

        future = states.iloc[
            history_end:forecast_end
        ]

        # ----------------------------------------------------
        # Time continuity
        #
        # Only accept windows where the entire window spans
        # exactly the expected number of minutes.
        # ----------------------------------------------------

        window_start = (
            states.iloc[start]["Timestamp"]
        )

        window_end = (
            states.iloc[forecast_end - 1]["Timestamp"]
        )

        expected_end = (
            window_start
            + pd.Timedelta(
                minutes=total_window - 1
            )
        )

        if window_end != expected_end:

            continue

        # ----------------------------------------------------
        # Input
        # ----------------------------------------------------

        sequence = history[
            FEATURE_COLUMNS
        ].to_numpy(
            dtype=np.float32
        )

        # ----------------------------------------------------
        # 15-minute target
        # ----------------------------------------------------

        attack_present = (
            future["Attack_State"] == 1
        ).any()

        target = int(
            attack_present
        )

        future_attack_types = sorted(
            future.loc[
                future["Attack_State"] == 1,
                "Attack_Type"
            ].unique()
        )

        X.append(
            sequence
        )

        y.append(
            target
        )

        metadata.append({

            "Forecast_Start":
                future["Timestamp"].iloc[0],

            "Forecast_End":
                future["Timestamp"].iloc[-1],

            "Target_Forecast15":
                target,

            "Future_Attack_Types":
                ",".join(
                    future_attack_types
                ),

            "Source_File":
                states["Source_File"].iloc[0],
        })

    return (
        np.asarray(
            X,
            dtype=np.float32
        ),
        np.asarray(
            y,
            dtype=np.int64
        ),
        pd.DataFrame(
            metadata
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("UNSEEN ATTACK DATASET CREATION")
    print("=" * 80)

    print(
        f"\nUnseen attack: {UNSEEN_ATTACK}"
    )

    print(
        f"History window: "
        f"{HISTORY_MINUTES} minutes"
    )

    print(
        f"Forecast horizon: "
        f"{FORECAST_MINUTES} minutes"
    )

    files = sorted(
        DATA_DIR.glob("*.csv")
    )

    if not files:

        raise FileNotFoundError(
            f"No CSV files found in {DATA_DIR}"
        )

    training_X = []
    training_y = []
    training_metadata = []

    unseen_X = []
    unseen_y = []
    unseen_metadata = []

    # ========================================================
    # PROCESS EACH SOURCE
    # ========================================================

    for file_path in files:

        print("\n" + "-" * 80)

        print(
            f"Processing: {file_path.name}"
        )

        df = pd.read_csv(
            file_path,
            usecols=[
                "Dst Port",
                "Protocol",
                "Timestamp",
                "Flow Duration",
                "Tot Fwd Pkts",
                "Tot Bwd Pkts",
                "TotLen Fwd Pkts",
                "TotLen Bwd Pkts",
                "Flow Byts/s",
                "Flow Pkts/s",
                "Pkt Len Mean",
                "Pkt Len Std",
                "Fwd Pkts/s",
                "Bwd Pkts/s",
                "SYN Flag Cnt",
                "RST Flag Cnt",
                "PSH Flag Cnt",
                "ACK Flag Cnt",
                "Label",
            ],
            low_memory=False
        )

        print(
            f"Flows loaded: {len(df):,}"
        )

        states = create_network_states(
            df,
            file_path.name
        )

        print(
            f"Network states: {len(states):,}"
        )

        print("\nAttack states:")

        attack_states = states[
            states["Attack_State"] == 1
        ]

        if attack_states.empty:

            print("  None")

        else:

            print(
                attack_states[
                    "Attack_Type"
                ]
                .value_counts()
                .to_string()
            )

        # ----------------------------------------------------
        # Create sequences
        # ----------------------------------------------------

        X, y, metadata = create_sequences(
            states
        )

        print(
            f"\nSequences created: {len(X)}"
        )

        if len(X) == 0:

            print(
                "WARNING: No valid temporal sequences."
            )

            continue

        # ----------------------------------------------------
        # Identify unseen attack source
        # ----------------------------------------------------

        contains_unseen = (
            states["Attack_Type"]
            .eq(UNSEEN_ATTACK)
            .any()
        )

        if contains_unseen:

            print(
                "\n→ RESERVED AS UNSEEN TEST"
            )

            unseen_X.append(X)
            unseen_y.append(y)
            unseen_metadata.append(
                metadata
            )

        else:

            print(
                "\n→ USED FOR TRAINING"
            )

            training_X.append(X)
            training_y.append(y)
            training_metadata.append(
                metadata
            )

    # ========================================================
    # CHECK
    # ========================================================

    if not training_X:

        raise RuntimeError(
            "No training sequences were created."
        )

    if not unseen_X:

        raise RuntimeError(
            f"No unseen '{UNSEEN_ATTACK}' sequences were created."
        )

    # ========================================================
    # COMBINE
    # ========================================================

    X_train = np.concatenate(
        training_X,
        axis=0
    )

    y_train = np.concatenate(
        training_y,
        axis=0
    )

    metadata_train = pd.concat(
        training_metadata,
        ignore_index=True
    )

    X_unseen = np.concatenate(
        unseen_X,
        axis=0
    )

    y_unseen = np.concatenate(
        unseen_y,
        axis=0
    )

    metadata_unseen = pd.concat(
        unseen_metadata,
        ignore_index=True
    )

    # ========================================================
    # SAFETY CHECK
    # ========================================================

    training_future_types = set()

    for value in metadata_train[
        "Future_Attack_Types"
    ].dropna():

        if not value:
            continue

        for attack in value.split(","):

            if attack:
                training_future_types.add(
                    attack
                )

    if UNSEEN_ATTACK in training_future_types:

        raise RuntimeError(
            "DATA LEAKAGE: unseen attack appears "
            "in training targets."
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 80)
    print("GENERALIZATION DATASET SUMMARY")
    print("=" * 80)

    print(
        "\nTraining sequences:"
    )

    print(
        len(X_train)
    )

    print(
        "\nTraining shape:"
    )

    print(
        X_train.shape
    )

    print(
        "\nTraining target distribution:"
    )

    print(
        pd.Series(
            y_train
        )
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nTraining future attack types:"
    )

    if training_future_types:

        for attack in sorted(
            training_future_types
        ):

            print(
                f"  {attack}"
            )

    else:

        print(
            "  None"
        )

    print(
        "\nUnseen test sequences:"
    )

    print(
        len(X_unseen)
    )

    print(
        "\nUnseen shape:"
    )

    print(
        X_unseen.shape
    )

    print(
        "\nUnseen target distribution:"
    )

    print(
        pd.Series(
            y_unseen
        )
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nUnseen attack:"
    )

    print(
        f"  {UNSEEN_ATTACK}"
    )

    print(
        "\nUnseen future attack types:"
    )

    print(
        metadata_unseen[
            "Future_Attack_Types"
        ]
        .value_counts()
        .to_string()
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
        OUTPUT_DIR / "y_train.npy",
        y_train
    )

    np.save(
        OUTPUT_DIR / "X_unseen.npy",
        X_unseen
    )

    np.save(
        OUTPUT_DIR / "y_unseen.npy",
        y_unseen
    )

    metadata_train.to_csv(
        OUTPUT_DIR / "metadata_train.csv",
        index=False
    )

    metadata_unseen.to_csv(
        OUTPUT_DIR / "metadata_unseen.csv",
        index=False
    )

    print("\n" + "=" * 80)
    print("UNSEEN ATTACK DATASET CREATED")
    print("=" * 80)

    print(
        "\nSaved to:"
    )

    print(
        OUTPUT_DIR
    )

    print("\nFiles:")

    print(
        "  X_train.npy"
    )

    print(
        "  y_train.npy"
    )

    print(
        "  X_unseen.npy"
    )

    print(
        "  y_unseen.npy"
    )

    print(
        "  metadata_train.csv"
    )

    print(
        "  metadata_unseen.csv"
    )


if __name__ == "__main__":
    main()