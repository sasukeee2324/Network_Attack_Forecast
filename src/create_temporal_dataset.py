from pathlib import Path
import pandas as pd
import numpy as np


INPUT_DIR = Path("data/processed/CIC-IDS2018")
OUTPUT_DIR = Path("data/processed/temporal")


def process_file(file_path):
    print("\n" + "=" * 80)
    print(f"Processing: {file_path.name}")
    print("=" * 80)

    # Load only the columns needed for our first temporal model
    columns = [
        "Timestamp",
        "Dst Port",
        "Protocol",
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
    ]

    df = pd.read_csv(
        file_path,
        usecols=columns,
        low_memory=False
    )

    print(f"Flows loaded: {len(df):,}")

    # ------------------------------------------------------------
    # Timestamp
    # ------------------------------------------------------------
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce"
    )

    df = df.dropna(subset=["Timestamp"])

    # ------------------------------------------------------------
    # Numeric conversion
    # ------------------------------------------------------------
    numeric_columns = [
        column for column in columns
        if column not in ["Timestamp", "Label"]
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Replace infinity
    df[numeric_columns] = df[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # ------------------------------------------------------------
    # Create 1-minute time bucket
    # ------------------------------------------------------------
    df["Minute"] = df["Timestamp"].dt.floor("min")

    # ------------------------------------------------------------
    # Attack indicator
    # ------------------------------------------------------------
    df["Is_Attack"] = (
        df["Label"] != "Benign"
    ).astype(int)

    # ------------------------------------------------------------
    # Aggregate network behaviour per minute
    # ------------------------------------------------------------
    states = df.groupby("Minute").agg(
        Flow_Count=("Label", "size"),

        Attack_Flow_Count=("Is_Attack", "sum"),

        Avg_Flow_Duration=("Flow Duration", "mean"),

        Avg_Fwd_Packets=("Tot Fwd Pkts", "mean"),
        Avg_Bwd_Packets=("Tot Bwd Pkts", "mean"),

        Avg_Fwd_Bytes=("TotLen Fwd Pkts", "mean"),
        Avg_Bwd_Bytes=("TotLen Bwd Pkts", "mean"),

        Avg_Bytes_Per_Second=("Flow Byts/s", "mean"),
        Avg_Packets_Per_Second=("Flow Pkts/s", "mean"),

        Avg_Packet_Length=("Pkt Len Mean", "mean"),
        Avg_Packet_Length_Std=("Pkt Len Std", "mean"),

        Avg_Fwd_Packets_Per_Second=("Fwd Pkts/s", "mean"),
        Avg_Bwd_Packets_Per_Second=("Bwd Pkts/s", "mean"),

        SYN_Count=("SYN Flag Cnt", "sum"),
        RST_Count=("RST Flag Cnt", "sum"),
        PSH_Count=("PSH Flag Cnt", "sum"),
        ACK_Count=("ACK Flag Cnt", "sum"),

        Unique_Destination_Ports=("Dst Port", "nunique"),
        Unique_Protocols=("Protocol", "nunique"),
    )

    # ------------------------------------------------------------
    # Attack ratio
    # ------------------------------------------------------------
    states["Attack_Ratio"] = (
        states["Attack_Flow_Count"]
        / states["Flow_Count"]
    )

    # ------------------------------------------------------------
    # Determine dominant attack label in each minute
    # ------------------------------------------------------------
    attack_labels = (
        df[df["Label"] != "Benign"]
        .groupby("Minute")["Label"]
        .agg(lambda x: x.value_counts().index[0])
    )

    states["Attack_Type"] = (
        attack_labels
        .reindex(states.index)
        .fillna("Benign")
    )

    # ------------------------------------------------------------
    # Binary state
    # ------------------------------------------------------------
    states["Attack_State"] = (
        states["Attack_Flow_Count"] > 0
    ).astype(int)

    # ------------------------------------------------------------
    # Clean numerical NaN values
    # ------------------------------------------------------------
    numeric_state_columns = states.select_dtypes(
        include=np.number
    ).columns

    states[numeric_state_columns] = (
        states[numeric_state_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # ------------------------------------------------------------
    # Restore timestamp as a column
    # ------------------------------------------------------------
    states = states.reset_index()

    # ------------------------------------------------------------
    # Add source day
    # ------------------------------------------------------------
    states["Source_File"] = file_path.name

    print(f"Network states created: {len(states):,}")

    return states


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    files = sorted(INPUT_DIR.glob("*.csv"))

    if not files:
        print("No processed CSV files found.")
        return

    all_states = []

    for file_path in files:
        states = process_file(file_path)
        all_states.append(states)

    # ------------------------------------------------------------
    # Combine all days
    # ------------------------------------------------------------
    temporal_df = pd.concat(
        all_states,
        ignore_index=True
    )

    # Sort chronologically
    temporal_df = temporal_df.sort_values(
        "Minute"
    ).reset_index(drop=True)

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    output_path = OUTPUT_DIR / "network_states.csv"

    temporal_df.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 80)
    print("TEMPORAL DATASET CREATED")
    print("=" * 80)

    print(f"Total network states: {len(temporal_df):,}")

    print(
        f"Time range: "
        f"{temporal_df['Minute'].min()} → "
        f"{temporal_df['Minute'].max()}"
    )

    print("\nAttack state distribution:")

    print(
        temporal_df["Attack_State"]
        .value_counts()
    )

    print("\nAttack type distribution:")

    print(
        temporal_df["Attack_Type"]
        .value_counts()
    )

    print("\nSaved to:")

    print(output_path)


if __name__ == "__main__":
    main()