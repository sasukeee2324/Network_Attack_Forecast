from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data/processed/CIC-IDS2018")


def explore_file(file_path):
    print("\n" + "=" * 90)
    print(f"FILE: {file_path.name}")
    print("=" * 90)

    # Only load the columns we need for this analysis
    df = pd.read_csv(
        file_path,
        usecols=["Timestamp", "Label"],
        low_memory=False
    )

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce"
    )

    df = df.dropna(subset=["Timestamp", "Label"])

    print(f"\nRows: {len(df):,}")
    print(f"Time range: {df['Timestamp'].min()} → {df['Timestamp'].max()}")

    # ------------------------------------------------------------
    # Label distribution
    # ------------------------------------------------------------
    print("\nLabel distribution:")

    label_counts = df["Label"].value_counts()

    for label, count in label_counts.items():
        percentage = count / len(df) * 100
        print(f"  {label}: {count:,} ({percentage:.2f}%)")

    # ------------------------------------------------------------
    # Attack labels
    # ------------------------------------------------------------
    attack_df = df[df["Label"] != "Benign"].copy()

    print(f"\nTotal attack flows: {len(attack_df):,}")

    if len(attack_df) > 0:

        print("\nAttack types:")

        attack_counts = attack_df["Label"].value_counts()

        for label, count in attack_counts.items():
            percentage = count / len(attack_df) * 100
            print(f"  {label}: {count:,} ({percentage:.2f}%)")

        print("\nAttack time ranges:")

        for label in attack_counts.index:

            subset = attack_df[attack_df["Label"] == label]

            print(
                f"  {label}: "
                f"{subset['Timestamp'].min()} → "
                f"{subset['Timestamp'].max()}"
            )

    # ------------------------------------------------------------
    # Traffic per minute
    # ------------------------------------------------------------
    df["Minute"] = df["Timestamp"].dt.floor("min")

    traffic_per_minute = df.groupby("Minute").size()

    print("\nTraffic per minute:")

    print(f"  Mean:   {traffic_per_minute.mean():,.2f}")
    print(f"  Median: {traffic_per_minute.median():,.2f}")
    print(f"  Maximum:{traffic_per_minute.max():,.0f}")

    # ------------------------------------------------------------
    # Attack flows per minute
    # ------------------------------------------------------------
    attack_df["Minute"] = attack_df["Timestamp"].dt.floor("min")

    attack_per_minute = attack_df.groupby("Minute").size()

    if len(attack_per_minute) > 0:

        print("\nAttack flows per minute:")

        print(f"  Mean:   {attack_per_minute.mean():,.2f}")
        print(f"  Median: {attack_per_minute.median():,.2f}")
        print(f"  Maximum:{attack_per_minute.max():,.0f}")

        print("\nTop 10 attack-heavy minutes:")

        top_minutes = attack_per_minute.sort_values(
            ascending=False
        ).head(10)

        for timestamp, count in top_minutes.items():
            print(f"  {timestamp}: {count:,}")

    # ------------------------------------------------------------
    # Hourly attack distribution
    # ------------------------------------------------------------
    attack_df["Hour"] = attack_df["Timestamp"].dt.floor("h")

    hourly_attacks = attack_df.groupby(
        ["Hour", "Label"]
    ).size()

    print("\nAttack activity by hour:")

    print(hourly_attacks.head(30))


def main():

    csv_files = sorted(DATA_DIR.glob("*.csv"))

    if not csv_files:
        print(f"No processed CSV files found in {DATA_DIR}")
        return

    print(f"Found {len(csv_files)} processed files.")

    for file_path in csv_files:
        explore_file(file_path)

    print("\n" + "=" * 90)
    print("EXPLORATION COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()