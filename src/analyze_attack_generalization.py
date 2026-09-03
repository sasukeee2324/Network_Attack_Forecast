from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/processed/CIC-IDS2018")


def main():

    print("=" * 80)
    print("ATTACK GENERALIZATION ANALYSIS")
    print("=" * 80)

    files = sorted(
        DATA_DIR.glob("*.csv")
    )

    for file_path in files:

        print("\n" + "-" * 80)
        print(f"FILE: {file_path.name}")
        print("-" * 80)

        df = pd.read_csv(
            file_path,
            usecols=[
                "Timestamp",
                "Label"
            ],
            low_memory=False
        )

        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["Timestamp", "Label"]
        )

        print(
            f"\nRows: {len(df):,}"
        )

        print("\nLabels:")

        print(
            df["Label"]
            .value_counts()
            .to_string()
        )

        print("\nAttack labels:")

        attacks = df[
            df["Label"].str.lower()
            != "benign"
        ]

        if len(attacks) == 0:

            print("No attacks found.")

            continue

        print(
            attacks["Label"]
            .value_counts()
            .to_string()
        )

        print("\nAttack time ranges:")

        for label, group in attacks.groupby(
            "Label"
        ):

            print(
                f"  {label}: "
                f"{group['Timestamp'].min()} "
                f"→ "
                f"{group['Timestamp'].max()}"
            )


if __name__ == "__main__":
    main()