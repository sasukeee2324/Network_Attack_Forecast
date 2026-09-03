from pathlib import Path
import pandas as pd
import numpy as np


RAW_DIR = Path("data/raw/CIC-IDS2018")
PROCESSED_DIR = Path("data/processed/CIC-IDS2018")


def clean_file(file_path):
    print("\n" + "=" * 80)
    print(f"Processing: {file_path.name}")
    print("=" * 80)

    # Read the CSV
    df = pd.read_csv(file_path, low_memory=False)

    original_rows = len(df)
    print(f"Original rows: {original_rows:,}")

    # ------------------------------------------------------------
    # 1. Remove accidental embedded header rows
    # ------------------------------------------------------------
    # Some CIC-IDS2018 files contain rows where Label == "Label".
    bad_label_rows = df["Label"].astype(str).str.strip().eq("Label")

    print(f"Embedded header rows removed: {bad_label_rows.sum():,}")

    df = df.loc[~bad_label_rows].copy()

    # ------------------------------------------------------------
    # 2. Clean label column
    # ------------------------------------------------------------
    df["Label"] = df["Label"].astype(str).str.strip()

    # Remove empty/missing labels
    invalid_labels = (
        df["Label"].isna()
        | df["Label"].eq("")
        | df["Label"].eq("nan")
    )

    print(f"Invalid label rows removed: {invalid_labels.sum():,}")

    df = df.loc[~invalid_labels].copy()

    # ------------------------------------------------------------
    # 3. Parse timestamps
    # ------------------------------------------------------------
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        dayfirst=True,
        errors="coerce"
    )

    invalid_timestamps = df["Timestamp"].isna()

    print(f"Invalid timestamp rows: {invalid_timestamps.sum():,}")

    df = df.loc[~invalid_timestamps].copy()

    # ------------------------------------------------------------
    # 4. Remove obviously corrupted timestamps
    # ------------------------------------------------------------
    # CIC-IDS2018 traffic is from 2018.
    valid_timestamp = (
        (df["Timestamp"] >= "2018-01-01")
        & (df["Timestamp"] < "2019-01-01")
    )

    corrupted_timestamps = (~valid_timestamp).sum()

    print(f"Out-of-range timestamp rows removed: {corrupted_timestamps:,}")

    df = df.loc[valid_timestamp].copy()

    # ------------------------------------------------------------
    # 5. Convert numeric columns
    # ------------------------------------------------------------
    numeric_columns = [
        column
        for column in df.columns
        if column not in ["Timestamp", "Label"]
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # ------------------------------------------------------------
    # 6. Replace infinite values
    # ------------------------------------------------------------
    numeric_df = df[numeric_columns]

    positive_inf = np.isposinf(numeric_df).sum().sum()
    negative_inf = np.isneginf(numeric_df).sum().sum()

    print(f"Positive infinity values replaced: {positive_inf:,}")
    print(f"Negative infinity values replaced: {negative_inf:,}")

    df[numeric_columns] = numeric_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # ------------------------------------------------------------
    # 7. Remove duplicate flows
    # ------------------------------------------------------------
    duplicates = df.duplicated().sum()

    print(f"Duplicate rows removed: {duplicates:,}")

    df = df.drop_duplicates().copy()

    # ------------------------------------------------------------
    # 8. Sort chronologically
    # ------------------------------------------------------------
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # ------------------------------------------------------------
    # 9. Save cleaned file
    # ------------------------------------------------------------
    output_path = PROCESSED_DIR / file_path.name

    df.to_csv(output_path, index=False)

    # ------------------------------------------------------------
    # 10. Final statistics
    # ------------------------------------------------------------
    print(f"\nFinal rows: {len(df):,}")
    print(f"Rows removed: {original_rows - len(df):,}")
    print(f"Final columns: {len(df.columns)}")

    print("\nLabels:")
    print(df["Label"].value_counts())

    print(f"\nSaved to:")
    print(output_path)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(RAW_DIR.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {RAW_DIR}")
        return

    print(f"Found {len(csv_files)} files.")

    for file_path in csv_files:
        clean_file(file_path)

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()