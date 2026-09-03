from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data/raw/CIC-IDS2018")


def analyze_file(file_path):
    print("\n" + "=" * 80)
    print(f"FILE: {file_path.name}")
    print("=" * 80)

    # Read one file at a time
    df = pd.read_csv(file_path)

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # ------------------------------------------------------------------
    # Missing values
    # ------------------------------------------------------------------
    missing = df.isna().sum()
    missing_total = missing.sum()

    print(f"\nTotal missing values: {missing_total:,}")

    if missing_total > 0:
        print("\nColumns with missing values:")
        print(missing[missing > 0].sort_values(ascending=False))

    # ------------------------------------------------------------------
    # Infinite values
    # ------------------------------------------------------------------
    numeric_df = df.select_dtypes(include=np.number)

    positive_inf = np.isposinf(numeric_df).sum().sum()
    negative_inf = np.isneginf(numeric_df).sum().sum()
    total_inf = positive_inf + negative_inf

    print(f"\nPositive infinity values: {positive_inf:,}")
    print(f"Negative infinity values: {negative_inf:,}")
    print(f"Total infinity values: {total_inf:,}")

    # ------------------------------------------------------------------
    # Duplicate rows
    # ------------------------------------------------------------------
    duplicates = df.duplicated().sum()

    print(f"\nDuplicate rows: {duplicates:,}")

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------
    print("\nLabel distribution:")

    label_counts = df["Label"].value_counts(dropna=False)

    for label, count in label_counts.items():
        percentage = count / len(df) * 100
        print(f"  {label}: {count:,} ({percentage:.2f}%)")

    # ------------------------------------------------------------------
    # Timestamp
    # ------------------------------------------------------------------
    timestamps = pd.to_datetime(
        df["Timestamp"],
        dayfirst=True,
        errors="coerce"
    )

    print("\nTimestamp information:")

    print(f"  Invalid timestamps: {timestamps.isna().sum():,}")

    if timestamps.notna().any():
        print(f"  Start: {timestamps.min()}")
        print(f"  End:   {timestamps.max()}")

    # ------------------------------------------------------------------
    # Numeric feature summary
    # ------------------------------------------------------------------
    print("\nNumeric feature summary:")

    print(f"  Numeric columns: {len(numeric_df.columns)}")

    zero_variance = numeric_df.nunique()
    zero_variance_columns = zero_variance[zero_variance <= 1]

    print(f"  Constant columns: {len(zero_variance_columns)}")

    if len(zero_variance_columns) > 0:
        print("  Constant column names:")
        print(f"    {list(zero_variance_columns.index)}")


def main():
    csv_files = sorted(DATA_DIR.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in: {DATA_DIR}")
        return

    print(f"Found {len(csv_files)} CSV files.")

    for file_path in csv_files:
        analyze_file(file_path)

    print("\n" + "=" * 80)
    print("DATASET ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()