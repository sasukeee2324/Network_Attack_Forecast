from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/processed/temporal/network_states.csv"
)

OUTPUT_FILE = Path(
    "data/processed/temporal/network_states_enhanced.csv"
)


# ============================================================
# BASE TRAFFIC FEATURES
# ============================================================
#
# These are legitimate traffic-behaviour measurements.
#
# IMPORTANT:
# Attack_Flow_Count is intentionally NOT included.
# Attack_Ratio is intentionally NOT included.
#
# Both are derived from attack labels and would leak target
# information into the forecasting model.
# ============================================================

BASE_FEATURES = [
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
# FEATURES FOR TEMPORAL ENGINEERING
# ============================================================

TREND_FEATURES = [
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
]


# ============================================================
# SAFE PERCENT CHANGE
# ============================================================

def safe_pct_change(series, periods=1):

    previous = series.shift(periods)

    denominator = (
        previous.abs()
        .clip(lower=1e-6)
    )

    result = (
        (series - previous)
        / denominator
    )

    return result.replace(
        [np.inf, -np.inf],
        np.nan
    )


# ============================================================
# ROLLING Z-SCORE
# ============================================================

def rolling_zscore(
    series,
    window=5
):

    rolling_mean = (
        series
        .rolling(
            window=window,
            min_periods=2
        )
        .mean()
    )

    rolling_std = (
        series
        .rolling(
            window=window,
            min_periods=2
        )
        .std()
    )

    result = (
        (series - rolling_mean)
        / rolling_std.replace(
            0,
            np.nan
        )
    )

    return result.replace(
        [np.inf, -np.inf],
        np.nan
    )


# ============================================================
# ROLLING SLOPE
# ============================================================

def rolling_slope(
    series,
    window=5
):

    values = series.to_numpy(
        dtype=float
    )

    result = np.full(
        len(values),
        np.nan,
        dtype=float
    )

    if len(values) < window:

        return pd.Series(
            result,
            index=series.index
        )

    x = np.arange(
        window,
        dtype=float
    )

    x_mean = x.mean()

    denominator = np.sum(
        (x - x_mean) ** 2
    )

    for i in range(
        window - 1,
        len(values)
    ):

        y = values[
            i - window + 1:
            i + 1
        ]

        if not np.all(
            np.isfinite(y)
        ):
            continue

        y_mean = y.mean()

        numerator = np.sum(
            (x - x_mean)
            * (y - y_mean)
        )

        result[i] = (
            numerator
            / denominator
        )

    return pd.Series(
        result,
        index=series.index
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("TEMPORAL FEATURE ENGINEERING")
    print("=" * 80)

    # ========================================================
    # LOAD
    # ========================================================

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nInput file not found:\n"
            f"{INPUT_FILE}\n"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"\nInput rows: {len(df):,}"
    )

    print(
        f"Input columns: {len(df.columns)}"
    )

    # ========================================================
    # VALIDATE REQUIRED COLUMNS
    # ========================================================

    required_columns = (
        BASE_FEATURES
        + [
            "Minute",
            "Source_File",
            "Attack_Flow_Count",
            "Attack_Ratio",
            "Attack_Type",
            "Attack_State",
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "\nMissing required columns:\n"
            + "\n".join(
                missing_columns
            )
        )

    # ========================================================
    # CONVERT MINUTE
    # ========================================================

    df["Minute"] = pd.to_datetime(
        df["Minute"],
        errors="coerce"
    )

    invalid_minutes = (
        df["Minute"].isna().sum()
    )

    if invalid_minutes > 0:

        raise ValueError(
            f"\nFound {invalid_minutes} "
            f"invalid Minute values."
        )

    # ========================================================
    # CONVERT BASE FEATURES TO NUMERIC
    # ========================================================

    for feature in BASE_FEATURES:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    df[BASE_FEATURES] = (
        df[BASE_FEATURES]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    # ========================================================
    # SORT CHRONOLOGICALLY
    # ========================================================

    df = df.sort_values(
        [
            "Source_File",
            "Minute",
        ]
    ).reset_index(
        drop=True
    )

    # ========================================================
    # STORE GENERATED FEATURES
    # ========================================================
    #
    # We build these separately instead of repeatedly adding
    # columns to the original dataframe.
    #
    # This prevents pandas DataFrame fragmentation warnings.
    # ========================================================

    generated_groups = []

    # ========================================================
    # PROCESS EACH SOURCE SEPARATELY
    # ========================================================

    print(
        "\nGenerating temporal features..."
    )

    for source, group in df.groupby(
        "Source_File",
        sort=False
    ):

        group = group.sort_values(
            "Minute"
        ).reset_index(
            drop=True
        )

        print(
            "\n" + "-" * 80
        )

        print(
            f"Source: {source}"
        )

        print(
            f"States: {len(group):,}"
        )

        generated = {}

        # ====================================================
        # TEMPORAL FEATURES
        # ====================================================

        for feature in TREND_FEATURES:

            series = group[feature]

            # ------------------------------------------------
            # 1-MINUTE DIFFERENCE
            # ------------------------------------------------

            generated[
                f"{feature}_Diff1"
            ] = series.diff(1)

            # ------------------------------------------------
            # 5-MINUTE DIFFERENCE
            # ------------------------------------------------

            generated[
                f"{feature}_Diff5"
            ] = series.diff(5)

            # ------------------------------------------------
            # 1-MINUTE PERCENT CHANGE
            # ------------------------------------------------

            generated[
                f"{feature}_PctChange1"
            ] = safe_pct_change(
                series,
                periods=1
            )

            # ------------------------------------------------
            # 5-MINUTE PERCENT CHANGE
            # ------------------------------------------------

            generated[
                f"{feature}_PctChange5"
            ] = safe_pct_change(
                series,
                periods=5
            )

            # ------------------------------------------------
            # 5-MINUTE ROLLING MEAN
            # ------------------------------------------------

            generated[
                f"{feature}_RollingMean5"
            ] = (
                series
                .rolling(
                    window=5,
                    min_periods=1
                )
                .mean()
            )

            # ------------------------------------------------
            # 5-MINUTE ROLLING STD
            # ------------------------------------------------

            generated[
                f"{feature}_RollingStd5"
            ] = (
                series
                .rolling(
                    window=5,
                    min_periods=2
                )
                .std()
            )

            # ------------------------------------------------
            # 5-MINUTE ROLLING Z-SCORE
            # ------------------------------------------------

            generated[
                f"{feature}_ZScore5"
            ] = rolling_zscore(
                series,
                window=5
            )

            # ------------------------------------------------
            # 5-MINUTE SLOPE
            # ------------------------------------------------

            generated[
                f"{feature}_Slope5"
            ] = rolling_slope(
                series,
                window=5
            )

        # ====================================================
        # NETWORK BEHAVIOUR RATIOS
        # ====================================================

        flow = (
            group["Flow_Count"]
            .clip(lower=1)
        )

        fwd_packets = (
            group["Avg_Fwd_Packets"]
            .abs()
            .clip(lower=1e-6)
        )

        fwd_bytes = (
            group["Avg_Fwd_Bytes"]
            .abs()
            .clip(lower=1e-6)
        )

        # ----------------------------------------------------
        # SYN / FLOW
        # ----------------------------------------------------

        generated[
            "SYN_Per_Flow"
        ] = (
            group["SYN_Count"]
            / flow
        )

        # ----------------------------------------------------
        # RST / FLOW
        # ----------------------------------------------------

        generated[
            "RST_Per_Flow"
        ] = (
            group["RST_Count"]
            / flow
        )

        # ----------------------------------------------------
        # PSH / FLOW
        # ----------------------------------------------------

        generated[
            "PSH_Per_Flow"
        ] = (
            group["PSH_Count"]
            / flow
        )

        # ----------------------------------------------------
        # ACK / FLOW
        # ----------------------------------------------------

        generated[
            "ACK_Per_Flow"
        ] = (
            group["ACK_Count"]
            / flow
        )

        # ----------------------------------------------------
        # BWD / FWD PACKET RATIO
        # ----------------------------------------------------

        generated[
            "Bwd_Fwd_Packet_Ratio"
        ] = (
            group["Avg_Bwd_Packets"]
            / fwd_packets
        )

        # ----------------------------------------------------
        # BWD / FWD BYTE RATIO
        # ----------------------------------------------------

        generated[
            "Bwd_Fwd_Byte_Ratio"
        ] = (
            group["Avg_Bwd_Bytes"]
            / fwd_bytes
        )

        # ----------------------------------------------------
        # PORTS / FLOW
        # ----------------------------------------------------

        generated[
            "Ports_Per_Flow"
        ] = (
            group["Unique_Destination_Ports"]
            / flow
        )

        # ====================================================
        # STORE THIS GROUP
        # ====================================================

        generated_groups.append(
            (
                group.index,
                pd.DataFrame(
                    generated
                )
            )
        )

    # ========================================================
    # REBUILD DATAFRAME
    # ========================================================
    #
    # We reconstruct each source using the original rows and
    # generated features.
    # ========================================================

    enhanced_groups = []

    for source, group in df.groupby(
        "Source_File",
        sort=False
    ):

        group = group.sort_values(
            "Minute"
        ).reset_index(
            drop=True
        )

        generated = {}

        # ====================================================
        # RECREATE FEATURES FOR THIS GROUP
        # ====================================================

        for feature in TREND_FEATURES:

            series = group[feature]

            generated[
                f"{feature}_Diff1"
            ] = series.diff(1)

            generated[
                f"{feature}_Diff5"
            ] = series.diff(5)

            generated[
                f"{feature}_PctChange1"
            ] = safe_pct_change(
                series,
                periods=1
            )

            generated[
                f"{feature}_PctChange5"
            ] = safe_pct_change(
                series,
                periods=5
            )

            generated[
                f"{feature}_RollingMean5"
            ] = (
                series
                .rolling(
                    window=5,
                    min_periods=1
                )
                .mean()
            )

            generated[
                f"{feature}_RollingStd5"
            ] = (
                series
                .rolling(
                    window=5,
                    min_periods=2
                )
                .std()
            )

            generated[
                f"{feature}_ZScore5"
            ] = rolling_zscore(
                series,
                window=5
            )

            generated[
                f"{feature}_Slope5"
            ] = rolling_slope(
                series,
                window=5
            )

        # ====================================================
        # RATIOS
        # ====================================================

        flow = (
            group["Flow_Count"]
            .clip(lower=1)
        )

        fwd_packets = (
            group["Avg_Fwd_Packets"]
            .abs()
            .clip(lower=1e-6)
        )

        fwd_bytes = (
            group["Avg_Fwd_Bytes"]
            .abs()
            .clip(lower=1e-6)
        )

        generated[
            "SYN_Per_Flow"
        ] = (
            group["SYN_Count"]
            / flow
        )

        generated[
            "RST_Per_Flow"
        ] = (
            group["RST_Count"]
            / flow
        )

        generated[
            "PSH_Per_Flow"
        ] = (
            group["PSH_Count"]
            / flow
        )

        generated[
            "ACK_Per_Flow"
        ] = (
            group["ACK_Count"]
            / flow
        )

        generated[
            "Bwd_Fwd_Packet_Ratio"
        ] = (
            group["Avg_Bwd_Packets"]
            / fwd_packets
        )

        generated[
            "Bwd_Fwd_Byte_Ratio"
        ] = (
            group["Avg_Bwd_Bytes"]
            / fwd_bytes
        )

        generated[
            "Ports_Per_Flow"
        ] = (
            group["Unique_Destination_Ports"]
            / flow
        )

        generated_df = pd.DataFrame(
            generated
        )

        enhanced_group = pd.concat(
            [
                group,
                generated_df,
            ],
            axis=1
        )

        enhanced_groups.append(
            enhanced_group
        )

    # ========================================================
    # COMBINE SOURCES
    # ========================================================

    df = pd.concat(
        enhanced_groups,
        ignore_index=True
    )

    # ========================================================
    # CLEAN INFINITIES
    # ========================================================

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # ========================================================
    # MODEL FEATURES
    # ========================================================
    #
    # These columns are NEVER allowed into the model:
    #
    # Attack_Flow_Count
    # Attack_Ratio
    # Attack_State
    # Attack_Type
    #
    # They contain or derive from attack labels.
    # ========================================================

    excluded_columns = [
        "Minute",
        "Source_File",
        "Attack_Flow_Count",
        "Attack_Ratio",
        "Attack_State",
        "Attack_Type",
    ]

    model_features = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    # ========================================================
    # VERIFY NUMERIC FEATURES
    # ========================================================

    non_numeric_features = [
        feature
        for feature in model_features
        if not pd.api.types.is_numeric_dtype(
            df[feature]
        )
    ]

    if non_numeric_features:

        raise ValueError(
            "\nNon-numeric model features detected:\n"
            + "\n".join(
                non_numeric_features
            )
        )

    # ========================================================
    # GENERATED FEATURE LIST
    # ========================================================

    new_features = [
        feature
        for feature in model_features
        if feature not in BASE_FEATURES
    ]

    # ========================================================
    # EXPECTED COUNTS
    # ========================================================

    temporal_features_per_base = 8

    expected_temporal_features = (
        len(TREND_FEATURES)
        * temporal_features_per_base
    )

    expected_ratio_features = 7

    expected_new_features = (
        expected_temporal_features
        + expected_ratio_features
    )

    expected_total_features = (
        len(BASE_FEATURES)
        + expected_new_features
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 80)
    print("TEMPORAL FEATURE SUMMARY")
    print("=" * 80)

    print(
        f"\nBase traffic features: "
        f"{len(BASE_FEATURES)}"
    )

    print(
        f"Temporal features: "
        f"{expected_temporal_features}"
    )

    print(
        f"Behaviour ratio features: "
        f"{expected_ratio_features}"
    )

    print(
        f"New temporal/behaviour features: "
        f"{len(new_features)}"
    )

    print(
        f"Total model features: "
        f"{len(model_features)}"
    )

    print(
        f"Expected total features: "
        f"{expected_total_features}"
    )

    if len(model_features) != expected_total_features:

        raise RuntimeError(
            "\nFEATURE COUNT CHECK FAILED!\n"
            f"Expected: {expected_total_features}\n"
            f"Actual:   {len(model_features)}"
        )

    print(
        "\nFeature count check: PASSED"
    )

    # ========================================================
    # LEAKAGE CHECK
    # ========================================================

    leakage_columns = [
        "Attack_Flow_Count",
        "Attack_Ratio",
        "Attack_State",
        "Attack_Type",
    ]

    leakage_in_features = [
        column
        for column in leakage_columns
        if column in model_features
    ]

    if leakage_in_features:

        raise RuntimeError(
            "\nLEAKAGE DETECTED!\n"
            + "\n".join(
                leakage_in_features
            )
        )

    print(
        "\nLeakage check: PASSED"
    )

    print(
        "Excluded label-derived columns:"
    )

    for column in leakage_columns:

        print(
            f"  {column}"
        )

    # ========================================================
    # NaN SUMMARY
    # ========================================================

    nan_counts = (
        df[model_features]
        .isna()
        .sum()
    )

    total_nan = (
        nan_counts.sum()
    )

    print(
        f"\nTotal model-feature NaN values: "
        f"{total_nan:,}"
    )

    print(
        "\nFeatures with most NaN values:"
    )

    print(
        nan_counts
        .sort_values(
            ascending=False
        )
        .head(10)
        .to_string()
    )

    # ========================================================
    # CHRONOLOGICAL CHECK
    # ========================================================

    print(
        "\nChecking chronological ordering..."
    )

    ordering_errors = 0

    for source, group in df.groupby(
        "Source_File",
        sort=False
    ):

        if not group[
            "Minute"
        ].is_monotonic_increasing:

            ordering_errors += 1

            print(
                f"  Ordering problem: {source}"
            )

    if ordering_errors:

        raise RuntimeError(
            "Chronological ordering check failed."
        )

    print(
        "Chronological ordering: PASSED"
    )

    # ========================================================
    # SOURCE BOUNDARIES
    # ========================================================

    print(
        "\nSource capture boundaries:"
    )

    for source, group in df.groupby(
        "Source_File",
        sort=False
    ):

        print(
            f"  {source}"
        )

        print(
            f"    {group['Minute'].min()}"
            f" → "
            f"{group['Minute'].max()}"
        )

        print(
            f"    States: {len(group):,}"
        )

    # ========================================================
    # EXAMPLE FEATURES
    # ========================================================

    print(
        "\nExample generated features:"
    )

    for feature in new_features[:30]:

        print(
            f"  {feature}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n" + "=" * 80)
    print("TEMPORAL FEATURE ENGINEERING COMPLETE")
    print("=" * 80)

    print(
        f"\nInput:"
    )

    print(
        f"  {INPUT_FILE}"
    )

    print(
        f"\nOutput:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    print(
        f"\nRows:"
        f" {len(df):,}"
    )

    print(
        f"Model features:"
        f" {len(model_features)}"
    )

    print(
        "\nSafety:"
    )

    print(
        "  Raw dataset untouched."
    )

    print(
        "  Original network_states.csv untouched."
    )

    print(
        "  Attack_Flow_Count excluded."
    )

    print(
        "  Attack_Ratio excluded."
    )

    print(
        "  Attack_State excluded."
    )

    print(
        "  Attack_Type excluded."
    )

    print(
        "\nNext step:"
    )

    print(
        "Create enhanced temporal sequences."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()