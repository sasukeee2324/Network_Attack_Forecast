from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

IMPORTANCE_FILE = Path(
    "models/xgboost_enhanced_v2_feature_importance.csv"
)

ENHANCED_FEATURE_FILE = Path(
    "data/processed/temporal/network_states_enhanced.csv"
)

OUTPUT_DIR = Path("models")

TOP_N = 30


# ============================================================
# FEATURE FAMILY
# ============================================================

def classify_feature(name):
    """
    Identify what kind of feature this is.

    The feature name comes from the actual enhanced
    network_states_enhanced.csv file.
    """

    name = str(name)

    # Base traffic features have none of these suffixes.
    if "RollingMean" in name:
        return "Rolling Mean"

    if "RollingStd" in name:
        return "Rolling Std"

    if "ZScore" in name:
        return "Z-Score"

    if "PctChange" in name:
        return "Percentage Change"

    if "Diff" in name:
        return "Difference"

    if "Slope" in name:
        return "Slope"

    if "Ratio" in name:
        return "Behaviour Ratio"

    return "Base Traffic"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("ENHANCED XGBOOST FEATURE IMPORTANCE ANALYSIS V2")
    print("=" * 80)

    # ========================================================
    # CHECK FILES
    # ========================================================

    if not IMPORTANCE_FILE.exists():

        raise FileNotFoundError(
            f"\nImportance file not found:\n"
            f"{IMPORTANCE_FILE}"
        )

    if not ENHANCED_FEATURE_FILE.exists():

        raise FileNotFoundError(
            f"\nEnhanced feature file not found:\n"
            f"{ENHANCED_FEATURE_FILE}"
        )

    # ========================================================
    # LOAD IMPORTANCE
    # ========================================================

    importance_df = pd.read_csv(
        IMPORTANCE_FILE
    )

    print(
        f"\nImportance rows: "
        f"{len(importance_df)}"
    )

    # ========================================================
    # LOAD REAL FEATURE NAMES
    # ========================================================

    print(
        "\nLoading actual feature names..."
    )

    enhanced_df = pd.read_csv(
        ENHANCED_FEATURE_FILE,
        nrows=1
    )

    # These are the columns used by the enhanced model.
    excluded_columns = {
        "Minute",
        "Attack_Flow_Count",
        "Attack_Ratio",
        "Attack_Type",
        "Attack_State",
        "Source_File",
    }

    feature_names = [
        column
        for column in enhanced_df.columns
        if column not in excluded_columns
    ]

    print(
        f"Actual model features: "
        f"{len(feature_names)}"
    )

    if len(feature_names) != 161:

        raise ValueError(
            f"Expected 161 model features, "
            f"found {len(feature_names)}."
        )

    # ========================================================
    # MAP FLATTENED FEATURES
    # ========================================================

    mapped_names = []

    for feature in importance_df["Feature"]:

        feature = str(feature)

        # Expected format:
        #
        # timestep_1_feature_1
        # timestep_10_feature_161
        #
        parts = feature.split("_")

        if len(parts) != 4:

            raise ValueError(
                f"Unexpected feature format: "
                f"{feature}"
            )

        timestep = int(
            parts[1]
        )

        feature_index = int(
            parts[3]
        )

        if timestep < 1 or timestep > 10:

            raise ValueError(
                f"Invalid timestep: "
                f"{timestep}"
            )

        if feature_index < 1 or feature_index > 161:

            raise ValueError(
                f"Invalid feature index: "
                f"{feature_index}"
            )

        real_name = feature_names[
            feature_index - 1
        ]

        mapped_names.append(
            real_name
        )

    importance_df[
        "Timestep"
    ] = (
        importance_df["Feature"]
        .str.extract(
            r"timestep_(\d+)"
        )[0]
        .astype(int)
    )

    importance_df[
        "Feature_Index"
    ] = (
        importance_df["Feature"]
        .str.extract(
            r"feature_(\d+)"
        )[0]
        .astype(int)
    )

    importance_df[
        "Real_Feature"
    ] = mapped_names

    importance_df[
        "Feature_Family"
    ] = importance_df[
        "Real_Feature"
    ].apply(
        classify_feature
    )

    importance_df["Importance"] = pd.to_numeric(
        importance_df["Importance"],
        errors="coerce"
    )

    importance_df = importance_df.dropna(
        subset=["Importance"]
    )

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # 1. TOP FEATURES
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "1. TOP FEATURES"
    )

    print(
        "=" * 80
    )

    total_importance = (
        importance_df["Importance"].sum()
    )

    top = importance_df.head(
        TOP_N
    ).copy()

    top[
        "Cumulative_Percent"
    ] = (
        top["Importance"].cumsum()
        / total_importance
        * 100
    )

    print()

    print(
        top[
            [
                "Timestep",
                "Feature_Index",
                "Real_Feature",
                "Feature_Family",
                "Importance",
                "Cumulative_Percent",
            ]
        ].to_string(
            index=False,
            formatters={
                "Importance":
                    "{:.6f}".format,
                "Cumulative_Percent":
                    "{:.2f}%".format,
            }
        )
    )

    # ========================================================
    # 2. UNIQUE FEATURE IMPORTANCE
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "2. AGGREGATED REAL FEATURE IMPORTANCE"
    )

    print(
        "=" * 80
    )

    # A real feature can appear at all 10 timesteps.
    #
    # Sum importance across timesteps to discover which
    # behavioral variables matter most overall.

    aggregated = (
        importance_df
        .groupby(
            [
                "Feature_Index",
                "Real_Feature",
                "Feature_Family",
            ]
        )["Importance"]
        .agg(
            [
                "sum",
                "mean",
                "max",
                "count",
            ]
        )
        .sort_values(
            "sum",
            ascending=False
        )
        .reset_index()
    )

    aggregated[
        "Percent_of_Total"
    ] = (
        aggregated["sum"]
        / aggregated["sum"].sum()
        * 100
    )

    print()

    print(
        aggregated.head(30).to_string(
            index=False,
            formatters={
                "sum":
                    "{:.6f}".format,
                "mean":
                    "{:.6f}".format,
                "max":
                    "{:.6f}".format,
                "Percent_of_Total":
                    "{:.2f}%".format,
            }
        )
    )

    # ========================================================
    # 3. FEATURE FAMILY ANALYSIS
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "3. FEATURE FAMILY ANALYSIS"
    )

    print(
        "=" * 80
    )

    family = (
        importance_df
        .groupby(
            "Feature_Family"
        )["Importance"]
        .agg(
            [
                "count",
                "sum",
                "mean",
                "max",
            ]
        )
        .sort_values(
            "sum",
            ascending=False
        )
    )

    family[
        "Percent_of_Total"
    ] = (
        family["sum"]
        / family["sum"].sum()
        * 100
    )

    print()

    print(
        family.to_string(
            formatters={
                "sum":
                    "{:.6f}".format,
                "mean":
                    "{:.6f}".format,
                "max":
                    "{:.6f}".format,
                "Percent_of_Total":
                    "{:.2f}%".format,
            }
        )
    )

    # ========================================================
    # 4. TEMPORAL VS BASE
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "4. BASE VS TEMPORAL IMPORTANCE"
    )

    print(
        "=" * 80
    )

    temporal_families = {
        "Rolling Mean",
        "Rolling Std",
        "Z-Score",
        "Percentage Change",
        "Difference",
        "Slope",
        "Behaviour Ratio",
    }

    importance_df[
        "Temporal_or_Base"
    ] = importance_df[
        "Feature_Family"
    ].apply(
        lambda x:
        "Temporal / Behaviour"
        if x in temporal_families
        else "Base Traffic"
    )

    temporal_summary = (
        importance_df
        .groupby(
            "Temporal_or_Base"
        )["Importance"]
        .agg(
            [
                "count",
                "sum",
                "mean",
            ]
        )
    )

    temporal_summary[
        "Percent_of_Total"
    ] = (
        temporal_summary["sum"]
        / temporal_summary["sum"].sum()
        * 100
    )

    print()

    print(
        temporal_summary.to_string(
            formatters={
                "sum":
                    "{:.6f}".format,
                "mean":
                    "{:.6f}".format,
                "Percent_of_Total":
                    "{:.2f}%".format,
            }
        )
    )

    # ========================================================
    # 5. TIMESTEP IMPORTANCE
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "5. TEMPORAL TIMESTEP IMPORTANCE"
    )

    print(
        "=" * 80
    )

    timestep = (
        importance_df
        .groupby(
            "Timestep"
        )["Importance"]
        .agg(
            [
                "count",
                "sum",
                "mean",
                "max",
            ]
        )
        .sort_index()
    )

    timestep[
        "Percent_of_Total"
    ] = (
        timestep["sum"]
        / timestep["sum"].sum()
        * 100
    )

    print()

    print(
        timestep.to_string(
            formatters={
                "sum":
                    "{:.6f}".format,
                "mean":
                    "{:.6f}".format,
                "max":
                    "{:.6f}".format,
                "Percent_of_Total":
                    "{:.2f}%".format,
            }
        )
    )

    # ========================================================
    # 6. TOP TEMPORAL FEATURES
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "6. TOP TEMPORAL / BEHAVIOUR FEATURES"
    )

    print(
        "=" * 80
    )

    temporal_features = (
        importance_df[
            importance_df[
                "Temporal_or_Base"
            ]
            == "Temporal / Behaviour"
        ]
        .copy()
    )

    print()

    print(
        temporal_features.head(30)[
            [
                "Timestep",
                "Real_Feature",
                "Feature_Family",
                "Importance",
            ]
        ].to_string(
            index=False,
            formatters={
                "Importance":
                    "{:.6f}".format,
            }
        )
    )

    # ========================================================
    # 7. TOP BASE FEATURES
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "7. TOP BASE TRAFFIC FEATURES"
    )

    print(
        "=" * 80
    )

    base_features = (
        importance_df[
            importance_df[
                "Temporal_or_Base"
            ]
            == "Base Traffic"
        ]
        .copy()
    )

    print()

    print(
        base_features.head(30)[
            [
                "Timestep",
                "Real_Feature",
                "Importance",
            ]
        ].to_string(
            index=False,
            formatters={
                "Importance":
                    "{:.6f}".format,
            }
        )
    )

    # ========================================================
    # 8. IMPORTANCE CONCENTRATION
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "8. IMPORTANCE CONCENTRATION"
    )

    print(
        "=" * 80
    )

    for percent in [
        1,
        5,
        10,
        20,
        50,
    ]:

        count = max(
            1,
            int(
                len(importance_df)
                * percent
                / 100
            )
        )

        contribution = (
            importance_df
            .head(count)["Importance"]
            .sum()
            / total_importance
            * 100
        )

        print(
            f"Top {percent:>2}% "
            f"({count:>4} inputs): "
            f"{contribution:6.2f}%"
        )

    # ========================================================
    # 9. ZERO IMPORTANCE
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "9. ZERO-IMPORTANCE INPUTS"
    )

    print(
        "=" * 80
    )

    zero = importance_df[
        importance_df["Importance"] == 0
    ]

    print(
        f"\nZero-importance inputs: "
        f"{len(zero)} / "
        f"{len(importance_df)}"
    )

    # ========================================================
    # 10. ZERO-IMPORTANCE REAL FEATURES
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "10. REAL FEATURES WITH NO IMPORTANCE"
    )

    print(
        "=" * 80
    )

    real_feature_importance = (
        importance_df
        .groupby(
            [
                "Feature_Index",
                "Real_Feature",
                "Feature_Family",
            ]
        )["Importance"]
        .sum()
        .reset_index()
    )

    unused_real_features = (
        real_feature_importance[
            real_feature_importance[
                "Importance"
            ] == 0
        ]
    )

    print(
        f"\nUnused real features: "
        f"{len(unused_real_features)} / 161"
    )

    if len(unused_real_features) > 0:

        print()

        print(
            unused_real_features[
                [
                    "Feature_Index",
                    "Real_Feature",
                    "Feature_Family",
                ]
            ].to_string(
                index=False
            )
        )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    mapped_file = (
        OUTPUT_DIR /
        "xgboost_enhanced_v2_mapped_importance.csv"
    )

    aggregated_file = (
        OUTPUT_DIR /
        "xgboost_enhanced_v2_aggregated_features.csv"
    )

    family_file = (
        OUTPUT_DIR /
        "xgboost_enhanced_v2_feature_families_v2.csv"
    )

    timestep_file = (
        OUTPUT_DIR /
        "xgboost_enhanced_v2_timestep_importance_v2.csv"
    )

    temporal_file = (
        OUTPUT_DIR /
        "xgboost_enhanced_v2_temporal_importance.csv"
    )

    mapped_file_df = importance_df[
        [
            "Feature",
            "Timestep",
            "Feature_Index",
            "Real_Feature",
            "Feature_Family",
            "Temporal_or_Base",
            "Importance",
        ]
    ]

    mapped_file_df.to_csv(
        mapped_file,
        index=False
    )

    aggregated.to_csv(
        aggregated_file,
        index=False
    )

    family.reset_index().to_csv(
        family_file,
        index=False
    )

    timestep.reset_index().to_csv(
        timestep_file,
        index=False
    )

    temporal_features.to_csv(
        temporal_file,
        index=False
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "FEATURE IMPORTANCE ANALYSIS COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "\nGenerated:"
    )

    print(
        f"  {mapped_file}"
    )

    print(
        f"  {aggregated_file}"
    )

    print(
        f"  {family_file}"
    )

    print(
        f"  {timestep_file}"
    )

    print(
        f"  {temporal_file}"
    )

    print(
        "\nThe analysis now uses the actual"
        " 161 enhanced feature names."
    )

    print(
        "\nDo NOT prune features yet."
    )

    print(
        "We will use these results to decide"
        " whether feature reduction is justified."
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()