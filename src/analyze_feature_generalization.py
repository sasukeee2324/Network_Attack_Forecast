from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/processed/unseen_attack")
MODEL_DIR = Path("models")

FEATURE_NAMES = [
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
# HELPERS
# ============================================================

def flatten_temporal_data(X):
    """
    Convert:
        (samples, 10 minutes, 18 features)

    into:
        (samples * 10, 18)

    This lets us inspect feature distributions
    across all network states.
    """

    return X.reshape(
        -1,
        X.shape[-1]
    )


def summarize_features(
    X,
    name
):

    flat = flatten_temporal_data(X)

    rows = []

    for i, feature in enumerate(
        FEATURE_NAMES
    ):

        values = flat[:, i]

        values = values[
            np.isfinite(values)
        ]

        if len(values) == 0:

            continue

        rows.append({

            "Feature":
                feature,

            "Mean":
                np.mean(values),

            "Std":
                np.std(values),

            "Median":
                np.median(values),

            "Min":
                np.min(values),

            "Max":
                np.max(values),

            "P25":
                np.percentile(
                    values,
                    25
                ),

            "P75":
                np.percentile(
                    values,
                    75
                ),

            "Dataset":
                name
        })

    return pd.DataFrame(rows)


def compare_groups(
    benign,
    attack,
    attack_name
):

    rows = []

    for i, feature in enumerate(
        FEATURE_NAMES
    ):

        benign_values = benign[:, i]
        attack_values = attack[:, i]

        benign_values = benign_values[
            np.isfinite(benign_values)
        ]

        attack_values = attack_values[
            np.isfinite(attack_values)
        ]

        if (
            len(benign_values) == 0
            or len(attack_values) == 0
        ):

            continue

        benign_mean = np.mean(
            benign_values
        )

        attack_mean = np.mean(
            attack_values
        )

        benign_median = np.median(
            benign_values
        )

        attack_median = np.median(
            attack_values
        )

        # Standardized difference using
        # pooled standard deviation.
        benign_std = np.std(
            benign_values
        )

        attack_std = np.std(
            attack_values
        )

        pooled_std = np.sqrt(
            (
                benign_std ** 2
                + attack_std ** 2
            ) / 2
        )

        if pooled_std > 1e-12:

            effect_size = (
                attack_mean
                - benign_mean
            ) / pooled_std

        else:

            effect_size = 0.0

        rows.append({

            "Feature":
                feature,

            "Benign_Mean":
                benign_mean,

            f"{attack_name}_Mean":
                attack_mean,

            "Mean_Difference":
                attack_mean
                - benign_mean,

            "Benign_Median":
                benign_median,

            f"{attack_name}_Median":
                attack_median,

            "Effect_Size":
                effect_size,

            "Absolute_Effect_Size":
                abs(effect_size)
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("FEATURE GENERALIZATION ANALYSIS")
    print("=" * 80)

    # ========================================================
    # LOAD DATA
    # ========================================================

    X_train = np.load(
        DATA_DIR / "X_train.npy"
    )

    y_train = np.load(
        DATA_DIR / "y_train.npy"
    )

    X_unseen = np.load(
        DATA_DIR / "X_unseen.npy"
    )

    y_unseen = np.load(
        DATA_DIR / "y_unseen.npy"
    )

    print("\nLoaded:")

    print(
        f"X_train:  {X_train.shape}"
    )

    print(
        f"y_train:  {y_train.shape}"
    )

    print(
        f"X_unseen: {X_unseen.shape}"
    )

    print(
        f"y_unseen: {y_unseen.shape}"
    )

    # ========================================================
    # BASIC VALIDATION
    # ========================================================

    if X_train.shape[-1] != len(
        FEATURE_NAMES
    ):

        raise ValueError(
            "Number of features in X_train "
            "does not match FEATURE_NAMES."
        )

    if X_unseen.shape[-1] != len(
        FEATURE_NAMES
    ):

        raise ValueError(
            "Number of features in X_unseen "
            "does not match FEATURE_NAMES."
        )

    # ========================================================
    # FLATTEN
    # ========================================================

    train_flat = flatten_temporal_data(
        X_train
    )

    unseen_flat = flatten_temporal_data(
        X_unseen
    )

    train_labels = np.repeat(
        y_train,
        X_train.shape[1]
    )

    unseen_labels = np.repeat(
        y_unseen,
        X_unseen.shape[1]
    )

    # ========================================================
    # GROUPS
    # ========================================================

    known_benign = train_flat[
        train_labels == 0
    ]

    known_attack = train_flat[
        train_labels == 1
    ]

    unseen_benign = unseen_flat[
        unseen_labels == 0
    ]

    unseen_infilteration = unseen_flat[
        unseen_labels == 1
    ]

    print("\n" + "=" * 80)
    print("NETWORK STATE COUNTS")
    print("=" * 80)

    print(
        f"\nKnown benign states: "
        f"{len(known_benign):,}"
    )

    print(
        f"Known attack states: "
        f"{len(known_attack):,}"
    )

    print(
        f"Unseen benign states: "
        f"{len(unseen_benign):,}"
    )

    print(
        f"Unseen Infilteration states: "
        f"{len(unseen_infilteration):,}"
    )

    # ========================================================
    # FEATURE SUMMARY
    # ========================================================

    print("\n" + "=" * 80)
    print("1. FEATURE DISTRIBUTIONS")
    print("=" * 80)

    summaries = []

    summaries.append(
        summarize_features(
            known_benign,
            "Known_Benign"
        )
    )

    summaries.append(
        summarize_features(
            known_attack,
            "Known_Attack"
        )
    )

    summaries.append(
        summarize_features(
            unseen_benign,
            "Unseen_Benign"
        )
    )

    summaries.append(
        summarize_features(
            unseen_infilteration,
            "Unseen_Infilteration"
        )
    )

    summary_df = pd.concat(
        summaries,
        ignore_index=True
    )

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    # ========================================================
    # KNOWN ATTACK VS KNOWN BENIGN
    # ========================================================

    print("\n" + "=" * 80)
    print("2. KNOWN ATTACK VS KNOWN BENIGN")
    print("=" * 80)

    known_comparison = compare_groups(
        known_benign,
        known_attack,
        "Known_Attack"
    )

    known_comparison = (
        known_comparison
        .sort_values(
            "Absolute_Effect_Size",
            ascending=False
        )
    )

    print(
        "\nFeatures with strongest "
        "known-attack separation:"
    )

    print(
        known_comparison[
            [
                "Feature",
                "Benign_Mean",
                "Known_Attack_Mean",
                "Effect_Size"
            ]
        ].head(18).to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    # ========================================================
    # UNSEEN INFILTERATION VS UNSEEN BENIGN
    # ========================================================

    print("\n" + "=" * 80)
    print("3. UNSEEN INFILTERATION VS UNSEEN BENIGN")
    print("=" * 80)

    unseen_comparison = compare_groups(
        unseen_benign,
        unseen_infilteration,
        "Infilteration"
    )

    unseen_comparison = (
        unseen_comparison
        .sort_values(
            "Absolute_Effect_Size",
            ascending=False
        )
    )

    print(
        "\nFeatures with strongest "
        "Infilteration separation:"
    )

    print(
        unseen_comparison[
            [
                "Feature",
                "Benign_Mean",
                "Infilteration_Mean",
                "Effect_Size"
            ]
        ].head(18).to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    # ========================================================
    # GENERALIZATION GAP
    # ========================================================

    print("\n" + "=" * 80)
    print("4. GENERALIZATION GAP")
    print("=" * 80)

    merged = known_comparison[
        [
            "Feature",
            "Effect_Size"
        ]
    ].rename(

        columns={
            "Effect_Size":
                "Known_Effect_Size"
        }
    )

    unseen_effects = unseen_comparison[
        [
            "Feature",
            "Effect_Size"
        ]
    ].rename(

        columns={
            "Effect_Size":
                "Unseen_Effect_Size"
        }
    )

    gap_df = merged.merge(
        unseen_effects,
        on="Feature"
    )

    gap_df["Effect_Size_Gap"] = (
        gap_df["Unseen_Effect_Size"]
        - gap_df["Known_Effect_Size"]
    )

    gap_df["Generalization_Ratio"] = (
        gap_df["Unseen_Effect_Size"]
        / (
            gap_df["Known_Effect_Size"]
            .abs()
            + 1e-12
        )
    )

    gap_df["Unseen_Absolute_Effect"] = (
        gap_df["Unseen_Effect_Size"]
        .abs()
    )

    gap_df = gap_df.sort_values(
        "Unseen_Absolute_Effect",
        ascending=False
    )

    print(
        "\nFeature generalization:"
    )

    print(
        gap_df[
            [
                "Feature",
                "Known_Effect_Size",
                "Unseen_Effect_Size",
                "Generalization_Ratio"
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    # ========================================================
    # FEATURES THAT CHANGE DIRECTION
    # ========================================================

    print("\n" + "=" * 80)
    print("5. FEATURE DIRECTION CHANGES")
    print("=" * 80)

    direction_changes = []

    for _, row in gap_df.iterrows():

        known_effect = row[
            "Known_Effect_Size"
        ]

        unseen_effect = row[
            "Unseen_Effect_Size"
        ]

        if (
            known_effect != 0
            and unseen_effect != 0
            and np.sign(known_effect)
            != np.sign(unseen_effect)
        ):

            direction_changes.append(
                row
            )

    if direction_changes:

        direction_df = pd.DataFrame(
            direction_changes
        )

        print(
            "\nThese features behave in "
            "opposite directions for known "
            "attacks and Infilteration:"
        )

        print(
            direction_df[
                [
                    "Feature",
                    "Known_Effect_Size",
                    "Unseen_Effect_Size"
                ]
            ].to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}"
            )
        )

    else:

        print(
            "\nNo major feature direction "
            "changes detected."
        )

    # ========================================================
    # FEATURES WITH USEFUL UNSEEN SIGNAL
    # ========================================================

    print("\n" + "=" * 80)
    print("6. FEATURES WITH UNSEEN SIGNAL")
    print("=" * 80)

    useful_unseen = gap_df[
        gap_df["Unseen_Absolute_Effect"]
        >= 0.20
    ]

    if len(useful_unseen) > 0:

        print(
            "\nFeatures with absolute "
            "unseen effect >= 0.20:"
        )

        print(
            useful_unseen[
                [
                    "Feature",
                    "Known_Effect_Size",
                    "Unseen_Effect_Size",
                    "Generalization_Ratio"
                ]
            ].to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}"
            )
        )

    else:

        print(
            "\nNo features reached an "
            "absolute effect size of 0.20."
        )

    # ========================================================
    # FEATURE VARIABILITY
    # ========================================================

    print("\n" + "=" * 80)
    print("7. FEATURE VARIABILITY")
    print("=" * 80)

    variability_rows = []

    for i, feature in enumerate(
        FEATURE_NAMES
    ):

        known_std = np.std(
            known_benign[:, i]
        )

        unseen_std = np.std(
            unseen_benign[:, i]
        )

        variability_rows.append({

            "Feature":
                feature,

            "Known_Benign_Std":
                known_std,

            "Unseen_Benign_Std":
                unseen_std,

            "Std_Ratio":
                unseen_std
                / (
                    known_std
                    + 1e-12
                )
        })

    variability_df = pd.DataFrame(
        variability_rows
    )

    variability_df = (
        variability_df
        .sort_values(
            "Std_Ratio",
            ascending=False
        )
    )

    print(
        variability_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_path = (
        MODEL_DIR
        / "feature_generalization_summary.csv"
    )

    known_path = (
        MODEL_DIR
        / "feature_generalization_known.csv"
    )

    unseen_path = (
        MODEL_DIR
        / "feature_generalization_unseen.csv"
    )

    gap_path = (
        MODEL_DIR
        / "feature_generalization_gap.csv"
    )

    variability_path = (
        MODEL_DIR
        / "feature_generalization_variability.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    known_comparison.to_csv(
        known_path,
        index=False
    )

    unseen_comparison.to_csv(
        unseen_path,
        index=False
    )

    gap_df.to_csv(
        gap_path,
        index=False
    )

    variability_df.to_csv(
        variability_path,
        index=False
    )

    # ========================================================
    # FINAL INTERPRETATION
    # ========================================================

    print("\n" + "=" * 80)
    print("8. INTERPRETATION")
    print("=" * 80)

    strongest_unseen = (
        unseen_comparison
        .iloc[0]
    )

    print(
        f"\nStrongest unseen feature: "
        f"{strongest_unseen['Feature']}"
    )

    print(
        f"Unseen effect size: "
        f"{strongest_unseen['Effect_Size']:.4f}"
    )

    known_strongest = (
        known_comparison
        .iloc[0]
    )

    print(
        f"\nStrongest known-attack feature: "
        f"{known_strongest['Feature']}"
    )

    print(
        f"Known effect size: "
        f"{known_strongest['Effect_Size']:.4f}"
    )

    print(
        "\nWhat we are looking for:"
    )

    print(
        "1. Features that distinguish known attacks."
    )

    print(
        "2. Features that ALSO distinguish "
        "unseen Infilteration."
    )

    print(
        "3. Features whose behaviour changes "
        "substantially between known and unseen attacks."
    )

    print(
        "4. Features with meaningful temporal "
        "variation that the current model may "
        "not be exploiting."
    )

    print("\nSaved:")

    print(
        f"  {summary_path}"
    )

    print(
        f"  {known_path}"
    )

    print(
        f"  {unseen_path}"
    )

    print(
        f"  {gap_path}"
    )

    print(
        f"  {variability_path}"
    )

    print("\n" + "=" * 80)
    print("FEATURE GENERALIZATION ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()