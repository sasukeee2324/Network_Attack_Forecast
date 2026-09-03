from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# FILES
# ============================================================

MODEL_DIR = Path("models")

PREDICTIONS_FILE = (
    MODEL_DIR
    / "pytorch_gru_unseen_infilteration_predictions.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

THRESHOLDS = np.arange(
    0.40,
    0.61,
    0.01
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("UNSEEN INFILTERATION PREDICTION ANALYSIS")
    print("=" * 80)

    # --------------------------------------------------------
    # Load predictions
    # --------------------------------------------------------

    if not PREDICTIONS_FILE.exists():

        raise FileNotFoundError(
            f"Prediction file not found:\n"
            f"{PREDICTIONS_FILE}"
        )

    df = pd.read_csv(
        PREDICTIONS_FILE
    )

    required_columns = {
        "Actual",
        "Probability",
        "Prediction",
        "Threshold",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    y_true = df["Actual"].to_numpy(
        dtype=int
    )

    probabilities = df[
        "Probability"
    ].to_numpy(
        dtype=float
    )

    print(
        f"\nTotal samples: {len(df)}"
    )

    print(
        f"Benign samples: "
        f"{np.sum(y_true == 0)}"
    )

    print(
        f"Infilteration samples: "
        f"{np.sum(y_true == 1)}"
    )

    # ========================================================
    # 1. PROBABILITY SEPARATION
    # ========================================================

    print("\n" + "=" * 80)
    print("1. PROBABILITY SEPARATION")
    print("=" * 80)

    benign_prob = probabilities[
        y_true == 0
    ]

    attack_prob = probabilities[
        y_true == 1
    ]

    def describe(
        name,
        values
    ):

        print(
            f"\n{name}:"
        )

        print(
            f"  Count:  {len(values)}"
        )

        print(
            f"  Mean:   {np.mean(values):.6f}"
        )

        print(
            f"  Std:    {np.std(values):.6f}"
        )

        print(
            f"  Min:    {np.min(values):.6f}"
        )

        print(
            f"  25%:    {np.percentile(values, 25):.6f}"
        )

        print(
            f"  Median: {np.median(values):.6f}"
        )

        print(
            f"  75%:    {np.percentile(values, 75):.6f}"
        )

        print(
            f"  Max:    {np.max(values):.6f}"
        )

    describe(
        "BENIGN",
        benign_prob
    )

    describe(
        "INFILTERATION",
        attack_prob
    )

    mean_difference = (
        np.mean(attack_prob)
        - np.mean(benign_prob)
    )

    print(
        f"\nMean probability difference "
        f"(Attack - Benign): "
        f"{mean_difference:.6f}"
    )

    # ========================================================
    # 2. OVERLAP
    # ========================================================

    print("\n" + "=" * 80)
    print("2. PROBABILITY OVERLAP")
    print("=" * 80)

    print(
        "\nHow often does the model assign "
        "higher probability to benign traffic?"
    )

    benign_above_attack_median = np.mean(
        benign_prob
        >= np.median(attack_prob)
    )

    attack_below_benign_median = np.mean(
        attack_prob
        <= np.median(benign_prob)
    )

    print(
        f"\nBenign probability >= "
        f"attack median: "
        f"{benign_above_attack_median:.2%}"
    )

    print(
        f"Attack probability <= "
        f"benign median: "
        f"{attack_below_benign_median:.2%}"
    )

    # ========================================================
    # 3. PERCENTILES
    # ========================================================

    print("\n" + "=" * 80)
    print("3. PROBABILITY PERCENTILES")
    print("=" * 80)

    percentile_rows = []

    for percentile in [
        1,
        5,
        10,
        25,
        50,
        75,
        90,
        95,
        99,
    ]:

        percentile_rows.append({

            "Percentile":
                percentile,

            "Benign":
                np.percentile(
                    benign_prob,
                    percentile
                ),

            "Infilteration":
                np.percentile(
                    attack_prob,
                    percentile
                ),
        })

    percentile_df = pd.DataFrame(
        percentile_rows
    )

    print(
        percentile_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}"
        )
    )

    # ========================================================
    # 4. ROC / PR
    # ========================================================

    print("\n" + "=" * 80)
    print("4. RANKING QUALITY")
    print("=" * 80)

    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities
    )

    print(
        f"\nROC-AUC: {roc_auc:.6f}"
    )

    print(
        f"PR-AUC:  {pr_auc:.6f}"
    )

    print(
        "\nInterpretation:"
    )

    if roc_auc < 0.55:

        print(
            "  ROC-AUC is close to random."
        )

    elif roc_auc < 0.70:

        print(
            "  ROC-AUC shows weak separation."
        )

    elif roc_auc < 0.80:

        print(
            "  ROC-AUC shows moderate separation."
        )

    else:

        print(
            "  ROC-AUC shows strong separation."
        )

    # ========================================================
    # 5. THRESHOLD ANALYSIS
    # ========================================================

    print("\n" + "=" * 80)
    print("5. THRESHOLD ANALYSIS")
    print("=" * 80)

    threshold_rows = []

    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        true_negative = np.sum(
            (y_true == 0)
            & (predictions == 0)
        )

        false_positive = np.sum(
            (y_true == 0)
            & (predictions == 1)
        )

        false_negative = np.sum(
            (y_true == 1)
            & (predictions == 0)
        )

        true_positive = np.sum(
            (y_true == 1)
            & (predictions == 1)
        )

        precision = (
            true_positive
            / max(
                true_positive
                + false_positive,
                1
            )
        )

        recall = (
            true_positive
            / max(
                true_positive
                + false_negative,
                1
            )
        )

        f1 = (
            2
            * precision
            * recall
            / max(
                precision + recall,
                1e-12
            )
        )

        specificity = (
            true_negative
            / max(
                true_negative
                + false_positive,
                1
            )
        )

        threshold_rows.append({

            "Threshold":
                threshold,

            "True_Negative":
                true_negative,

            "False_Positive":
                false_positive,

            "False_Negative":
                false_negative,

            "True_Positive":
                true_positive,

            "Precision":
                precision,

            "Recall":
                recall,

            "F1":
                f1,

            "Specificity":
                specificity,
        })

    threshold_df = pd.DataFrame(
        threshold_rows
    )

    print(
        threshold_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    # ========================================================
    # 6. BEST THRESHOLD BY F1
    # ========================================================

    best_row = threshold_df.loc[
        threshold_df["F1"].idxmax()
    ]

    print("\n" + "=" * 80)
    print("6. BEST THRESHOLD IN RETROSPECT")
    print("=" * 80)

    print(
        f"\nBest threshold: "
        f"{best_row['Threshold']:.2f}"
    )

    print(
        f"Precision:    "
        f"{best_row['Precision']:.4f}"
    )

    print(
        f"Recall:       "
        f"{best_row['Recall']:.4f}"
    )

    print(
        f"F1:           "
        f"{best_row['F1']:.4f}"
    )

    print(
        f"Specificity:  "
        f"{best_row['Specificity']:.4f}"
    )

    print(
        "\nWARNING:"
    )

    print(
        "This threshold analysis is diagnostic only."
    )

    print(
        "Do NOT use the unseen-test threshold "
        "to claim model performance."
    )

    # ========================================================
    # 7. CURRENT 0.50 BEHAVIOUR
    # ========================================================

    print("\n" + "=" * 80)
    print("7. CURRENT 0.50 THRESHOLD")
    print("=" * 80)

    current_predictions = (
        probabilities >= 0.50
    ).astype(int)

    benign_predicted_attack = np.sum(
        (y_true == 0)
        & (current_predictions == 1)
    )

    benign_total = np.sum(
        y_true == 0
    )

    attack_predicted_attack = np.sum(
        (y_true == 1)
        & (current_predictions == 1)
    )

    attack_total = np.sum(
        y_true == 1
    )

    print(
        f"\nBenign predicted as attack:"
        f" {benign_predicted_attack}/"
        f"{benign_total}"
        f" ({benign_predicted_attack / benign_total:.2%})"
    )

    print(
        f"Infilteration detected:"
        f" {attack_predicted_attack}/"
        f"{attack_total}"
        f" ({attack_predicted_attack / attack_total:.2%})"
    )

    # ========================================================
    # 8. CONCLUSION
    # ========================================================

    print("\n" + "=" * 80)
    print("8. CONCLUSION")
    print("=" * 80)

    print()

    if (
        np.std(probabilities)
        < 0.01
    ):

        print(
            "The model probabilities are extremely compressed."
        )

        print(
            "The GRU is producing almost the same score "
            "for every sample."
        )

        print(
            "The current 0.50 classifier is therefore "
            "not meaningfully separating benign and "
            "Infilteration traffic."
        )

    elif mean_difference > 0:

        print(
            "Infilteration receives higher probabilities "
            "on average than benign traffic."
        )

        print(
            "There is evidence of some useful separation."
        )

    else:

        print(
            "The model does not assign higher probabilities "
            "to Infilteration on average."
        )

    print(
        "\nROC-AUC and probability separation are more "
        "informative here than the raw 0.50 F1."
    )

    # ========================================================
    # SAVE
    # ========================================================

    threshold_path = (
        MODEL_DIR
        / "pytorch_gru_unseen_threshold_diagnostics.csv"
    )

    percentile_path = (
        MODEL_DIR
        / "pytorch_gru_unseen_probability_percentiles.csv"
    )

    threshold_df.to_csv(
        threshold_path,
        index=False
    )

    percentile_df.to_csv(
        percentile_path,
        index=False
    )

    print("\nSaved:")

    print(
        f"  {threshold_path}"
    )

    print(
        f"  {percentile_path}"
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()