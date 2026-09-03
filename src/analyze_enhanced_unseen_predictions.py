"""
Analyze original vs enhanced XGBoost on unseen Infilteration.

This is a DIAGNOSTIC analysis only.

Models:
    Original XGBoost: 18 features
    Enhanced XGBoost: 161 features

Unseen attack:
    Infilteration

Important:
    Threshold analysis is diagnostic.
    Do NOT select a threshold from the unseen test set
    for the headline/generalization result.
"""

import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)


# ============================================================
# CONFIGURATION
# ============================================================

ORIGINAL_PATH = (
    "models/xgboost_unseen_infilteration_predictions.csv"
)

ENHANCED_PATH = (
    "models/xgboost_enhanced_unseen_infilteration_predictions.csv"
)

OUTPUT_DIR = "models"

THRESHOLD_ANALYSIS_PATH = os.path.join(
    OUTPUT_DIR,
    "xgboost_unseen_threshold_comparison.csv"
)

SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "xgboost_unseen_generalization_comparison.csv"
)

CURVE_PATH = os.path.join(
    OUTPUT_DIR,
    "xgboost_unseen_curve_data.csv"
)

PROBABILITY_SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "xgboost_unseen_probability_comparison.csv"
)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_predictions(path, model_name):

    print(f"\nLoading {model_name}:")
    print(f"  {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Prediction file not found:\n{path}"
        )

    df = pd.read_csv(path)

    # --------------------------------------------------------
    # Normalize target-column naming
    #
    # Original XGBoost:
    #     Actual
    #
    # Enhanced XGBoost:
    #     Future_Attack
    #
    # Internally we use Future_Attack for both.
    # --------------------------------------------------------

    if "Actual" in df.columns:
        df = df.rename(
            columns={
                "Actual": "Future_Attack"
            }
        )

    required = [
        "Future_Attack",
        "Probability",
        "Prediction",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path} is missing columns: {missing}"
        )

    # Convert to numeric types
    df["Future_Attack"] = (
        pd.to_numeric(
            df["Future_Attack"],
            errors="raise"
        ).astype(int)
    )

    df["Probability"] = (
        pd.to_numeric(
            df["Probability"],
            errors="raise"
        ).astype(float)
    )

    df["Prediction"] = (
        pd.to_numeric(
            df["Prediction"],
            errors="raise"
        ).astype(int)
    )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    if not df["Future_Attack"].isin([0, 1]).all():
        raise ValueError(
            f"{path} contains invalid target values."
        )

    if not np.isfinite(
        df["Probability"].values
    ).all():
        raise ValueError(
            f"{path} contains NaN/inf probabilities."
        )

    if (
        (df["Probability"] < 0)
        | (df["Probability"] > 1)
    ).any():
        raise ValueError(
            f"{path} contains probabilities outside [0, 1]."
        )

    print(f"  Rows: {len(df)}")
    print("  ✓ Required columns verified")
    print("  ✓ Target values verified")
    print("  ✓ Probabilities verified")

    return df


# ============================================================
# BASIC METRICS
# ============================================================

def calculate_metrics(df, threshold=0.50):

    y_true = df[
        "Future_Attack"
    ].astype(int).values

    probabilities = df[
        "Probability"
    ].astype(float).values

    predictions = (
        probabilities >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    return {
        "Threshold": threshold,

        "Accuracy": accuracy_score(
            y_true,
            predictions
        ),

        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "F1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "ROC_AUC": roc_auc_score(
            y_true,
            probabilities
        ),

        "PR_AUC": average_precision_score(
            y_true,
            probabilities
        ),

        "TN": cm[0, 0],
        "FP": cm[0, 1],
        "FN": cm[1, 0],
        "TP": cm[1, 1],
    }


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

def threshold_analysis(
    df,
    model_name
):

    y_true = df[
        "Future_Attack"
    ].astype(int).values

    probabilities = df[
        "Probability"
    ].astype(float).values

    # Diagnostic thresholds only.
    thresholds = np.arange(
        0.01,
        1.00,
        0.01
    )

    rows = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        cm = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1]
        )

        rows.append({
            "Model": model_name,

            "Threshold": threshold,

            "Accuracy": accuracy_score(
                y_true,
                predictions
            ),

            "Precision": precision_score(
                y_true,
                predictions,
                zero_division=0
            ),

            "Recall": recall_score(
                y_true,
                predictions,
                zero_division=0
            ),

            "F1": f1_score(
                y_true,
                predictions,
                zero_division=0
            ),

            "TN": cm[0, 0],
            "FP": cm[0, 1],
            "FN": cm[1, 0],
            "TP": cm[1, 1],

            "Attack_Predictions": int(
                predictions.sum()
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# PROBABILITY DISTRIBUTION
# ============================================================

def probability_summary(
    df,
    model_name
):

    y_true = df[
        "Future_Attack"
    ].astype(int).values

    probabilities = df[
        "Probability"
    ].astype(float).values

    benign = probabilities[
        y_true == 0
    ]

    attack = probabilities[
        y_true == 1
    ]

    rows = []

    for label, values in [
        ("Benign", benign),
        ("Infilteration", attack),
    ]:

        rows.append({
            "Model": model_name,
            "Class": label,
            "Count": len(values),

            "Mean": values.mean(),
            "Std": values.std(),
            "Min": values.min(),

            "25th_Percentile": np.percentile(
                values,
                25
            ),

            "Median": np.median(
                values
            ),

            "75th_Percentile": np.percentile(
                values,
                75
            ),

            "Max": values.max(),
        })

    return pd.DataFrame(rows)


# ============================================================
# CURVE DATA
# ============================================================

def curve_data(
    df,
    model_name
):

    y_true = df[
        "Future_Attack"
    ].astype(int).values

    probabilities = df[
        "Probability"
    ].astype(float).values

    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    fpr, tpr, roc_thresholds = roc_curve(
        y_true,
        probabilities
    )

    roc_df = pd.DataFrame({
        "Model": model_name,
        "Curve": "ROC",
        "X": fpr,
        "Y": tpr,
        "Threshold": roc_thresholds,
    })

    # --------------------------------------------------------
    # Precision-Recall
    # --------------------------------------------------------

    precision, recall, pr_thresholds = (
        precision_recall_curve(
            y_true,
            probabilities
        )
    )

    # PR curve contains one more point than
    # the number of thresholds.
    pr_thresholds_padded = np.append(
        pr_thresholds,
        np.nan
    )

    pr_df = pd.DataFrame({
        "Model": model_name,
        "Curve": "PR",
        "X": recall,
        "Y": precision,
        "Threshold": pr_thresholds_padded,
    })

    return pd.concat(
        [
            roc_df,
            pr_df,
        ],
        ignore_index=True
    )


# ============================================================
# PRINT SELECTED THRESHOLDS
# ============================================================

def print_selected_thresholds(
    threshold_df,
    model_name
):

    print("\n" + "-" * 80)
    print(
        f"{model_name} THRESHOLD ANALYSIS"
    )
    print("-" * 80)

    selected = threshold_df[
        threshold_df["Threshold"].isin([
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
        ])
    ].copy()

    display_columns = [
        "Threshold",
        "Precision",
        "Recall",
        "F1",
        "FP",
        "FN",
        "TP",
        "Attack_Predictions",
    ]

    print(
        selected[
            display_columns
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Diagnostic best F1
    # --------------------------------------------------------

    best_f1_index = threshold_df[
        "F1"
    ].idxmax()

    best_f1 = threshold_df.loc[
        best_f1_index
    ]

    print(
        "\nHighest diagnostic F1:"
    )

    print(
        f"  Threshold: "
        f"{best_f1['Threshold']:.2f}"
    )

    print(
        f"  F1: "
        f"{best_f1['F1']:.4f}"
    )

    print(
        f"  Precision: "
        f"{best_f1['Precision']:.4f}"
    )

    print(
        f"  Recall: "
        f"{best_f1['Recall']:.4f}"
    )

    print(
        f"  False Positives: "
        f"{int(best_f1['FP'])}"
    )

    print(
        f"  False Negatives: "
        f"{int(best_f1['FN'])}"
    )

    print(
        "\n⚠ This threshold is diagnostic only."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "UNSEEN INFILTERATION PREDICTION ANALYSIS"
    )
    print("=" * 80)

    print("\nComparing:")
    print(
        "  Original XGBoost  = 18 features"
    )
    print(
        "  Enhanced XGBoost  = 161 features"
    )

    print("\nUnseen attack:")
    print("  Infilteration")

    # ========================================================
    # LOAD
    # ========================================================

    original = load_predictions(
        ORIGINAL_PATH,
        "Original XGBoost"
    )

    enhanced = load_predictions(
        ENHANCED_PATH,
        "Enhanced XGBoost"
    )

    # ========================================================
    # DATASET DISTRIBUTIONS
    # ========================================================

    print("\n" + "=" * 80)
    print("DATASET DISTRIBUTIONS")
    print("=" * 80)

    for name, df in [
        ("Original", original),
        ("Enhanced", enhanced),
    ]:

        counts = (
            df["Future_Attack"]
            .value_counts()
            .sort_index()
        )

        print(f"\n{name}:")

        for cls, count in counts.items():

            label = (
                "Benign"
                if cls == 0
                else "Infilteration"
            )

            print(
                f"  {label}: {count}"
            )

    # ========================================================
    # HEADLINE METRICS
    # ========================================================

    print("\n" + "=" * 80)
    print(
        "HEADLINE RESULTS AT THRESHOLD 0.50"
    )
    print("=" * 80)

    original_metrics = calculate_metrics(
        original,
        threshold=0.50
    )

    enhanced_metrics = calculate_metrics(
        enhanced,
        threshold=0.50
    )

    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    print("\nOriginal XGBoost:")

    print(
        f"  Accuracy:  "
        f"{original_metrics['Accuracy']:.4f}"
    )

    print(
        f"  Precision: "
        f"{original_metrics['Precision']:.4f}"
    )

    print(
        f"  Recall:    "
        f"{original_metrics['Recall']:.4f}"
    )

    print(
        f"  F1:        "
        f"{original_metrics['F1']:.4f}"
    )

    print(
        f"  ROC-AUC:   "
        f"{original_metrics['ROC_AUC']:.4f}"
    )

    print(
        f"  PR-AUC:    "
        f"{original_metrics['PR_AUC']:.4f}"
    )

    # --------------------------------------------------------
    # Enhanced
    # --------------------------------------------------------

    print("\nEnhanced XGBoost:")

    print(
        f"  Accuracy:  "
        f"{enhanced_metrics['Accuracy']:.4f}"
    )

    print(
        f"  Precision: "
        f"{enhanced_metrics['Precision']:.4f}"
    )

    print(
        f"  Recall:    "
        f"{enhanced_metrics['Recall']:.4f}"
    )

    print(
        f"  F1:        "
        f"{enhanced_metrics['F1']:.4f}"
    )

    print(
        f"  ROC-AUC:   "
        f"{enhanced_metrics['ROC_AUC']:.4f}"
    )

    print(
        f"  PR-AUC:    "
        f"{enhanced_metrics['PR_AUC']:.4f}"
    )

    # ========================================================
    # CHANGE
    # ========================================================

    print("\n" + "=" * 80)
    print(
        "CHANGE: ENHANCED - ORIGINAL"
    )
    print("=" * 80)

    for metric in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
        "PR_AUC",
    ]:

        change = (
            enhanced_metrics[metric]
            - original_metrics[metric]
        )

        print(
            f"{metric:<10}: "
            f"{change:+.4f}"
        )

    # ========================================================
    # THRESHOLD ANALYSIS
    # ========================================================

    original_thresholds = (
        threshold_analysis(
            original,
            "Original XGBoost"
        )
    )

    enhanced_thresholds = (
        threshold_analysis(
            enhanced,
            "Enhanced XGBoost"
        )
    )

    all_thresholds = pd.concat(
        [
            original_thresholds,
            enhanced_thresholds,
        ],
        ignore_index=True
    )

    all_thresholds.to_csv(
        THRESHOLD_ANALYSIS_PATH,
        index=False
    )

    print_selected_thresholds(
        original_thresholds,
        "Original XGBoost"
    )

    print_selected_thresholds(
        enhanced_thresholds,
        "Enhanced XGBoost"
    )

    # ========================================================
    # PROBABILITY DISTRIBUTION
    # ========================================================

    print("\n" + "=" * 80)
    print("PROBABILITY DISTRIBUTION")
    print("=" * 80)

    original_probability = (
        probability_summary(
            original,
            "Original XGBoost"
        )
    )

    enhanced_probability = (
        probability_summary(
            enhanced,
            "Enhanced XGBoost"
        )
    )

    probability_df = pd.concat(
        [
            original_probability,
            enhanced_probability,
        ],
        ignore_index=True
    )

    print(
        probability_df.to_string(
            index=False
        )
    )

    probability_df.to_csv(
        PROBABILITY_SUMMARY_PATH,
        index=False
    )

    # ========================================================
    # CURVE DATA
    # ========================================================

    original_curves = curve_data(
        original,
        "Original XGBoost"
    )

    enhanced_curves = curve_data(
        enhanced,
        "Enhanced XGBoost"
    )

    curves = pd.concat(
        [
            original_curves,
            enhanced_curves,
        ],
        ignore_index=True
    )

    curves.to_csv(
        CURVE_PATH,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = pd.DataFrame([
        {
            "Model": "Original XGBoost",
            "Features": 18,
            **original_metrics,
        },
        {
            "Model": "Enhanced XGBoost",
            "Features": 161,
            **enhanced_metrics,
        },
    ])

    summary.to_csv(
        SUMMARY_PATH,
        index=False
    )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    if (
        enhanced_metrics["PR_AUC"]
        > original_metrics["PR_AUC"]
    ):
        print(
            "\n✓ Enhanced model has higher PR-AUC."
        )
        print(
            "  This suggests improved ranking of"
        )
        print(
            "  unseen Infilteration samples."
        )
    elif (
        enhanced_metrics["PR_AUC"]
        < original_metrics["PR_AUC"]
    ):
        print(
            "\n✗ Enhanced model has lower PR-AUC."
        )
        print(
            "  Temporal features did not improve"
        )
        print(
            "  unseen Infilteration ranking."
        )
    else:
        print(
            "\n= Enhanced and original PR-AUC are equal."
        )

    if (
        enhanced_metrics["ROC_AUC"]
        > original_metrics["ROC_AUC"]
    ):
        print(
            "✓ Enhanced model has higher ROC-AUC."
        )
    elif (
        enhanced_metrics["ROC_AUC"]
        < original_metrics["ROC_AUC"]
    ):
        print(
            "✗ Enhanced model has lower ROC-AUC."
        )
    else:
        print(
            "= Enhanced and original ROC-AUC are equal."
        )

    if (
        enhanced_metrics["F1"]
        > original_metrics["F1"]
    ):
        print(
            "✓ Enhanced model has higher F1 at 0.50."
        )
    elif (
        enhanced_metrics["F1"]
        < original_metrics["F1"]
    ):
        print(
            "✗ Enhanced model has lower F1 at 0.50."
        )
    else:
        print(
            "= Enhanced and original F1 are equal."
        )

    print(
        "\nImportant conclusion:"
    )

    print(
        "Threshold curves are diagnostic only."
    )

    print(
        "The unseen Infilteration set must NOT be"
    )

    print(
        "used to choose a deployment threshold."
    )

    # ========================================================
    # FILES
    # ========================================================

    print("\n" + "=" * 80)
    print("FILES SAVED")
    print("=" * 80)

    print(
        f"\n  {THRESHOLD_ANALYSIS_PATH}"
    )

    print(
        f"  {SUMMARY_PATH}"
    )

    print(
        f"  {CURVE_PATH}"
    )

    print(
        f"  {PROBABILITY_SUMMARY_PATH}"
    )

    print(
        f"  {CURVE_PATH}"
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()