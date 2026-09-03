from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path(
    "data/processed/enhanced_training"
)

MODEL_DIR = Path(
    "models"
)

X_TRAIN_FILE = DATA_DIR / "X_train.npy"
X_VALIDATION_FILE = DATA_DIR / "X_validation.npy"
X_TEST_FILE = DATA_DIR / "X_test.npy"

Y_TRAIN_FILE = DATA_DIR / "y_train.npy"
Y_VALIDATION_FILE = DATA_DIR / "y_validation.npy"
Y_TEST_FILE = DATA_DIR / "y_test.npy"

MODEL_FILE = (
    MODEL_DIR /
    "xgboost_enhanced_forecast15_v2.json"
)

METRICS_FILE = (
    MODEL_DIR /
    "xgboost_enhanced_v2_metrics.csv"
)

THRESHOLD_FILE = (
    MODEL_DIR /
    "xgboost_enhanced_v2_threshold_analysis.csv"
)

VALIDATION_PREDICTIONS_FILE = (
    MODEL_DIR /
    "xgboost_enhanced_v2_validation_predictions.csv"
)

TEST_PREDICTIONS_FILE = (
    MODEL_DIR /
    "xgboost_enhanced_v2_test_predictions.csv"
)

FEATURE_IMPORTANCE_FILE = (
    MODEL_DIR /
    "xgboost_enhanced_v2_feature_importance.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

RANDOM_STATE = 42

N_ESTIMATORS = 500
LEARNING_RATE = 0.03
MAX_DEPTH = 4
MIN_CHILD_WEIGHT = 3
SUBSAMPLE = 0.85
COLSAMPLE_BYTREE = 0.85

EARLY_STOPPING_ROUNDS = 40

THRESHOLDS = np.arange(
    0.20,
    0.81,
    0.05
)


# ============================================================
# HELPERS
# ============================================================

def load_required_file(path):

    if not path.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )

    return np.load(path)


def print_distribution(name, y):

    benign = int(
        np.sum(y == 0)
    )

    attack = int(
        np.sum(y == 1)
    )

    total = len(y)

    print(f"\n{name}:")

    print(
        f"  No attack: {benign}"
        f" ({benign / total * 100:.2f}%)"
    )

    print(
        f"  Attack:    {attack}"
        f" ({attack / total * 100:.2f}%)"
    )


def evaluate_predictions(
    y_true,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities
    )

    cm = confusion_matrix(
        y_true,
        predictions
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "predictions": predictions,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("ENHANCED XGBOOST TEMPORAL FORECASTING MODEL V2")
    print("=" * 80)

    print(
        f"\nXGBoost version: {xgb.__version__}"
    )

    print(
        "\nUsing canonical chronological split:"
    )

    print(
        f"  {DATA_DIR}"
    )

    print(
        "\nNo new split will be created."
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    X_train = load_required_file(
        X_TRAIN_FILE
    )

    X_validation = load_required_file(
        X_VALIDATION_FILE
    )

    X_test = load_required_file(
        X_TEST_FILE
    )

    y_train = load_required_file(
        Y_TRAIN_FILE
    )

    y_validation = load_required_file(
        Y_VALIDATION_FILE
    )

    y_test = load_required_file(
        Y_TEST_FILE
    )

    # ========================================================
    # SHAPES
    # ========================================================

    print(
        "\nOriginal shapes:"
    )

    print(
        f"X_train:       {X_train.shape}"
    )

    print(
        f"X_validation:  {X_validation.shape}"
    )

    print(
        f"X_test:        {X_test.shape}"
    )

    print(
        f"y_train:       {y_train.shape}"
    )

    print(
        f"y_validation:  {y_validation.shape}"
    )

    print(
        f"y_test:        {y_test.shape}"
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    if X_train.ndim != 3:
        raise ValueError(
            "X_train must have shape "
            "(samples, history, features)."
        )

    if X_validation.ndim != 3:
        raise ValueError(
            "X_validation must have shape "
            "(samples, history, features)."
        )

    if X_test.ndim != 3:
        raise ValueError(
            "X_test must have shape "
            "(samples, history, features)."
        )

    if X_train.shape[2] != 161:
        raise ValueError(
            f"Expected 161 features, "
            f"got {X_train.shape[2]}."
        )

    if X_validation.shape[2] != 161:
        raise ValueError(
            "Validation feature count mismatch."
        )

    if X_test.shape[2] != 161:
        raise ValueError(
            "Test feature count mismatch."
        )

    if len(X_train) != len(y_train):
        raise ValueError(
            "Training X/y mismatch."
        )

    if len(X_validation) != len(y_validation):
        raise ValueError(
            "Validation X/y mismatch."
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            "Test X/y mismatch."
        )

    if not np.isfinite(X_train).all():
        raise ValueError(
            "X_train contains NaN/inf."
        )

    if not np.isfinite(X_validation).all():
        raise ValueError(
            "X_validation contains NaN/inf."
        )

    if not np.isfinite(X_test).all():
        raise ValueError(
            "X_test contains NaN/inf."
        )

    print(
        "\nFeature validity check: PASSED"
    )

    # ========================================================
    # TARGET VALIDATION
    # ========================================================

    for name, target in [
        ("y_train", y_train),
        ("y_validation", y_validation),
        ("y_test", y_test),
    ]:

        unique = np.unique(target)

        if not np.all(
            np.isin(unique, [0, 1])
        ):

            raise ValueError(
                f"{name} contains values "
                f"other than 0/1: {unique}"
            )

    if len(np.unique(y_validation)) < 2:

        raise ValueError(
            "Validation target contains "
            "only one class. ROC-AUC and "
            "threshold selection are invalid."
        )

    if len(np.unique(y_test)) < 2:

        raise ValueError(
            "Test target contains only "
            "one class."
        )

    print(
        "Target validity check: PASSED"
    )

    # ========================================================
    # DISTRIBUTIONS
    # ========================================================

    print_distribution(
        "Training distribution",
        y_train
    )

    print_distribution(
        "Validation distribution",
        y_validation
    )

    print_distribution(
        "Test distribution",
        y_test
    )

    # ========================================================
    # FLATTEN
    # ========================================================

    X_train_flat = X_train.reshape(
        X_train.shape[0],
        -1
    )

    X_validation_flat = X_validation.reshape(
        X_validation.shape[0],
        -1
    )

    X_test_flat = X_test.reshape(
        X_test.shape[0],
        -1
    )

    print(
        "\nFlattened shapes:"
    )

    print(
        f"X_train:       {X_train_flat.shape}"
    )

    print(
        f"X_validation:  {X_validation_flat.shape}"
    )

    print(
        f"X_test:        {X_test_flat.shape}"
    )

    # ========================================================
    # CLASS WEIGHT
    # ========================================================

    negative_count = np.sum(
        y_train == 0
    )

    positive_count = np.sum(
        y_train == 1
    )

    if positive_count == 0:

        raise ValueError(
            "No positive samples in training."
        )

    scale_pos_weight = (
        negative_count
        / positive_count
    )

    print(
        "\nTraining class distribution:"
    )

    print(
        f"No attack: {negative_count}"
    )

    print(
        f"Attack:    {positive_count}"
    )

    print(
        f"scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = xgb.XGBClassifier(

        n_estimators=N_ESTIMATORS,

        learning_rate=LEARNING_RATE,

        max_depth=MAX_DEPTH,

        min_child_weight=MIN_CHILD_WEIGHT,

        subsample=SUBSAMPLE,

        colsample_bytree=COLSAMPLE_BYTREE,

        objective="binary:logistic",

        eval_metric="logloss",

        scale_pos_weight=scale_pos_weight,

        random_state=RANDOM_STATE,

        tree_method="hist",

        n_jobs=-1,
    )

    # ========================================================
    # TRAIN
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "TRAINING ENHANCED XGBOOST"
    )

    print(
        "=" * 80
    )

    model.fit(
        X_train_flat,
        y_train,

        eval_set=[
            (
                X_validation_flat,
                y_validation
            )
        ],

        verbose=False
    )

    print(
        "\nTraining complete."
    )

    # ========================================================
    # BEST ITERATION
    # ========================================================

    best_iteration = getattr(
        model,
        "best_iteration",
        None
    )

    if best_iteration is not None:

        print(
            f"Best boosting iteration: "
            f"{best_iteration}"
        )

    # ========================================================
    # VALIDATION PROBABILITIES
    # ========================================================

    validation_probabilities = (
        model.predict_proba(
            X_validation_flat
        )[:, 1]
    )

    # ========================================================
    # THRESHOLD SEARCH
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "VALIDATION THRESHOLD SEARCH"
    )

    print(
        "=" * 80
    )

    threshold_rows = []

    best_threshold = None
    best_f1 = -1

    for threshold in THRESHOLDS:

        result = evaluate_predictions(
            y_validation,
            validation_probabilities,
            threshold
        )

        row = {
            "Threshold": threshold,
            "Precision": result["precision"],
            "Recall": result["recall"],
            "F1": result["f1"],
        }

        threshold_rows.append(
            row
        )

        if result["f1"] > best_f1:

            best_f1 = result["f1"]

            best_threshold = (
                threshold
            )

    threshold_df = pd.DataFrame(
        threshold_rows
    )

    print(
        f"\nSelected validation threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        "\nThreshold comparison:"
    )

    print(
        threshold_df.to_string(
            index=False,
            formatters={
                "Threshold": "{:.2f}".format,
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1": "{:.4f}".format,
            }
        )
    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    validation_result = (
        evaluate_predictions(
            y_validation,
            validation_probabilities,
            best_threshold
        )
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "VALIDATION RESULTS"
    )

    print(
        "=" * 80
    )

    print(
        f"\nDecision threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Accuracy:  "
        f"{validation_result['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{validation_result['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{validation_result['recall']:.4f}"
    )

    print(
        f"F1 Score:  "
        f"{validation_result['f1']:.4f}"
    )

    print(
        f"ROC-AUC:   "
        f"{validation_result['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC:    "
        f"{validation_result['pr_auc']:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        validation_result[
            "confusion_matrix"
        ]
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_validation,
            validation_result[
                "predictions"
            ],
            target_names=[
                "No Attack",
                "Attack"
            ],
            zero_division=0
        )
    )

    # ========================================================
    # TEST
    # ========================================================

    test_probabilities = (
        model.predict_proba(
            X_test_flat
        )[:, 1]
    )

    test_result = evaluate_predictions(
        y_test,
        test_probabilities,
        best_threshold
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "TEST RESULTS"
    )

    print(
        "=" * 80
    )

    print(
        f"\nDecision threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Accuracy:  "
        f"{test_result['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{test_result['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{test_result['recall']:.4f}"
    )

    print(
        f"F1 Score:  "
        f"{test_result['f1']:.4f}"
    )

    print(
        f"ROC-AUC:   "
        f"{test_result['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC:    "
        f"{test_result['pr_auc']:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        test_result[
            "confusion_matrix"
        ]
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            test_result[
                "predictions"
            ],
            target_names=[
                "No Attack",
                "Attack"
            ],
            zero_division=0
        )
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model.save_model(
        MODEL_FILE
    )

    # ========================================================
    # SAVE THRESHOLD ANALYSIS
    # ========================================================

    threshold_df.to_csv(
        THRESHOLD_FILE,
        index=False
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    validation_predictions = pd.DataFrame(
        {
            "y_true": y_validation,
            "probability": validation_probabilities,
            "prediction": validation_result[
                "predictions"
            ],
            "threshold": best_threshold,
        }
    )

    validation_predictions.to_csv(
        VALIDATION_PREDICTIONS_FILE,
        index=False
    )

    test_predictions = pd.DataFrame(
        {
            "y_true": y_test,
            "probability": test_probabilities,
            "prediction": test_result[
                "predictions"
            ],
            "threshold": best_threshold,
        }
    )

    test_predictions.to_csv(
        TEST_PREDICTIONS_FILE,
        index=False
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    importance = model.feature_importances_

    feature_names = []

    for timestep in range(
        X_train.shape[1]
    ):

        for feature in range(
            X_train.shape[2]
        ):

            feature_names.append(
                f"timestep_{timestep + 1}"
                f"_feature_{feature + 1}"
            )

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importance,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    importance_df.to_csv(
        FEATURE_IMPORTANCE_FILE,
        index=False
    )

    # ========================================================
    # METRICS
    # ========================================================

    metrics_df = pd.DataFrame(
        [
            {
                "Model": "Enhanced XGBoost V2",
                "Features": X_train.shape[2],
                "History_Minutes": X_train.shape[1],
                "Forecast_Minutes": 15,
                "Threshold": best_threshold,

                "Validation_Accuracy":
                    validation_result["accuracy"],

                "Validation_Precision":
                    validation_result["precision"],

                "Validation_Recall":
                    validation_result["recall"],

                "Validation_F1":
                    validation_result["f1"],

                "Validation_ROC_AUC":
                    validation_result["roc_auc"],

                "Validation_PR_AUC":
                    validation_result["pr_auc"],

                "Test_Accuracy":
                    test_result["accuracy"],

                "Test_Precision":
                    test_result["precision"],

                "Test_Recall":
                    test_result["recall"],

                "Test_F1":
                    test_result["f1"],

                "Test_ROC_AUC":
                    test_result["roc_auc"],

                "Test_PR_AUC":
                    test_result["pr_auc"],

                "Best_Iteration":
                    best_iteration,
            }
        ]
    )

    metrics_df.to_csv(
        METRICS_FILE,
        index=False
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "ENHANCED XGBOOST V2 COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"\nFeatures: "
        f"{X_train.shape[2]}"
    )

    print(
        f"History: "
        f"{X_train.shape[1]} minutes"
    )

    print(
        "Forecast: 15 minutes"
    )

    print(
        f"\nSelected threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Validation F1: "
        f"{validation_result['f1']:.4f}"
    )

    print(
        f"Validation ROC-AUC: "
        f"{validation_result['roc_auc']:.4f}"
    )

    print(
        f"Test F1: "
        f"{test_result['f1']:.4f}"
    )

    print(
        f"Test ROC-AUC: "
        f"{test_result['roc_auc']:.4f}"
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {MODEL_FILE}"
    )

    print(
        f"  {METRICS_FILE}"
    )

    print(
        f"  {THRESHOLD_FILE}"
    )

    print(
        f"  {VALIDATION_PREDICTIONS_FILE}"
    )

    print(
        f"  {TEST_PREDICTIONS_FILE}"
    )

    print(
        f"  {FEATURE_IMPORTANCE_FILE}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "  Canonical chronological split was used."
    )

    print(
        "  Validation was used for threshold selection."
    )

    print(
        "  Test data was not used for threshold selection."
    )

    print(
        "  No attack-label-derived features were used."
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()