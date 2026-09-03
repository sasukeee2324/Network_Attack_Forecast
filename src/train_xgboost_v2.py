from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("data/processed/training")
MODEL_DIR = Path("models")

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    X_train = np.load(DATA_DIR / "X_train.npy")
    X_validation = np.load(DATA_DIR / "X_validation.npy")
    X_test = np.load(DATA_DIR / "X_test.npy")

    y_train = np.load(DATA_DIR / "y_train.npy")
    y_validation = np.load(DATA_DIR / "y_validation.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )


# ============================================================
# FLATTEN SEQUENCES
# ============================================================

def flatten_sequences(X):

    return X.reshape(
        X.shape[0],
        -1
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

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
    }


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def find_best_threshold(
    y_true,
    probabilities
):

    thresholds = np.arange(
        0.20,
        0.81,
        0.05
    )

    results = []

    for threshold in thresholds:

        results.append(
            calculate_metrics(
                y_true,
                probabilities,
                threshold
            )
        )

    results_df = pd.DataFrame(
        results
    )

    best_index = results_df[
        "F1"
    ].idxmax()

    best_threshold = float(
        results_df.loc[
            best_index,
            "Threshold"
        ]
    )

    return (
        best_threshold,
        results_df
    )


# ============================================================
# DETAILED EVALUATION
# ============================================================

def evaluate_model(
    name,
    model,
    X,
    y,
    threshold
):

    probabilities = model.predict_proba(
        X
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y,
        probabilities
    )

    pr_auc = average_precision_score(
        y,
        probabilities
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    print(
        f"\nDecision threshold: "
        f"{threshold:.2f}"
    )

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1 Score:  {f1:.4f}"
    )

    print(
        f"ROC-AUC:   {roc_auc:.4f}"
    )

    print(
        f"PR-AUC:    {pr_auc:.4f}"
    )

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            y,
            predictions,
            target_names=[
                "No Attack",
                "Attack"
            ],
            zero_division=0
        )
    )

    results = {
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
    }

    return (
        results,
        probabilities,
        predictions
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("XGBOOST BASELINE V2")
    print("=" * 80)

    print(
        f"\nXGBoost version: "
        f"{xgb.__version__}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = load_data()

    print("\nOriginal shapes:")

    print(
        f"X_train:       {X_train.shape}"
    )

    print(
        f"X_validation:  {X_validation.shape}"
    )

    print(
        f"X_test:        {X_test.shape}"
    )

    # --------------------------------------------------------
    # Flatten
    # --------------------------------------------------------

    X_train = flatten_sequences(
        X_train
    )

    X_validation = flatten_sequences(
        X_validation
    )

    X_test = flatten_sequences(
        X_test
    )

    print("\nFlattened shapes:")

    print(
        f"X_train:       {X_train.shape}"
    )

    print(
        f"X_validation:  {X_validation.shape}"
    )

    print(
        f"X_test:        {X_test.shape}"
    )

    # --------------------------------------------------------
    # Class balance
    # --------------------------------------------------------

    negative = np.sum(
        y_train == 0
    )

    positive = np.sum(
        y_train == 1
    )

    if positive == 0:

        raise ValueError(
            "Training set contains no attack samples."
        )

    scale_pos_weight = (
        negative / positive
    )

    print("\nTraining class distribution:")

    print(
        f"No attack: {negative}"
    )

    print(
        f"Attack:    {positive}"
    )

    print(
        f"scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    # --------------------------------------------------------
    # XGBoost V2
    # --------------------------------------------------------

    early_stop = xgb.callback.EarlyStopping(
        rounds=75,
        save_best=True,
        maximize=False
    )

    model = xgb.XGBClassifier(

        n_estimators=1000,

        max_depth=4,

        learning_rate=0.03,

        min_child_weight=2,

        subsample=0.8,

        colsample_bytree=0.8,

        gamma=0.1,

        reg_alpha=0.1,

        reg_lambda=1.0,

        objective="binary:logistic",

        eval_metric="logloss",

        scale_pos_weight=scale_pos_weight,

        random_state=RANDOM_STATE,

        n_jobs=-1,

        tree_method="hist",

        callbacks=[
            early_stop
        ],
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print(
        "\nTraining XGBoost V2..."
    )

    model.fit(

        X_train,

        y_train,

        eval_set=[
            (
                X_validation,
                y_validation
            )
        ],

        verbose=False,
    )

    print(
        "Training complete."
    )

    # --------------------------------------------------------
    # Best iteration
    # --------------------------------------------------------

    print(
        f"\nBest boosting iteration: "
        f"{model.best_iteration}"
    )

    print(
        f"Best validation score: "
        f"{model.best_score:.6f}"
    )

    # --------------------------------------------------------
    # Validation probabilities
    # --------------------------------------------------------

    validation_probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    # --------------------------------------------------------
    # Threshold selection
    #
    # ONLY validation is used here.
    # --------------------------------------------------------

    print(
        "\nSearching decision thresholds..."
    )

    (
        best_threshold,
        threshold_results
    ) = find_best_threshold(
        y_validation,
        validation_probabilities
    )

    print(
        f"\nBest validation threshold: "
        f"{best_threshold:.2f}"
    )

    print("\nThreshold comparison:")

    print(
        threshold_results[
            [
                "Threshold",
                "Precision",
                "Recall",
                "F1"
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save threshold analysis
    # --------------------------------------------------------

    threshold_results.to_csv(
        MODEL_DIR
        / "xgboost_v2_threshold_analysis.csv",
        index=False
    )

    # --------------------------------------------------------
    # Validation evaluation
    # --------------------------------------------------------

    (
        validation_results,
        validation_probabilities,
        validation_predictions
    ) = evaluate_model(

        "VALIDATION RESULTS",

        model,

        X_validation,

        y_validation,

        best_threshold
    )

    # --------------------------------------------------------
    # Test evaluation
    #
    # IMPORTANT:
    # Threshold is frozen from validation.
    # --------------------------------------------------------

    (
        test_results,
        test_probabilities,
        test_predictions
    ) = evaluate_model(

        "TEST RESULTS",

        model,

        X_test,

        y_test,

        best_threshold
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR
        / "xgboost_forecast15_v2.json"
    )

    model.save_model(
        model_path
    )

    print(
        f"\nModel saved to:"
    )

    print(
        model_path
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics = pd.DataFrame([
        {
            "Split": "Validation",
            **validation_results
        },
        {
            "Split": "Test",
            **test_results
        },
    ])

    metrics_path = (
        MODEL_DIR
        / "xgboost_v2_metrics.csv"
    )

    metrics.to_csv(
        metrics_path,
        index=False
    )

    print(
        f"\nMetrics saved to:"
    )

    print(
        metrics_path
    )

    # --------------------------------------------------------
    # Save validation predictions
    # --------------------------------------------------------

    validation_predictions_df = pd.DataFrame({

        "Actual":
            y_validation,

        "Probability":
            validation_probabilities,

        "Prediction":
            validation_predictions,

        "Threshold":
            best_threshold,
    })

    validation_predictions_df.to_csv(

        MODEL_DIR
        / "xgboost_v2_validation_predictions.csv",

        index=False
    )

    # --------------------------------------------------------
    # Save test predictions
    # --------------------------------------------------------

    test_predictions_df = pd.DataFrame({

        "Actual":
            y_test,

        "Probability":
            test_probabilities,

        "Prediction":
            test_predictions,

        "Threshold":
            best_threshold,
    })

    test_predictions_df.to_csv(

        MODEL_DIR
        / "xgboost_v2_test_predictions.csv",

        index=False
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance = (
        model.feature_importances_
    )

    importance_df = pd.DataFrame({

        "Feature_Index":
            np.arange(
                len(importance)
            ),

        "Importance":
            importance,

    }).sort_values(

        "Importance",

        ascending=False
    )

    importance_df.to_csv(

        MODEL_DIR
        / "xgboost_v2_feature_importance.csv",

        index=False
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("XGBOOST V2 COMPLETE")
    print("=" * 80)

    print(
        f"\nSelected threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Best boosting iteration: "
        f"{model.best_iteration}"
    )

    print(
        f"Validation F1: "
        f"{validation_results['F1']:.4f}"
    )

    print(
        f"Test F1: "
        f"{test_results['F1']:.4f}"
    )

    print(
        f"Test ROC-AUC: "
        f"{test_results['ROC_AUC']:.4f}"
    )

    print(
        f"Test PR-AUC: "
        f"{test_results['PR_AUC']:.4f}"
    )

    print("\nSaved files:")

    print(
        "  models/xgboost_forecast15_v2.json"
    )

    print(
        "  models/xgboost_v2_metrics.csv"
    )

    print(
        "  models/xgboost_v2_threshold_analysis.csv"
    )

    print(
        "  models/xgboost_v2_validation_predictions.csv"
    )

    print(
        "  models/xgboost_v2_test_predictions.csv"
    )

    print(
        "  models/xgboost_v2_feature_importance.csv"
    )


if __name__ == "__main__":
    main()