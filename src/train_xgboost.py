from pathlib import Path

import numpy as np
import pandas as pd

from xgboost import XGBClassifier

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
# LOAD TRAINING DATA
# ============================================================

def load_data():

    X_train = np.load(
        DATA_DIR / "X_train.npy"
    )

    X_validation = np.load(
        DATA_DIR / "X_validation.npy"
    )

    X_test = np.load(
        DATA_DIR / "X_test.npy"
    )

    y_train = np.load(
        DATA_DIR / "y_train.npy"
    )

    y_validation = np.load(
        DATA_DIR / "y_validation.npy"
    )

    y_test = np.load(
        DATA_DIR / "y_test.npy"
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )


# ============================================================
# FLATTEN TEMPORAL SEQUENCES
# ============================================================

def flatten_sequences(X):

    """
    Convert:

        (samples, 10 minutes, 18 features)

    into:

        (samples, 180 features)

    XGBoost receives each complete 10-minute
    history as one feature vector.
    """

    return X.reshape(
        X.shape[0],
        -1
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    name,
    model,
    X,
    y
):

    probabilities = model.predict_proba(
        X
    )[:, 1]

    predictions = (
        probabilities >= 0.5
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
        f"\nAccuracy:  {accuracy:.4f}"
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

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("XGBOOST BASELINE")
    print("=" * 80)

    # --------------------------------------------------------
    # Load data
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

    print(
        f"y_train:       {y_train.shape}"
    )

    print(
        f"y_validation:  {y_validation.shape}"
    )

    print(
        f"y_test:        {y_test.shape}"
    )

    # --------------------------------------------------------
    # Flatten sequences
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
    # Class imbalance
    # --------------------------------------------------------

    negative = np.sum(
        y_train == 0
    )

    positive = np.sum(
        y_train == 1
    )

    if positive == 0:

        raise ValueError(
            "Training set contains no positive samples."
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
    # Create XGBoost classifier
    # --------------------------------------------------------

    model = XGBClassifier(

        n_estimators=300,

        max_depth=5,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="binary:logistic",

        eval_metric="logloss",

        scale_pos_weight=scale_pos_weight,

        random_state=RANDOM_STATE,

        n_jobs=-1,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nTraining XGBoost...")

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
    # Validation
    # --------------------------------------------------------

    validation_results = evaluate_model(
        "VALIDATION RESULTS",
        model,
        X_validation,
        y_validation
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_results = evaluate_model(
        "TEST RESULTS",
        model,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Create model directory
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR
        / "xgboost_forecast15.json"
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
        / "xgboost_metrics.csv"
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
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("XGBOOST BASELINE COMPLETE")
    print("=" * 80)

    print(
        f"\nTest F1: "
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


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()