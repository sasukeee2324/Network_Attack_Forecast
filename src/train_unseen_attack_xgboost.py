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

DATA_DIR = Path("data/processed/unseen_attack")
MODEL_DIR = Path("models")

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

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

    return (
        X_train,
        y_train,
        X_unseen,
        y_unseen,
    )


# ============================================================
# FLATTEN TEMPORAL SEQUENCES
# ============================================================

def flatten_sequences(X):

    return X.reshape(
        X.shape[0],
        -1
    )


# ============================================================
# TRAINING
# ============================================================

def main():

    print("=" * 80)
    print("UNSEEN ATTACK XGBOOST EVALUATION")
    print("=" * 80)

    print(
        "\nExperiment:"
    )

    print(
        "Train without Infilteration"
    )

    print(
        "Test on completely unseen Infilteration"
    )

    print(
        f"\nXGBoost version: "
        f"{xgb.__version__}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_unseen,
        y_unseen,
    ) = load_data()

    print("\nOriginal shapes:")

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

    # --------------------------------------------------------
    # Flatten
    # --------------------------------------------------------

    X_train = flatten_sequences(
        X_train
    )

    X_unseen = flatten_sequences(
        X_unseen
    )

    print("\nFlattened shapes:")

    print(
        f"X_train:  {X_train.shape}"
    )

    print(
        f"X_unseen: {X_unseen.shape}"
    )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    negative = np.sum(
        y_train == 0
    )

    positive = np.sum(
        y_train == 1
    )

    if positive == 0:

        raise ValueError(
            "Training data contains no attack samples."
        )

    scale_pos_weight = (
        negative / positive
    )

    print("\nTraining distribution:")

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

    print("\nUnseen test distribution:")

    print(
        f"No attack: "
        f"{np.sum(y_unseen == 0)}"
    )

    print(
        f"Attack:    "
        f"{np.sum(y_unseen == 1)}"
    )

    # --------------------------------------------------------
    # Model
    #
    # No validation set is used here.
    #
    # This experiment is specifically measuring whether the
    # model can generalize to a completely unseen attack type.
    # --------------------------------------------------------

    model = xgb.XGBClassifier(

        n_estimators=300,

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
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print(
        "\nTraining XGBoost on known attacks..."
    )

    model.fit(
        X_train,
        y_train,
        verbose=False
    )

    print(
        "Training complete."
    )

    # --------------------------------------------------------
    # Predict unseen attack
    # --------------------------------------------------------

    print(
        "\nEvaluating on unseen Infilteration..."
    )

    probabilities = model.predict_proba(
        X_unseen
    )[:, 1]

    threshold = 0.50

    predictions = (
        probabilities >= threshold
    ).astype(int)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_unseen,
        predictions
    )

    precision = precision_score(
        y_unseen,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_unseen,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_unseen,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_unseen,
        probabilities
    )

    pr_auc = average_precision_score(
        y_unseen,
        probabilities
    )

    cm = confusion_matrix(
        y_unseen,
        predictions
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("UNSEEN INFILTERATION RESULTS")
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
            y_unseen,
            predictions,
            target_names=[
                "No Attack",
                "Attack"
            ],
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Probability distribution
    # --------------------------------------------------------

    print("\nPrediction probability summary:")

    print(
        pd.Series(
            probabilities
        ).describe().to_string()
    )

    # --------------------------------------------------------
    # Save directory
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
        / "xgboost_unseen_infilteration.json"
    )

    model.save_model(
        model_path
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics = pd.DataFrame([{

        "Experiment":
            "Unseen Infilteration",

        "Accuracy":
            accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1":
            f1,

        "ROC_AUC":
            roc_auc,

        "PR_AUC":
            pr_auc,

        "Threshold":
            threshold,

    }])

    metrics_path = (
        MODEL_DIR
        / "xgboost_unseen_infilteration_metrics.csv"
    )

    metrics.to_csv(
        metrics_path,
        index=False
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    predictions_df = pd.DataFrame({

        "Actual":
            y_unseen,

        "Probability":
            probabilities,

        "Prediction":
            predictions,

        "Threshold":
            threshold,

    })

    predictions_path = (
        MODEL_DIR
        / "xgboost_unseen_infilteration_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
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

    importance_path = (
        MODEL_DIR
        / "xgboost_unseen_infilteration_feature_importance.csv"
    )

    importance_df.to_csv(
        importance_path,
        index=False
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("UNSEEN ATTACK EXPERIMENT COMPLETE")
    print("=" * 80)

    print(
        "\nThe model was NOT trained on:"
    )

    print(
        "  Infilteration"
    )

    print(
        "\nFinal unseen-attack results:"
    )

    print(
        f"  F1:      {f1:.4f}"
    )

    print(
        f"  Recall:  {recall:.4f}"
    )

    print(
        f"  ROC-AUC: {roc_auc:.4f}"
    )

    print(
        f"  PR-AUC:  {pr_auc:.4f}"
    )

    print("\nSaved:")

    print(
        model_path
    )

    print(
        metrics_path
    )

    print(
        predictions_path
    )

    print(
        importance_path
    )


if __name__ == "__main__":
    main()