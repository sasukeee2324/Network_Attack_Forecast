"""
Train XGBoost on known attacks and evaluate on
completely unseen Infilteration using 161 enhanced features.

Training:
    Bot
    FTP-BruteForce
    SSH-Bruteforce

Unseen test:
    Infilteration

No Infilteration samples are used during training.
"""

import os
import json
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
# CONFIG
# ============================================================

DATA_DIR = "data/processed/enhanced_unseen_attack"
MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "xgboost_enhanced_unseen_infilteration.json"
)

METRICS_PATH = os.path.join(
    MODEL_DIR,
    "xgboost_enhanced_unseen_infilteration_metrics.csv"
)

PREDICTIONS_PATH = os.path.join(
    MODEL_DIR,
    "xgboost_enhanced_unseen_infilteration_predictions.csv"
)

FEATURE_IMPORTANCE_PATH = os.path.join(
    MODEL_DIR,
    "xgboost_enhanced_unseen_infilteration_feature_importance.csv"
)

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def main():

    print("=" * 80)
    print("ENHANCED XGBOOST UNSEEN ATTACK EXPERIMENT")
    print("=" * 80)

    print("\nLoading dataset...")

    X_train = np.load(
        os.path.join(DATA_DIR, "X_train.npy")
    )

    y_train = np.load(
        os.path.join(DATA_DIR, "y_train.npy")
    )

    X_unseen = np.load(
        os.path.join(DATA_DIR, "X_unseen.npy")
    )

    y_unseen = np.load(
        os.path.join(DATA_DIR, "y_unseen.npy")
    )

    metadata_train = pd.read_csv(
        os.path.join(DATA_DIR, "metadata_train.csv")
    )

    metadata_unseen = pd.read_csv(
        os.path.join(DATA_DIR, "metadata_unseen.csv")
    )

    feature_names = pd.read_csv(
        os.path.join(DATA_DIR, "feature_names.csv")
    )["Feature"].tolist()

    # --------------------------------------------------------
    # Basic checks
    # --------------------------------------------------------

    print("\nDataset shapes:")

    print(f"X_train:  {X_train.shape}")
    print(f"y_train:  {y_train.shape}")
    print(f"X_unseen: {X_unseen.shape}")
    print(f"y_unseen: {y_unseen.shape}")

    assert X_train.ndim == 3
    assert X_unseen.ndim == 3

    assert X_train.shape[1] == 10
    assert X_unseen.shape[1] == 10

    assert X_train.shape[2] == 161
    assert X_unseen.shape[2] == 161

    assert len(feature_names) == 161

    assert np.isfinite(X_train).all()
    assert np.isfinite(X_unseen).all()

    print("\n✓ Dataset dimensions verified")
    print("✓ 161 enhanced features verified")
    print("✓ 10-minute history verified")
    print("✓ All values finite")

    # ========================================================
    # FLATTEN SEQUENCES
    # ========================================================

    print("\n" + "=" * 80)
    print("PREPARING XGBOOST INPUT")
    print("=" * 80)

    X_train_flat = X_train.reshape(
        X_train.shape[0],
        -1
    )

    X_unseen_flat = X_unseen.reshape(
        X_unseen.shape[0],
        -1
    )

    print(
        f"\nFlattened training shape: "
        f"{X_train_flat.shape}"
    )

    print(
        f"Flattened unseen shape: "
        f"{X_unseen_flat.shape}"
    )

    expected_features = 10 * 161

    assert X_train_flat.shape[1] == expected_features
    assert X_unseen_flat.shape[1] == expected_features

    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    print("\nTraining classes:")

    train_counts = np.bincount(y_train)

    print(
        f"  Benign: {train_counts[0]}"
    )

    print(
        f"  Attack: {train_counts[1]}"
    )

    print("\nUnseen test classes:")

    unseen_counts = np.bincount(y_unseen)

    print(
        f"  Benign: {unseen_counts[0]}"
    )

    print(
        f"  Infilteration: {unseen_counts[1]}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    print("\n" + "=" * 80)
    print("TRAINING XGBOOST")
    print("=" * 80)

    # Compute scale_pos_weight from training only
    scale_pos_weight = (
        train_counts[0] / train_counts[1]
    )

    print(
        f"\nscale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.05,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )

    print("\nStarting training...")

    model.fit(
        X_train_flat,
        y_train,
        verbose=False,
    )

    print("✓ Training complete")

    # ========================================================
    # PREDICTION
    # ========================================================

    print("\n" + "=" * 80)
    print("EVALUATING UNSEEN INFILTERATION")
    print("=" * 80)

    probabilities = model.predict_proba(
        X_unseen_flat
    )[:, 1]

    threshold = 0.50

    predictions = (
        probabilities >= threshold
    ).astype(int)

    # ========================================================
    # METRICS
    # ========================================================

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

    # ========================================================
    # RESULTS
    # ========================================================

    print("\nRESULTS")
    print("-" * 80)

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
                "Benign",
                "Infilteration"
            ],
            zero_division=0
        )
    )

    # ========================================================
    # PROBABILITY ANALYSIS
    # ========================================================

    print("\n" + "=" * 80)
    print("PROBABILITY ANALYSIS")
    print("=" * 80)

    benign_probs = probabilities[
        y_unseen == 0
    ]

    attack_probs = probabilities[
        y_unseen == 1
    ]

    print("\nBenign probability:")
    print(
        f"  Mean: {benign_probs.mean():.6f}"
    )
    print(
        f"  Std:  {benign_probs.std():.6f}"
    )
    print(
        f"  Min:  {benign_probs.min():.6f}"
    )
    print(
        f"  Max:  {benign_probs.max():.6f}"
    )

    print("\nInfilteration probability:")
    print(
        f"  Mean: {attack_probs.mean():.6f}"
    )
    print(
        f"  Std:  {attack_probs.std():.6f}"
    )
    print(
        f"  Min:  {attack_probs.min():.6f}"
    )
    print(
        f"  Max:  {attack_probs.max():.6f}"
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    os.makedirs(MODEL_DIR, exist_ok=True)

    model.save_model(MODEL_PATH)

    print(
        f"\n✓ Model saved: {MODEL_PATH}"
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics = pd.DataFrame([{
        "Model": "Enhanced XGBoost",
        "Features": 161,
        "History_Minutes": 10,
        "Forecast_Minutes": 15,
        "Unseen_Attack": "Infilteration",
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "TN": cm[0, 0],
        "FP": cm[0, 1],
        "FN": cm[1, 0],
        "TP": cm[1, 1],
    }])

    metrics.to_csv(
        METRICS_PATH,
        index=False
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    predictions_df = metadata_unseen.copy()

    predictions_df["Probability"] = probabilities
    predictions_df["Prediction"] = predictions

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    print("\nCalculating feature importance...")

    importances = model.feature_importances_

    importance_rows = []

    for timestep in range(10):

        start = timestep * 161
        end = start + 161

        for feature_idx, feature_name in enumerate(
            feature_names
        ):

            importance_rows.append({
                "Timestep": timestep + 1,
                "Feature": feature_name,
                "Importance": importances[
                    start + feature_idx
                ]
            })

    importance_df = pd.DataFrame(
        importance_rows
    )

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    importance_df.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False
    )

    print(
        f"✓ Feature importance saved: "
        f"{FEATURE_IMPORTANCE_PATH}"
    )

    # ========================================================
    # TOP FEATURES
    # ========================================================

    print("\nTop 20 flattened features:")

    print(
        importance_df.head(20).to_string(
            index=False
        )
    )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    print("\nThis experiment used:")
    print("  ✓ 161 enhanced features")
    print("  ✓ 10-minute history")
    print("  ✓ 15-minute forecast horizon")
    print("  ✓ Bot")
    print("  ✓ FTP-BruteForce")
    print("  ✓ SSH-Bruteforce")

    print("\nCompletely unseen:")
    print("  🔬 Infilteration")

    print("\nImportant:")
    print(
        "Do not tune the threshold using the unseen "
        "Infilteration test set."
    )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()