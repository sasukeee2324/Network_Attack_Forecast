from pathlib import Path
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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
# CONFIGURATION
# ============================================================

TRAIN_DIR = Path("data/processed/training")
MODEL_DIR = Path("models")

RANDOM_STATE = 42

BATCH_SIZE = 64
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.25

LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 100
PATIENCE = 12

THRESHOLD = 0.50


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DATASET
# ============================================================

class NetworkDataset(torch.utils.data.Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.float32
        )

    def __len__(self):

        return len(self.y)

    def __getitem__(self, index):

        return (
            self.X[index],
            self.y[index]
        )


# ============================================================
# GRU MODEL
# ============================================================

class AttackForecastGRU(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout
    ):

        super().__init__()

        self.gru = nn.GRU(

            input_size=input_size,

            hidden_size=hidden_size,

            num_layers=num_layers,

            batch_first=True,

            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                hidden_size,
                32
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                32,
                1
            )
        )

    def forward(self, x):

        # ----------------------------------------------------
        # x:
        # (batch, 10 minutes, 18 features)
        # ----------------------------------------------------

        output, hidden = self.gru(x)

        # Last timestep
        last_output = output[:, -1, :]

        last_output = self.dropout(
            last_output
        )

        logits = self.classifier(
            last_output
        )

        return logits.squeeze(1)


# ============================================================
# LOAD DATA
# ============================================================

def load_training_data():

    X_train = np.load(
        TRAIN_DIR / "X_train.npy"
    )

    X_validation = np.load(
        TRAIN_DIR / "X_validation.npy"
    )

    X_test = np.load(
        TRAIN_DIR / "X_test.npy"
    )

    y_train = np.load(
        TRAIN_DIR / "y_train.npy"
    )

    y_validation = np.load(
        TRAIN_DIR / "y_validation.npy"
    )

    y_test = np.load(
        TRAIN_DIR / "y_test.npy"
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
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold=0.50
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {

        "Accuracy":
            accuracy_score(
                y_true,
                predictions
            ),

        "Precision":
            precision_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "F1":
            f1_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "ROC_AUC":
            roc_auc_score(
                y_true,
                probabilities
            ),

        "PR_AUC":
            average_precision_score(
                y_true,
                probabilities
            ),
    }

    return (
        metrics,
        predictions
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    all_probabilities = []
    all_targets = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(
                DEVICE
            )

            y_batch = y_batch.to(
                DEVICE
            )

            logits = model(
                X_batch
            )

            loss = criterion(
                logits,
                y_batch
            )

            total_loss += (
                loss.item()
                * len(y_batch)
            )

            probabilities = torch.sigmoid(
                logits
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

            all_targets.extend(
                y_batch.cpu().numpy()
            )

    average_loss = (
        total_loss
        / len(loader.dataset)
    )

    return (
        average_loss,
        np.asarray(
            all_targets,
            dtype=np.int64
        ),
        np.asarray(
            all_probabilities
        )
    )


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

    rows = []

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

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

        rows.append({

            "Threshold":
                round(
                    float(threshold),
                    2
                ),

            "Precision":
                precision,

            "Recall":
                recall,

            "F1":
                f1,
        })

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = float(
                threshold
            )

    return (
        best_threshold,
        pd.DataFrame(rows)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("PYTORCH GRU TEMPORAL FORECASTING MODEL")
    print("=" * 80)

    print(
        f"\nPyTorch version: "
        f"{torch.__version__}"
    )

    print(
        f"Device: {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    else:

        print(
            "GPU not available. "
            "Using CPU."
        )

    # ========================================================
    # LOAD
    # ========================================================

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = load_training_data()

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

    # ========================================================
    # BASIC VALIDATION
    # ========================================================

    if X_train.ndim != 3:

        raise ValueError(
            "Expected X_train to have shape "
            "(samples, timesteps, features)."
        )

    samples, timesteps, features = (
        X_train.shape
    )

    print("\nTemporal structure:")

    print(
        f"Samples:       {samples}"
    )

    print(
        f"History:       {timesteps} minutes"
    )

    print(
        f"Features:      {features}"
    )

    # ========================================================
    # DATA LOADERS
    # ========================================================

    train_dataset = NetworkDataset(
        X_train,
        y_train
    )

    validation_dataset = NetworkDataset(
        X_validation,
        y_validation
    )

    test_dataset = NetworkDataset(
        X_test,
        y_test
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # ========================================================
    # CLASS WEIGHT
    # ========================================================

    negative = np.sum(
        y_train == 0
    )

    positive = np.sum(
        y_train == 1
    )

    if positive == 0:

        raise ValueError(
            "No positive attack samples."
        )

    pos_weight = (
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
        f"Positive class weight: "
        f"{pos_weight:.4f}"
    )

    pos_weight_tensor = torch.tensor(
        [pos_weight],
        dtype=torch.float32,
        device=DEVICE
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = AttackForecastGRU(

        input_size=features,

        hidden_size=HIDDEN_SIZE,

        num_layers=NUM_LAYERS,

        dropout=DROPOUT

    ).to(DEVICE)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nModel:")

    print(model)

    print(
        f"\nTrainable parameters: "
        f"{parameter_count:,}"
    )

    # ========================================================
    # LOSS
    # ========================================================

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight_tensor
    )

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY
    )

    # ========================================================
    # TRAINING
    # ========================================================

    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None

    epochs_without_improvement = 0

    history = []

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(
                DEVICE
            )

            y_batch = y_batch.to(
                DEVICE
            )

            optimizer.zero_grad()

            logits = model(
                X_batch
            )

            loss = criterion(
                logits,
                y_batch
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            running_loss += (
                loss.item()
                * len(y_batch)
            )

        train_loss = (
            running_loss
            / len(train_loader.dataset)
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss, val_true, val_prob = (
            evaluate_model(
                model,
                validation_loader,
                criterion
            )
        )

        val_metrics, _ = (
            calculate_metrics(
                val_true,
                val_prob,
                THRESHOLD
            )
        )

        val_f1 = val_metrics["F1"]

        history.append({

            "Epoch":
                epoch,

            "Train_Loss":
                train_loss,

            "Validation_Loss":
                val_loss,

            "Validation_Accuracy":
                val_metrics["Accuracy"],

            "Validation_Precision":
                val_metrics["Precision"],

            "Validation_Recall":
                val_metrics["Recall"],

            "Validation_F1":
                val_metrics["F1"],

            "Validation_ROC_AUC":
                val_metrics["ROC_AUC"],

            "Validation_PR_AUC":
                val_metrics["PR_AUC"],
        })

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                f"\nEarly stopping at epoch "
                f"{epoch}."
            )

            break

    # ========================================================
    # RESTORE BEST MODEL
    # ========================================================

    if best_state is None:

        raise RuntimeError(
            "No best model state was saved."
        )

    model.load_state_dict(
        best_state
    )

    print(
        f"\nBest epoch: {best_epoch}"
    )

    print(
        f"Best validation F1: "
        f"{best_val_f1:.4f}"
    )

    # ========================================================
    # VALIDATION THRESHOLD SEARCH
    # ========================================================

    print("\n" + "=" * 80)
    print("VALIDATION THRESHOLD SEARCH")
    print("=" * 80)

    _, val_true, val_prob = (
        evaluate_model(
            model,
            validation_loader,
            criterion
        )
    )

    best_threshold, threshold_df = (
        find_best_threshold(
            val_true,
            val_prob
        )
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
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    # ========================================================
    # FINAL VALIDATION RESULTS
    # ========================================================

    val_metrics, val_predictions = (
        calculate_metrics(
            val_true,
            val_prob,
            best_threshold
        )
    )

    print("\n" + "=" * 80)
    print("FINAL VALIDATION RESULTS")
    print("=" * 80)

    print(
        f"\nThreshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Accuracy:  "
        f"{val_metrics['Accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{val_metrics['Precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{val_metrics['Recall']:.4f}"
    )

    print(
        f"F1 Score:  "
        f"{val_metrics['F1']:.4f}"
    )

    print(
        f"ROC-AUC:   "
        f"{val_metrics['ROC_AUC']:.4f}"
    )

    print(
        f"PR-AUC:    "
        f"{val_metrics['PR_AUC']:.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            val_true,
            val_predictions
        )
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)

    _, test_true, test_prob = (
        evaluate_model(
            model,
            test_loader,
            criterion
        )
    )

    test_metrics, test_predictions = (
        calculate_metrics(
            test_true,
            test_prob,
            best_threshold
        )
    )

    print(
        f"\nDecision threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Accuracy:  "
        f"{test_metrics['Accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{test_metrics['Precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{test_metrics['Recall']:.4f}"
    )

    print(
        f"F1 Score:  "
        f"{test_metrics['F1']:.4f}"
    )

    print(
        f"ROC-AUC:   "
        f"{test_metrics['ROC_AUC']:.4f}"
    )

    print(
        f"PR-AUC:    "
        f"{test_metrics['PR_AUC']:.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            test_true,
            test_predictions
        )
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            test_true,
            test_predictions,
            target_names=[
                "No Attack",
                "Attack"
            ],
            zero_division=0
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        MODEL_DIR
        / "pytorch_gru_forecast15.pt"
    )

    torch.save({

        "model_state_dict":
            model.state_dict(),

        "input_size":
            features,

        "hidden_size":
            HIDDEN_SIZE,

        "num_layers":
            NUM_LAYERS,

        "dropout":
            DROPOUT,

        "threshold":
            best_threshold,

        "best_epoch":
            best_epoch,

        "validation_f1":
            best_val_f1,

        "test_f1":
            test_metrics["F1"],

    }, model_path)

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history
    )

    history_path = (
        MODEL_DIR
        / "pytorch_gru_training_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics_df = pd.DataFrame([{

        "Model":
            "PyTorch GRU",

        "Best_Epoch":
            best_epoch,

        "Threshold":
            best_threshold,

        "Validation_Accuracy":
            val_metrics["Accuracy"],

        "Validation_Precision":
            val_metrics["Precision"],

        "Validation_Recall":
            val_metrics["Recall"],

        "Validation_F1":
            val_metrics["F1"],

        "Validation_ROC_AUC":
            val_metrics["ROC_AUC"],

        "Validation_PR_AUC":
            val_metrics["PR_AUC"],

        "Test_Accuracy":
            test_metrics["Accuracy"],

        "Test_Precision":
            test_metrics["Precision"],

        "Test_Recall":
            test_metrics["Recall"],

        "Test_F1":
            test_metrics["F1"],

        "Test_ROC_AUC":
            test_metrics["ROC_AUC"],

        "Test_PR_AUC":
            test_metrics["PR_AUC"],

    }])

    metrics_path = (
        MODEL_DIR
        / "pytorch_gru_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions_df = pd.DataFrame({

        "Actual":
            test_true,

        "Probability":
            test_prob,

        "Prediction":
            test_predictions,

        "Threshold":
            best_threshold,

    })

    predictions_path = (
        MODEL_DIR
        / "pytorch_gru_test_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    # --------------------------------------------------------
    # Validation predictions
    # --------------------------------------------------------

    validation_predictions_df = pd.DataFrame({

        "Actual":
            val_true,

        "Probability":
            val_prob,

        "Prediction":
            val_predictions,

        "Threshold":
            best_threshold,

    })

    validation_predictions_path = (
        MODEL_DIR
        / "pytorch_gru_validation_predictions.csv"
    )

    validation_predictions_df.to_csv(
        validation_predictions_path,
        index=False
    )

    # --------------------------------------------------------
    # Threshold analysis
    # --------------------------------------------------------

    threshold_path = (
        MODEL_DIR
        / "pytorch_gru_threshold_analysis.csv"
    )

    threshold_df.to_csv(
        threshold_path,
        index=False
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 80)
    print("PYTORCH GRU COMPLETE")
    print("=" * 80)

    print(
        f"\nBest epoch: "
        f"{best_epoch}"
    )

    print(
        f"Selected threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Validation F1: "
        f"{val_metrics['F1']:.4f}"
    )

    print(
        f"Validation ROC-AUC: "
        f"{val_metrics['ROC_AUC']:.4f}"
    )

    print(
        f"Validation PR-AUC: "
        f"{val_metrics['PR_AUC']:.4f}"
    )

    print(
        f"\nTest F1: "
        f"{test_metrics['F1']:.4f}"
    )

    print(
        f"Test ROC-AUC: "
        f"{test_metrics['ROC_AUC']:.4f}"
    )

    print(
        f"Test PR-AUC: "
        f"{test_metrics['PR_AUC']:.4f}"
    )

    print("\nSaved:")

    print(
        f"  {model_path}"
    )

    print(
        f"  {history_path}"
    )

    print(
        f"  {metrics_path}"
    )

    print(
        f"  {predictions_path}"
    )

    print(
        f"  {validation_predictions_path}"
    )

    print(
        f"  {threshold_path}"
    )


if __name__ == "__main__":
    main()