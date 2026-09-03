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

DATA_DIR = Path("data/processed/unseen_attack")
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
            )
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

        output, _ = self.gru(x)

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
        y_unseen
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

    all_targets = []
    all_probabilities = []

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

            all_targets.extend(
                y_batch.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
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
            )
    }

    return (
        metrics,
        predictions
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("UNSEEN ATTACK PYTORCH GRU EVALUATION")
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
        y_train,
        X_unseen,
        y_unseen
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

    # ========================================================
    # VALIDATION
    # ========================================================

    if X_train.ndim != 3:

        raise ValueError(
            "X_train must have shape "
            "(samples, timesteps, features)."
        )

    if X_unseen.ndim != 3:

        raise ValueError(
            "X_unseen must have shape "
            "(samples, timesteps, features)."
        )

    if X_train.shape[1:] != X_unseen.shape[1:]:

        raise ValueError(
            "Training and unseen sequences "
            "have different temporal shapes."
        )

    samples, timesteps, features = (
        X_train.shape
    )

    print("\nTemporal structure:")

    print(
        f"History:  {timesteps} minutes"
    )

    print(
        f"Features: {features}"
    )

    # ========================================================
    # DISTRIBUTIONS
    # ========================================================

    train_negative = np.sum(
        y_train == 0
    )

    train_positive = np.sum(
        y_train == 1
    )

    unseen_negative = np.sum(
        y_unseen == 0
    )

    unseen_positive = np.sum(
        y_unseen == 1
    )

    if train_positive == 0:

        raise ValueError(
            "Training data contains no attack samples."
        )

    if unseen_positive == 0:

        raise ValueError(
            "Unseen data contains no attack samples."
        )

    pos_weight = (
        train_negative
        / train_positive
    )

    print("\nTraining distribution:")

    print(
        f"No attack: {train_negative}"
    )

    print(
        f"Attack:    {train_positive}"
    )

    print(
        f"Positive class weight: "
        f"{pos_weight:.4f}"
    )

    print("\nUnseen Infilteration distribution:")

    print(
        f"No attack: {unseen_negative}"
    )

    print(
        f"Attack:    {unseen_positive}"
    )

    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = NetworkDataset(
        X_train,
        y_train
    )

    unseen_dataset = NetworkDataset(
        X_unseen,
        y_unseen
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    unseen_loader = torch.utils.data.DataLoader(
        unseen_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
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

    pos_weight_tensor = torch.tensor(
        [pos_weight],
        dtype=torch.float32,
        device=DEVICE
    )

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
    print("TRAINING ON KNOWN ATTACKS")
    print("=" * 80)

    print(
        "\nImportant:"
    )

    print(
        "No Infilteration samples are used during training."
    )

    # --------------------------------------------------------
    # We need a validation mechanism without using the
    # unseen Infilteration test set.
    #
    # Use a chronological portion of the known-attack
    # training data as an internal validation set.
    # --------------------------------------------------------

    split_index = int(
        len(X_train) * 0.80
    )

    X_internal_train = X_train[
        :split_index
    ]

    y_internal_train = y_train[
        :split_index
    ]

    X_internal_val = X_train[
        split_index:
    ]

    y_internal_val = y_train[
        split_index:
    ]

    print(
        f"\nInternal training sequences: "
        f"{len(X_internal_train)}"
    )

    print(
        f"Internal validation sequences: "
        f"{len(X_internal_val)}"
    )

    internal_train_dataset = NetworkDataset(
        X_internal_train,
        y_internal_train
    )

    internal_val_dataset = NetworkDataset(
        X_internal_val,
        y_internal_val
    )

    internal_train_loader = torch.utils.data.DataLoader(
        internal_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    internal_val_loader = torch.utils.data.DataLoader(
        internal_val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

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

        for X_batch, y_batch in internal_train_loader:

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
            / len(
                internal_train_loader.dataset
            )
        )

        # ----------------------------------------------------
        # Internal validation
        # ----------------------------------------------------

        val_loss, val_true, val_prob = (
            evaluate_model(
                model,
                internal_val_loader,
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

            "Validation_F1":
                val_f1,

            "Validation_ROC_AUC":
                val_metrics["ROC_AUC"],

            "Validation_PR_AUC":
                val_metrics["PR_AUC"]
        })

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

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
        f"\nBest epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best internal validation F1: "
        f"{best_val_f1:.4f}"
    )

    # ========================================================
    # UNSEEN TEST
    # ========================================================

    print("\n" + "=" * 80)
    print("UNSEEN INFILTERATION TEST")
    print("=" * 80)

    print(
        "\nThe model has never seen Infilteration during training."
    )

    print(
        "Threshold remains fixed at 0.50."
    )

    _, test_true, test_prob = (
        evaluate_model(
            model,
            unseen_loader,
            criterion
        )
    )

    test_metrics, test_predictions = (
        calculate_metrics(
            test_true,
            test_prob,
            THRESHOLD
        )
    )

    print(
        f"\nDecision threshold: "
        f"{THRESHOLD:.2f}"
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
    # PROBABILITY ANALYSIS
    # ========================================================

    print(
        "\nPrediction probability summary:"
    )

    print(
        pd.Series(
            test_prob
        ).describe().to_string()
    )

    # ========================================================
    # SAVE
    # ========================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR
        / "pytorch_gru_unseen_infilteration.pt"
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
            THRESHOLD,

        "best_epoch":
            best_epoch,

        "internal_validation_f1":
            best_val_f1,

        "unseen_f1":
            test_metrics["F1"],

        "unseen_roc_auc":
            test_metrics["ROC_AUC"],

        "unseen_pr_auc":
            test_metrics["PR_AUC"],

    }, model_path)

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    history_path = (
        MODEL_DIR
        / "pytorch_gru_unseen_infilteration_history.csv"
    )

    pd.DataFrame(
        history
    ).to_csv(
        history_path,
        index=False
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics_path = (
        MODEL_DIR
        / "pytorch_gru_unseen_infilteration_metrics.csv"
    )

    pd.DataFrame([{

        "Experiment":
            "Unseen Infilteration",

        "Model":
            "PyTorch GRU",

        "Best_Epoch":
            best_epoch,

        "Threshold":
            THRESHOLD,

        "Internal_Validation_F1":
            best_val_f1,

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

    }]).to_csv(
        metrics_path,
        index=False
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions_path = (
        MODEL_DIR
        / "pytorch_gru_unseen_infilteration_predictions.csv"
    )

    pd.DataFrame({

        "Actual":
            test_true,

        "Probability":
            test_prob,

        "Prediction":
            test_predictions,

        "Threshold":
            THRESHOLD,

    }).to_csv(
        predictions_path,
        index=False
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 80)
    print("UNSEEN ATTACK PYTORCH GRU COMPLETE")
    print("=" * 80)

    print(
        "\nTraining attacks:"
    )

    print(
        "  Bot"
    )

    print(
        "  FTP-BruteForce"
    )

    print(
        "  SSH-Bruteforce"
    )

    print(
        "\nUnseen attack:"
    )

    print(
        "  Infilteration"
    )

    print(
        "\nResults:"
    )

    print(
        f"  F1:      "
        f"{test_metrics['F1']:.4f}"
    )

    print(
        f"  Recall:  "
        f"{test_metrics['Recall']:.4f}"
    )

    print(
        f"  ROC-AUC: "
        f"{test_metrics['ROC_AUC']:.4f}"
    )

    print(
        f"  PR-AUC:  "
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


if __name__ == "__main__":
    main()