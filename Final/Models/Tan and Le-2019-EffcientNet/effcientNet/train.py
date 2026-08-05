import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from model import buildmodel

THIS_DIR = Path(__file__).resolve().parent

# THIS_DIR = .../Models/Tan and Le-2019-EffcientNet/effcientNet
# parents[2] = the project root (the folder that contains the `Models` package)
PROJECT_ROOT = THIS_DIR.parents[2]

# Add project root (for `Models.common.*`) and this dir (for `from model import`).
for p in (str(PROJECT_ROOT), str(THIS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from Models.common.data import (
    CLASS_NAMES,
    ClsDataset,
    class_weights,
    make_cls_split,
)

from Models.common.metrics import classification_metrics

from Models.common.runner import (
    plot_confusion,
    plot_curve,
    save_results,
    set_seed,
)

SEED = 42

BATCH_SIZE = 16

EPOCHS = 20

LEARNING_RATE = 0.0001

NUM_WORKERS = 0

MODEL_NAME = "EfficientNet_TanLe2019"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def get_device():
#     """
#     Select the best available device.

#     CUDA:
#         NVIDIA GPU

#     MPS:
#         Apple Silicon GPU

#     CPU:
#         Used when no GPU is available
#     """

#     if torch.cuda.is_available():
#         return torch.device("cuda")

#     if torch.backends.mps.is_available():
#         return torch.device("mps")

#     return torch.device("cpu")


def run_one_epoch(
    model,
    loader,
    loss_fn,
    device,
    optimizer=None,
):
    """
    Run one complete pass through a dataset.

    When optimizer is provided:
        training mode

    When optimizer is None:
        validation or test mode
    """

    # Determine whether this is training.
    is_training = optimizer is not None

    # Set the model mode.
    if is_training:
        model.train()
    else:
        model.eval()

    # Store total loss.
    total_loss = 0.0

    # Store number of correct predictions.
    total_correct = 0

    # Store number of images.
    total_samples = 0

    # Store all true labels.
    all_labels = []

    # Store all predicted labels.
    all_predictions = []

    # Store all predicted probabilities.
    all_probabilities = []

    # Only calculate gradients during training.
    with torch.set_grad_enabled(is_training):

        # Read one batch at a time.
        for images, labels in loader:

            # Move images to CPU, CUDA or MPS.
            images = images.to(device)

            # Move labels to the same device.
            labels = labels.to(device)

            # Clear gradients from the previous batch.
            if is_training:
                optimizer.zero_grad()

            # Forward propagation.
            logits = model(images)

            # Calculate classification loss.
            loss = loss_fn(logits, labels)

            # Back propagation and parameter update.
            if is_training:
                loss.backward()
                optimizer.step()

            # Convert logits into probabilities.
            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            # Select the class with the highest probability.
            predictions = torch.argmax(
                probabilities,
                dim=1,
            )

            # Number of images in this batch.
            current_batch_size = labels.size(0)

            # Add this batch loss.
            total_loss += loss.item() * current_batch_size

            # Count correct predictions.
            total_correct += (
                predictions == labels
            ).sum().item()

            # Count images.
            total_samples += current_batch_size

            # Move results back to CPU and save them.
            all_labels.extend(
                labels.detach().cpu().tolist()
            )

            all_predictions.extend(
                predictions.detach().cpu().tolist()
            )

            all_probabilities.extend(
                probabilities.detach().cpu().tolist()
            )

    # Calculate average loss.
    average_loss = total_loss / total_samples

    # Calculate accuracy.
    accuracy = total_correct / total_samples

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "labels": np.asarray(all_labels),
        "predictions": np.asarray(all_predictions),
        "probabilities": np.asarray(all_probabilities),
    }


def main():

    set_seed(SEED)


    split_df = make_cls_split(
        force=False,
        verbose=True,
    )

    train_dataset = ClsDataset(
        split_df,
        split="train",
        augment=True,
        rgb3=True,
    )

    val_dataset = ClsDataset(
        split_df,
        split="val",
        augment=False,
        rgb3=True,
    )

    test_dataset = ClsDataset(
        split_df,
        split="test",
        augment=False,
        rgb3=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print("Train images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Test images:", len(test_dataset))

    model = buildmodel(
        num_classes=len(CLASS_NAMES),
        pretrained=True,
        dropout=0.2,
        freeze=False,
    )

    # Move model to the selected device.
    model = model.to(device)

    weights = class_weights(
        split_df,
        split="train",
    )

    weights = weights.to(device)

    print("Class names:", CLASS_NAMES)
    print("Class weights:", weights)

    loss_fn = nn.CrossEntropyLoss(
        weight=weights,
    )

    trainable_parameters = filter(
        lambda parameter: parameter.requires_grad,
        model.parameters(),
    )

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=LEARNING_RATE,
    )

    checkpoint_path = (
        THIS_DIR / "best_efficientnet_b0.pt"
    )

    best_val_f1 = -1.0

    # Store results from every epoch.
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "val_macro_f1": [],
    }

    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):

        # Train for one epoch.
        train_result = run_one_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=optimizer,
        )

        # Validate for one epoch.
        val_result = run_one_epoch(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=None,
        )

        # Calculate validation metrics.
        val_metrics = classification_metrics(
            val_result["labels"],
            val_result["predictions"],
            val_result["probabilities"],
            CLASS_NAMES,
        )

        val_macro_f1 = val_metrics["macro_f1"]

        # Save history.
        history["train_loss"].append(
            train_result["loss"]
        )

        history["val_loss"].append(
            val_result["loss"]
        )

        history["train_accuracy"].append(
            train_result["accuracy"]
        )

        history["val_accuracy"].append(
            val_result["accuracy"]
        )

        history["val_macro_f1"].append(
            val_macro_f1
        )

        # Print this epoch's results.
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss: {train_result['loss']:.4f} | "
            f"train accuracy: {train_result['accuracy']:.4f} | "
            f"val loss: {val_result['loss']:.4f} | "
            f"val accuracy: {val_result['accuracy']:.4f} | "
            f"val macro-F1: {val_macro_f1:.4f}"
        )

        # Save the model when validation macro-F1 improves.
        if val_macro_f1 > best_val_f1:

            best_val_f1 = val_macro_f1

            torch.save(
                model.state_dict(),
                checkpoint_path,
            )

            print("Saved a new best model.")

    best_state_dict = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(best_state_dict)

    test_result = run_one_epoch(
        model=model,
        loader=test_loader,
        loss_fn=loss_fn,
        device=device,
        optimizer=None,
    )

    test_metrics = classification_metrics(
        test_result["labels"],
        test_result["predictions"],
        test_result["probabilities"],
        CLASS_NAMES,
    )

    print()
    print("Test Results")
    print("------------------------------")

    print(
        "Accuracy:",
        test_metrics["accuracy"],
    )

    print(
        "Balanced accuracy:",
        test_metrics["balanced_accuracy"],
    )

    print(
        "Macro-F1:",
        test_metrics["macro_f1"],
    )

    print(
        "Macro AUC:",
        test_metrics["macro_auc_ovr"],
    )

    print("Per-class results:")

    for class_name, result in test_metrics["per_class"].items():
        print(
            class_name,
            result,
        )

    print("Confusion matrix:")

    print(
        np.asarray(
            test_metrics["confusion_matrix"]
        )
    )
    train_min = (time.time() - t0) / 60

    output = {
        "model": "EfficientNet-B0",
        "paper": "Tan and Le (2019), EfficientNet [06]",
        "paper_number": 6,
        "task": "3-class classification (normal / benign / malignant)",
        "classes": list(CLASS_NAMES),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "best_val_macro_f1": best_val_f1,
        "train_minutes": round(train_min, 2),
        "test_classification": test_metrics,
        "test": test_metrics,
        "history": history,
    }

    save_results(
        MODEL_NAME,
        output,
    )

    plot_curve(
        history,
        MODEL_NAME,
        keys=(
            "train_loss",
            "val_loss",
            "val_macro_f1",
        ),
    )
    plot_confusion(
        test_metrics["confusion_matrix"],
        CLASS_NAMES,
        MODEL_NAME,
    )

    print()
    print("Training and testing finished.")

if __name__ == "__main__":
    main()