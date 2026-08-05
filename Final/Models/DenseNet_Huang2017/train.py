"""Train and evaluate DenseNet-121 on the BUSI three-class task.

Run from the project root with, for example:

    python Models/DenseNet/train.py --epochs 30 --batch-size 16
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Models.common.data import (  # noqa: E402
    CLASS_NAMES,
    ClsDataset,
    class_weights,
    make_cls_split,
)
from Models.common.metrics import classification_metrics  # noqa: E402
from Models.common.runner import (  # noqa: E402
    DEVICE,
    plot_confusion,
    plot_curve,
    save_results,
    set_seed,
)

try:  # Supports both module execution and direct script execution.
    from .model import build_model
except ImportError:  # pragma: no cover - used by ``python Models/DenseNet/train.py``
    from model import build_model


MODEL_NAME = "DenseNet_Huang2017"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DenseNet-121 on BUSI")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not load ImageNet pretrained weights.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay cannot be negative")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1)")
    if args.patience < 1:
        raise ValueError("--patience must be at least 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")


def resolve_image_paths(df):
    """Make cached CSV paths independent of the shell's working directory."""
    df = df.copy()

    def resolve(path: str) -> str:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        return str(candidate.resolve())

    df["path"] = df["path"].map(resolve)
    missing = [path for path in df["path"] if not Path(path).is_file()]
    if missing:
        examples = "\n".join(missing[:3])
        raise FileNotFoundError(
            f"{len(missing)} images referenced by split_cls.csv were not found. "
            f"Examples:\n{examples}"
        )
    return df


def make_loaders(df, args: argparse.Namespace):
    datasets = {
        "train": ClsDataset(df, "train", augment=True, rgb3=True),
        "val": ClsDataset(df, "val", augment=False, rgb3=True),
        "test": ClsDataset(df, "test", augment=False, rgb3=True),
    }
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": DEVICE.type == "cuda",
    }
    return {
        split: DataLoader(
            dataset,
            shuffle=split == "train",
            **common,
        )
        for split, dataset in datasets.items()
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size
    return total_loss / max(total_items, 1)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_proba: list[list[float]] = []

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        probabilities = torch.softmax(logits, dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(probabilities.argmax(dim=1).cpu().tolist())
        y_proba.extend(probabilities.cpu().tolist())

    metrics = classification_metrics(
        np.asarray(y_true),
        np.asarray(y_pred),
        np.asarray(y_proba),
        list(CLASS_NAMES),
    )
    return total_loss / max(total_items, 1), metrics


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_metrics: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_metrics": val_metrics,
            "class_names": list(CLASS_NAMES),
            "config": vars(args),
        },
        path,
    )


def load_checkpoint(path: Path, model: nn.Module) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:  # Compatibility with older PyTorch versions.
        checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    print(f"[DenseNet] project root: {PROJECT_ROOT}")
    print(f"[DenseNet] device: {DEVICE}")
    print(f"[DenseNet] classes: {list(CLASS_NAMES)}")

    split_df = resolve_image_paths(make_cls_split(verbose=True))
    loaders = make_loaders(split_df, args)
    print(
        "[DenseNet] samples: "
        + ", ".join(f"{name}={len(loader.dataset)}" for name, loader in loaders.items())
    )

    model = build_model(
        num_classes=len(CLASS_NAMES),
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
    ).to(DEVICE)
    weights = class_weights(split_df).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    results_dir = PROJECT_ROOT / "Models" / "results"
    checkpoint_path = results_dir / "checkpoints" / f"{MODEL_NAME}_best.pt"
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_macro_f1": [],
        "learning_rate": [],
    }
    best_macro_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer)
        val_loss, val_metrics = evaluate(model, loaders["val"], criterion)
        val_macro_f1 = float(val_metrics["macro_f1"])
        scheduler.step(val_macro_f1)
        current_lr = float(optimizer.param_groups[0]["lr"])

        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_macro_f1"].append(val_macro_f1)
        history["learning_rate"].append(current_lr)

        improved = val_macro_f1 > best_macro_f1
        if improved:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                val_metrics,
                args,
            )
        else:
            stale_epochs += 1

        marker = " *" if improved else ""
        print(
            f"epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_macro_f1={val_macro_f1:.4f} | lr={current_lr:.2e}{marker}"
        )

        if stale_epochs >= args.patience:
            print(f"[DenseNet] early stopping after {args.patience} stale epochs")
            break

    best_checkpoint = load_checkpoint(checkpoint_path, model)
    test_loss, test_metrics = evaluate(model, loaders["test"], criterion)
    elapsed = time.time() - start_time

    plot_curve(
        history,
        MODEL_NAME,
        keys=("train_loss", "val_loss", "val_macro_f1"),
    )
    plot_confusion(
        test_metrics["confusion_matrix"],
        list(CLASS_NAMES),
        MODEL_NAME,
    )

    result = {
        "model": "DenseNet121",
        "paper": "Huang et al. (2017), Densely Connected Convolutional Networks",
        "task": "BUSI three-class classification",
        "class_names": list(CLASS_NAMES),
        "device": str(DEVICE),
        "config": vars(args),
        "split_sizes": {name: len(loader.dataset) for name, loader in loaders.items()},
        "class_weights": [round(float(value), 6) for value in weights.cpu().tolist()],
        "best_epoch": int(best_epoch),
        "best_validation": best_checkpoint["val_metrics"],
        "test_loss": round(float(test_loss), 6),
        "test_classification": test_metrics,
        "test": test_metrics,
        "history": history,
        "checkpoint": os.path.relpath(checkpoint_path, PROJECT_ROOT),
        "train_minutes": round(elapsed / 60, 2),
        "elapsed_seconds": round(elapsed, 2),
    }
    result_path = save_results(MODEL_NAME, result)

    print(f"[DenseNet] best epoch: {best_epoch}")
    print(f"[DenseNet] best validation macro-F1: {best_macro_f1:.4f}")
    print(f"[DenseNet] test macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"[DenseNet] test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"[DenseNet] checkpoint: {checkpoint_path}")
    print(f"[DenseNet] results: {result_path}")


if __name__ == "__main__":
    main()
