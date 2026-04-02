from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data_utils import build_dataset_from_raw, stratified_split, summarize_feature_csv
from model import Bearing1DCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bearing fault detection with FFT/STFT features and 1D-CNN"
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("raw"))
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("feature_time_48k_2048_load_1.csv"),
        help="CSV is used for dataset summary logging and sanity checks.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--cache-path", type=Path, default=Path("artifacts/dataset_cache.npz"))
    parser.add_argument("--rebuild-cache", action="store_true")

    parser.add_argument("--window-size", type=int, default=2048)
    parser.add_argument("--step-size", type=int, default=1024)
    parser.add_argument("--sampling-rate", type=int, default=48000)
    parser.add_argument("--max-windows-per-file", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Enable DataLoader pin_memory (optional on Windows).",
    )

    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto")

    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)

    use_amp = device.type == "cuda"
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_targets: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []

    progress = tqdm(loader, leave=False)
    for batch_x, batch_y in progress:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

        if is_train:
            if scaler is None:
                raise RuntimeError("GradScaler is required in training mode.")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        preds = torch.argmax(logits, dim=1)
        total_loss += float(loss.item()) * int(batch_x.size(0))
        total_correct += int((preds == batch_y).sum().item())
        total_samples += int(batch_x.size(0))

        all_targets.append(batch_y.detach().cpu().numpy())
        all_preds.append(preds.detach().cpu().numpy())

    epoch_loss = total_loss / max(total_samples, 1)
    epoch_acc = total_correct / max(total_samples, 1)

    y_true = np.concatenate(all_targets) if all_targets else np.empty((0,), dtype=np.int64)
    y_pred = np.concatenate(all_preds) if all_preds else np.empty((0,), dtype=np.int64)
    return epoch_loss, epoch_acc, y_true, y_pred


def plot_training_curves(history: dict[str, list[float]], out_path: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training/Validation Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training/Validation Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def load_or_build_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    args.cache_path.parent.mkdir(parents=True, exist_ok=True)

    if args.cache_path.exists() and not args.rebuild_cache:
        cached = np.load(args.cache_path, allow_pickle=True)
        X = cached["X"].astype(np.float32)
        y = cached["y"].astype(np.int64)
        class_names = [str(v) for v in cached["class_names"].tolist()]
        metadata = json.loads(str(cached["metadata_json"].item()))
        metadata["cache_source"] = str(args.cache_path)
        print(f"Loaded cached dataset from: {args.cache_path}")
        return X, y, class_names, metadata

    X, y, class_names, metadata = build_dataset_from_raw(
        raw_dir=args.raw_dir,
        window_size=args.window_size,
        step_size=args.step_size,
        fs=args.sampling_rate,
        max_windows_per_file=args.max_windows_per_file,
        random_seed=args.seed,
    )

    np.savez_compressed(
        args.cache_path,
        X=X,
        y=y,
        class_names=np.asarray(class_names),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    print(f"Saved dataset cache to: {args.cache_path}")
    return X, y, class_names, metadata


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = select_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_summary = summarize_feature_csv(args.csv_path)
    if csv_summary is not None:
        print("CSV summary:", json.dumps(csv_summary, indent=2))
    else:
        print(f"CSV not found at {args.csv_path}. Continuing with raw MAT processing only.")

    X, y, class_names, dataset_metadata = load_or_build_dataset(args)

    splits = stratified_split(
        X,
        y,
        test_size=args.test_size,
        val_size=args.val_size,
        random_seed=args.seed,
    )
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    pin_memory = bool(args.pin_memory and device.type == "cuda")
    train_loader = make_dataloader(
        X_train,
        y_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = make_dataloader(
        X_val,
        y_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = make_dataloader(
        X_test,
        y_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = Bearing1DCNN(input_channels=X_train.shape[1], num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    best_checkpoint_path = args.output_dir / "best_model.pt"

    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _, _ = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
        )

        with torch.no_grad():
            val_loss, val_acc, _, _ = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
                scaler=None,
            )

        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": class_names,
                    "input_channels": int(X_train.shape[1]),
                    "window_size": int(X_train.shape[2]),
                    "best_val_acc": float(best_val_acc),
                    "args": serializable_args,
                },
                best_checkpoint_path,
            )

    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with torch.no_grad():
        test_loss, test_acc, y_true, y_pred = run_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=None,
        )

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

    training_plot_path = args.output_dir / "training_curves.png"
    cm_plot_path = args.output_dir / "confusion_matrix.png"
    metrics_path = args.output_dir / "metrics.json"
    report_path = args.output_dir / "classification_report.txt"

    plot_training_curves(history, training_plot_path)
    plot_confusion_matrix(cm, class_names, cm_plot_path)

    metrics: dict[str, Any] = {
        "device": {
            "selected": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "cuda_device_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
            "torch_version": torch.__version__,
        },
        "dataset": dataset_metadata,
        "splits": {
            "train": int(X_train.shape[0]),
            "val": int(X_val.shape[0]),
            "test": int(X_test.shape[0]),
        },
        "best_val_accuracy": float(best_val_acc),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "history": history,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "csv_summary": csv_summary,
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with report_path.open("w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n=== Test Classification Report ===")
    print(report_text)
    print("\nArtifacts saved:")
    print(f"- {best_checkpoint_path}")
    print(f"- {metrics_path}")
    print(f"- {report_path}")
    print(f"- {training_plot_path}")
    print(f"- {cm_plot_path}")


if __name__ == "__main__":
    main()
