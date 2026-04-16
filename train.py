"""Train a face-mask classifier from cropped face images."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.mask_detection.constants import DEFAULT_RAW_DATASET_ROOT
from src.mask_detection.dataset_tools import prepare_classification_dataset
from src.mask_detection.model import MaskClassifier
from src.mask_detection.runtime import (
    build_transforms,
    checkpoint_to_jsonable,
    save_json,
    select_device,
    set_seed,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a mask classification model.")
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_RAW_DATASET_ROOT,
        help="Raw dataset root with XML annotations and images.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/mask_faces",
        help="Prepared cropped dataset directory.",
    )
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--image-size", type=int, default=128, help="Square input image size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count. Keep at 0 on systems where multiprocessing is flaky.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory for checkpoints and training history.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="Rebuild the prepared cropped dataset before training.",
    )
    return parser


def ensure_dataset(args) -> dict:
    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    if args.force_prepare or not train_dir.exists():
        return prepare_classification_dataset(
            dataset_root=args.dataset_root,
            output_dir=data_dir,
            seed=args.seed,
            force=args.force_prepare,
        )
    metadata_path = data_dir / "metadata.json"
    if metadata_path.exists():
        import json

        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {"output_dir": str(data_dir)}


def build_dataloaders(args):
    train_dataset = ImageFolder(
        root=str(Path(args.data_dir) / "train"),
        transform=build_transforms(args.image_size, train=True),
    )
    val_dataset = ImageFolder(
        root=str(Path(args.data_dir) / "val"),
        transform=build_transforms(args.image_size, train=False),
    )
    test_dataset = ImageFolder(
        root=str(Path(args.data_dir) / "test"),
        transform=build_transforms(args.image_size, train=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return train_dataset, train_loader, val_loader, test_loader


def build_class_weights(dataset: ImageFolder) -> torch.Tensor:
    counts = Counter(dataset.targets)
    total = sum(counts.values())
    num_classes = len(dataset.classes)
    weights = []
    for class_index in range(num_classes):
        class_count = counts[class_index]
        weights.append(total / (num_classes * class_count))
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        predictions = outputs.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_examples += labels.size(0)

    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * inputs.size(0)
        predictions = outputs.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_examples += labels.size(0)

    return total_loss / total_examples, total_correct / total_examples


def save_checkpoint(path: Path, model, optimizer, epoch: int, args, class_names, metrics: dict):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "class_names": class_names,
        "image_size": args.image_size,
        "metrics": metrics,
    }
    torch.save(checkpoint, path)
    return checkpoint


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    metadata = ensure_dataset(args)

    device = select_device()
    train_dataset, train_loader, val_loader, test_loader = build_dataloaders(args)

    model = MaskClassifier(num_classes=len(train_dataset.classes)).to(device)
    class_weights = build_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val_acc = 0.0
    best_checkpoint = None

    print(f"Training on device: {device}")
    print(f"Classes: {train_dataset.classes}")
    print(f"Prepared data directory: {args.data_dir}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        last_checkpoint = save_checkpoint(
            output_dir / "last_model.pt",
            model,
            optimizer,
            epoch,
            args,
            train_dataset.classes,
            row,
        )
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_checkpoint = save_checkpoint(
                output_dir / "best_model.pt",
                model,
                optimizer,
                epoch,
                args,
                train_dataset.classes,
                row,
            )

    if best_checkpoint is None:
        best_checkpoint = last_checkpoint

    best_checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
    best_model = MaskClassifier(num_classes=len(train_dataset.classes)).to(device)
    best_model.load_state_dict(best_checkpoint["model_state_dict"])
    test_loss, test_acc = evaluate(best_model, test_loader, criterion, device)

    summary = {
        "dataset": metadata,
        "device": str(device),
        "epochs": args.epochs,
        "best_val_accuracy": best_val_acc,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "class_names": train_dataset.classes,
        "history": history,
    }
    save_json(output_dir / "training_summary.json", summary)
    save_json(output_dir / "best_model_metadata.json", checkpoint_to_jsonable(best_checkpoint))

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Best checkpoint saved to: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
