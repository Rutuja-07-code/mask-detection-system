"""Train a face-mask classifier from cropped face images."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import tensorflow as tf

from src.mask_detection.constants import DEFAULT_RAW_DATASET_ROOT
from src.mask_detection.dataset_tools import prepare_classification_dataset
from src.mask_detection.model import build_mask_classifier
from src.mask_detection.runtime import (
    build_image_datasets,
    checkpoint_to_jsonable,
    metadata_path_for_model,
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


def build_class_weights(data_dir: str | Path, class_names: list[str]) -> dict[int, float]:
    train_dir = Path(data_dir) / "train"
    counts = Counter()

    for class_name in class_names:
        counts[class_name] = sum(1 for path in (train_dir / class_name).glob("*") if path.is_file())

    total = sum(counts.values())
    num_classes = len(class_names)

    class_weights: dict[int, float] = {}
    for class_index, class_name in enumerate(class_names):
        class_count = counts[class_name]
        if class_count == 0:
            raise RuntimeError(f"Class '{class_name}' has no training images in {train_dir}")
        class_weights[class_index] = total / (num_classes * class_count)
    return class_weights


class LearningRateTracker(tf.keras.callbacks.Callback):
    """Capture the effective learning rate at the end of each epoch."""

    def __init__(self) -> None:
        super().__init__()
        self.values: list[float] = []

    def on_epoch_end(self, epoch, logs=None) -> None:
        learning_rate = self.model.optimizer.learning_rate
        if isinstance(learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
            value = learning_rate(self.model.optimizer.iterations)
        else:
            value = learning_rate
        self.values.append(float(tf.keras.backend.get_value(value)))


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    metadata = ensure_dataset(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, test_dataset, class_names = build_image_datasets(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    model = build_mask_classifier(image_size=args.image_size, num_classes=len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    class_weights = build_class_weights(args.data_dir, class_names)
    best_model_path = output_dir / "best_model.keras"
    last_model_path = output_dir / "last_model.keras"
    learning_rate_tracker = LearningRateTracker()

    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(last_model_path),
            save_best_only=False,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            mode="max",
            factor=0.5,
            patience=2,
            verbose=1,
        ),
        learning_rate_tracker,
    ]

    device = select_device()
    print(f"Training on device: {device}")
    print(f"Classes: {class_names}")
    print(f"Prepared data directory: {args.data_dir}")

    history_callback = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    history_rows = []
    history = history_callback.history
    epoch_count = len(history.get("loss", []))

    for epoch_index in range(epoch_count):
        history_rows.append(
            {
                "epoch": epoch_index + 1,
                "train_loss": float(history["loss"][epoch_index]),
                "train_accuracy": float(history["accuracy"][epoch_index]),
                "val_loss": float(history["val_loss"][epoch_index]),
                "val_accuracy": float(history["val_accuracy"][epoch_index]),
                "learning_rate": learning_rate_tracker.values[epoch_index],
            }
        )

    best_epoch = max(history_rows, key=lambda row: row["val_accuracy"])
    best_model = tf.keras.models.load_model(best_model_path)
    test_loss, test_acc = best_model.evaluate(test_dataset, verbose=0)

    summary = {
        "dataset": metadata,
        "device": device,
        "epochs": epoch_count,
        "best_val_accuracy": float(best_epoch["val_accuracy"]),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "class_names": class_names,
        "history": history_rows,
    }

    best_checkpoint = {
        "model_path": str(best_model_path.resolve()),
        "class_names": class_names,
        "image_size": args.image_size,
        "metrics": best_epoch,
    }
    last_checkpoint = {
        "model_path": str(last_model_path.resolve()),
        "class_names": class_names,
        "image_size": args.image_size,
        "metrics": history_rows[-1],
    }

    save_json(output_dir / "training_summary.json", summary)
    save_json(metadata_path_for_model(best_model_path), checkpoint_to_jsonable(best_checkpoint))
    save_json(metadata_path_for_model(last_model_path), checkpoint_to_jsonable(last_checkpoint))

    print(f"Best validation accuracy: {best_epoch['val_accuracy']:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Best checkpoint saved to: {best_model_path}")


if __name__ == "__main__":
    main()
