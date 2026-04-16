"""Convert XML annotations into train/val/test face crops."""

from __future__ import annotations

import argparse
import json

from src.mask_detection.constants import DEFAULT_RAW_DATASET_ROOT
from src.mask_detection.dataset_tools import prepare_classification_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare cropped face images for mask classification."
    )
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_RAW_DATASET_ROOT,
        help="Path to the raw dataset folder containing 'images' and 'annotations'.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/mask_faces",
        help="Directory where the cropped train/val/test dataset will be written.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and rebuild the output directory if it already exists.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = prepare_classification_dataset(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        force=args.force,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

