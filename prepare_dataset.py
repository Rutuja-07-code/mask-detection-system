"""Prepare the dataset for mask detection.

Supports two modes:
  --mode classify  : Crop face ROIs into class folders (legacy TensorFlow pipeline).
  --mode yolo      : Convert VOC XML → YOLO TXT labels (Ultralytics pipeline).
"""

from __future__ import annotations

import argparse
import json

from src.mask_detection.constants import DEFAULT_RAW_DATASET_ROOT
from src.mask_detection.dataset_tools import (
    prepare_classification_dataset,
    prepare_yolo_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the mask detection dataset in classification or YOLO format."
    )
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_RAW_DATASET_ROOT,
        help="Path to the raw dataset folder containing 'images' and 'annotations'.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory where the prepared dataset will be written. "
            "Defaults to 'data/mask_faces' (classify) or 'data/yolo' (yolo)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["classify", "yolo"],
        default="yolo",
        help="Dataset format to generate (default: yolo).",
    )
    parser.add_argument(
        "--yaml-path",
        default="dataset.yaml",
        help="[yolo mode] Where to write the dataset YAML file.",
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

    if args.mode == "yolo":
        output_dir = args.output_dir or "data/yolo"
        metadata = prepare_yolo_dataset(
            dataset_root=args.dataset_root,
            output_dir=output_dir,
            yaml_path=args.yaml_path,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            force=args.force,
        )
    else:
        output_dir = args.output_dir or "data/mask_faces"
        metadata = prepare_classification_dataset(
            dataset_root=args.dataset_root,
            output_dir=output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            force=args.force,
        )

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
