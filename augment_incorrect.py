"""Oversample the 'mask_weared_incorrect' class for YOLO training data.

This script augments only the training split (data/yolo/train) by creating
additional copies of images that contain the incorrect-mask class. It applies
safe transforms that keep bounding boxes valid:
- horizontal flip (updates YOLO bboxes)
- color jitter (brightness/contrast)
- gaussian blur

Usage:
  python augment_incorrect.py --data-dir data/yolo --target-count 500
  python augment_incorrect.py --data-dir data/yolo --multiplier 6

Defaults:
  target-count = max class count in train split
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Augment incorrect-mask class for YOLO.")                                
    parser.add_argument("--data-dir", default="data/yolo", help="YOLO dataset root.")
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="Target count for incorrect-mask labels in TRAIN split."
        " Default = max class count in train.",
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=None,
        help="Multiply incorrect-mask samples by this factor (overrides target-count).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for augmentation choices.",
    )
    parser.add_argument(
        "--prob-flip",
        type=float,
        default=0.5,
        help="Probability of horizontal flip.",
    )
    parser.add_argument(
        "--prob-color",
        type=float,
        default=0.8,
        help="Probability of color jitter.",
    )
    parser.add_argument(
        "--prob-blur",
        type=float,
        default=0.3,
        help="Probability of gaussian blur.",
    )
    return parser


def load_class_mapping(dataset_yaml: Path) -> dict[int, str]:
    data = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    names = data["names"]
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    return {i: n for i, n in enumerate(names)}
     

def count_labels(labels_dir: Path, class_name_by_id: dict[int, str]) -> dict[str, int]:
    counts = {name: 0 for name in class_name_by_id.values()}
    for label_file in labels_dir.glob("*.txt"):
        for line in label_file.read_text().splitlines():
            if not line.strip():
                continue
            cls_id = int(line.split()[0])
            counts[class_name_by_id[cls_id]] += 1
    return counts


def read_labels(label_file: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for line in label_file.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        cls_id = float(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        rows.append([cls_id, cx, cy, w, h])
    return rows


def write_labels(label_file: Path, rows: list[list[float]]) -> None:
    lines = []
    for cls_id, cx, cy, w, h in rows:
        lines.append(f"{int(cls_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    label_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def horizontal_flip(image: Image.Image, rows: list[list[float]]) -> tuple[Image.Image, list[list[float]]]:
    flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
    new_rows = []
    for cls_id, cx, cy, w, h in rows:
        new_rows.append([cls_id, 1.0 - cx, cy, w, h])
    return flipped, new_rows


def color_jitter(image: Image.Image) -> Image.Image:
    # Random brightness & contrast
    brightness = ImageEnhance.Brightness(image)
    contrast = ImageEnhance.Contrast(image)
    image = brightness.enhance(random.uniform(0.7, 1.3))
    image = contrast.enhance(random.uniform(0.7, 1.3))
    return image


def gaussian_blur(image: Image.Image) -> Image.Image:
    radius = random.uniform(0.5, 1.2)
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    labels_dir = data_dir / "train" / "labels"
    images_dir = data_dir / "train" / "images"
    dataset_yaml = Path("dataset.yaml")

    if not labels_dir.exists() or not images_dir.exists():
        raise SystemExit(f"Missing train split in {data_dir}. Run prepare_dataset.py first.")
    if not dataset_yaml.exists():
        raise SystemExit("dataset.yaml not found in project root.")

    class_name_by_id = load_class_mapping(dataset_yaml)
    incorrect_id = None
    for cls_id, name in class_name_by_id.items():
        if name == "mask_weared_incorrect":
            incorrect_id = cls_id
            break
    if incorrect_id is None:
        raise SystemExit("Class 'mask_weared_incorrect' not found in dataset.yaml.")

    counts = count_labels(labels_dir, class_name_by_id)
    incorrect_count = counts.get("mask_weared_incorrect", 0)
    target = args.target_count

    if args.multiplier is not None:
        target = incorrect_count * args.multiplier

    if target is None:
        target = max(counts.values())

    if incorrect_count >= target:
        print(f"Already balanced: incorrect={incorrect_count}, target={target}")
        return

    # Find candidate images containing incorrect class
    candidate_files: list[Path] = []
    for label_file in labels_dir.glob("*.txt"):
        rows = read_labels(label_file)
        if any(int(r[0]) == incorrect_id for r in rows):
            candidate_files.append(label_file)

    if not candidate_files:
        raise SystemExit("No training labels contain 'mask_weared_incorrect'.")

    needed = target - incorrect_count
    print(f"Current incorrect: {incorrect_count}. Target: {target}. Need: {needed}")

    created = 0
    index = 0

    while created < needed:
        label_file = candidate_files[index % len(candidate_files)]
        image_file = images_dir / (label_file.stem + ".jpg")
        if not image_file.exists():
            # Try png if jpg not found
            image_file = images_dir / (label_file.stem + ".png")
        if not image_file.exists():
            index += 1
            continue

        rows = read_labels(label_file)
        image = Image.open(image_file).convert("RGB")

        # Apply random transforms
        if random.random() < args.prob_flip:
            image, rows = horizontal_flip(image, rows)

        if random.random() < args.prob_color:
            image = color_jitter(image)

        if random.random() < args.prob_blur:
            image = gaussian_blur(image)

        suffix = f"_aug{created:05d}"
        new_image = images_dir / f"{label_file.stem}{suffix}.jpg"
        new_label = labels_dir / f"{label_file.stem}{suffix}.txt"

        image.save(new_image, quality=95)
        write_labels(new_label, rows)

        created += sum(1 for r in rows if int(r[0]) == incorrect_id)
        index += 1

    print(f"Added ~{created} incorrect-mask labels.")


if __name__ == "__main__":
    main()
