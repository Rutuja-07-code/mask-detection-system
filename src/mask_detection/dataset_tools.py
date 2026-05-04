"""Utilities for converting XML annotations into YOLO and cropped-face datasets."""

from __future__ import annotations

import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .constants import CLASS_NAMES


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FaceAnnotation:
    label: str
    bbox: tuple[int, int, int, int]  # (xmin, ymin, xmax, ymax) in pixels


@dataclass(frozen=True)
class ImageRecord:
    image_path: Path
    annotations: tuple[FaceAnnotation, ...]


# ---------------------------------------------------------------------------
# XML ingestion (shared by both pipelines)
# ---------------------------------------------------------------------------

def _safe_int(text: str | None, default: int = 0) -> int:
    if text is None:
        return default
    return int(float(text))


def load_image_records(dataset_root: str | Path) -> list[ImageRecord]:
    """Read every Pascal-VOC XML annotation and return a list of ImageRecords."""
    dataset_root = Path(dataset_root)
    annotations_dir = dataset_root / "annotations"
    images_dir = dataset_root / "images"

    if not annotations_dir.exists() or not images_dir.exists():
        raise FileNotFoundError(
            f"Expected 'annotations' and 'images' inside {dataset_root}"
        )

    records: list[ImageRecord] = []

    for xml_path in sorted(annotations_dir.glob("*.xml")):
        root = ET.parse(xml_path).getroot()
        filename = root.findtext("filename")
        if not filename:
            continue

        image_path = images_dir / filename
        if not image_path.exists():
            continue

        annotations: list[FaceAnnotation] = []
        for obj in root.findall("object"):
            label = (obj.findtext("name") or "").strip()
            if label not in CLASS_NAMES:
                continue

            box = obj.find("bndbox")
            if box is None:
                continue

            xmin = _safe_int(box.findtext("xmin"))
            ymin = _safe_int(box.findtext("ymin"))
            xmax = _safe_int(box.findtext("xmax"))
            ymax = _safe_int(box.findtext("ymax"))

            if xmax <= xmin or ymax <= ymin:
                continue

            annotations.append(FaceAnnotation(label=label, bbox=(xmin, ymin, xmax, ymax)))

        if annotations:
            records.append(ImageRecord(image_path=image_path, annotations=tuple(annotations)))

    if not records:
        raise RuntimeError(f"No valid annotations were found in {dataset_root}")

    return records


# ---------------------------------------------------------------------------
# Train / val / test splitter
# ---------------------------------------------------------------------------

def _split_records(
    records: list[ImageRecord],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, list[ImageRecord]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be below 1")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


# ---------------------------------------------------------------------------
# Pipeline 1: cropped-face classification dataset (legacy / TF)
# ---------------------------------------------------------------------------

def prepare_classification_dataset(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    force: bool = False,
) -> dict:
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)

    if output_dir.exists():
        if force:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise FileExistsError(
                f"{output_dir} already exists and is not empty. Pass force=True to rebuild it."
            )

    records = load_image_records(dataset_root)
    splits = _split_records(records, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)

    split_counts: dict[str, Counter] = defaultdict(Counter)
    image_counts: dict[str, int] = {}

    for split_name, split_records in splits.items():
        image_counts[split_name] = len(split_records)
        for class_name in CLASS_NAMES:
            (output_dir / split_name / class_name).mkdir(parents=True, exist_ok=True)

        for record in split_records:
            with Image.open(record.image_path) as image:
                rgb_image = image.convert("RGB")
                width, height = rgb_image.size

                for index, annotation in enumerate(record.annotations):
                    xmin, ymin, xmax, ymax = annotation.bbox
                    xmin = max(0, min(xmin, width - 1))
                    ymin = max(0, min(ymin, height - 1))
                    xmax = max(xmin + 1, min(xmax, width))
                    ymax = max(ymin + 1, min(ymax, height))

                    crop = rgb_image.crop((xmin, ymin, xmax, ymax))
                    output_name = f"{record.image_path.stem}_{index:02d}.png"
                    output_path = output_dir / split_name / annotation.label / output_name
                    crop.save(output_path)
                    split_counts[split_name][annotation.label] += 1

    metadata = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "seed": seed,
        "splits": {
            split_name: {
                "images": image_counts[split_name],
                "crops": sum(split_counts[split_name].values()),
                "class_counts": dict(split_counts[split_name]),
            }
            for split_name in ("train", "val", "test")
        },
        "classes": CLASS_NAMES,
    }

    metadata_path = output_dir / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


# ---------------------------------------------------------------------------
# Pipeline 2: YOLO object-detection dataset (images + .txt labels)
# ---------------------------------------------------------------------------

def _bbox_to_yolo(
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    img_width: int,
    img_height: int,
) -> tuple[float, float, float, float]:
    """Convert pixel (xmin,ymin,xmax,ymax) → YOLO (cx,cy,w,h) normalised [0,1]."""
    cx = ((xmin + xmax) / 2.0) / img_width
    cy = ((ymin + ymax) / 2.0) / img_height
    w = (xmax - xmin) / img_width
    h = (ymax - ymin) / img_height
    # Clamp to [0, 1] for safety
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    return cx, cy, w, h


def prepare_yolo_dataset(
    dataset_root: str | Path,
    output_dir: str | Path,
    yaml_path: str | Path | None = None,
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    force: bool = False,
) -> dict:
    """Convert a Pascal-VOC annotated dataset to YOLO detection format.

    Output layout::

        output_dir/
          train/
            images/   (symlinks or copies of original images)
            labels/   (<stem>.txt with one YOLO row per object)
          val/
            images/
            labels/
          test/
            images/
            labels/

    Also writes a ``dataset.yaml`` suitable for ``ultralytics.YOLO.train()``.
    """
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)

    if output_dir.exists():
        if force:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise FileExistsError(
                f"{output_dir} already exists and is not empty. Pass force=True to rebuild."
            )

    class_index = {name: i for i, name in enumerate(CLASS_NAMES)}

    records = load_image_records(dataset_root)
    splits = _split_records(records, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)

    split_counts: dict[str, Counter] = defaultdict(Counter)
    image_counts: dict[str, int] = {}

    for split_name, split_records in splits.items():
        image_counts[split_name] = len(split_records)
        images_out = output_dir / split_name / "images"
        labels_out = output_dir / split_name / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        for record in split_records:
            # --- copy image ---
            dest_image = images_out / record.image_path.name
            shutil.copy2(record.image_path, dest_image)

            # --- get image dimensions for normalisation ---
            with Image.open(record.image_path) as img:
                img_w, img_h = img.size

            # --- write YOLO label file ---
            label_lines: list[str] = []
            for ann in record.annotations:
                xmin, ymin, xmax, ymax = ann.bbox
                xmin = max(0, min(xmin, img_w))
                ymin = max(0, min(ymin, img_h))
                xmax = max(xmin + 1, min(xmax, img_w))
                ymax = max(ymin + 1, min(ymax, img_h))

                cx, cy, w, h = _bbox_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
                cls_id = class_index[ann.label]
                label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                split_counts[split_name][ann.label] += 1

            label_file = labels_out / (record.image_path.stem + ".txt")
            label_file.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    # --- write dataset.yaml ---
    if yaml_path is None:
        yaml_path = Path("dataset.yaml")
    yaml_path = Path(yaml_path)
    yaml_content = (
        f"# Auto-generated by prepare_dataset.py --mode yolo\n"
        f"path: {output_dir.resolve()}\n"
        f"train: train/images\n"
        f"val:   val/images\n"
        f"test:  test/images\n"
        f"\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"\n"
        f"names:\n"
    )
    for i, name in enumerate(CLASS_NAMES):
        yaml_content += f"  {i}: {name}\n"
    yaml_path.write_text(yaml_content, encoding="utf-8")

    metadata = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "yaml_path": str(yaml_path),
        "seed": seed,
        "splits": {
            split_name: {
                "images": image_counts[split_name],
                "annotations": sum(split_counts[split_name].values()),
                "class_counts": dict(split_counts[split_name]),
            }
            for split_name in ("train", "val", "test")
        },
        "classes": CLASS_NAMES,
    }

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[prepare_yolo_dataset] YOLO dataset written to: {output_dir}")
    print(f"[prepare_yolo_dataset] dataset.yaml written to: {yaml_path}")
    return metadata
