"""Utilities for converting XML annotations into a cropped-face dataset."""

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


@dataclass(frozen=True)
class FaceAnnotation:
    label: str
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class ImageRecord:
    image_path: Path
    annotations: tuple[FaceAnnotation, ...]


def _safe_int(text: str | None, default: int = 0) -> int:
    if text is None:
        return default
    return int(float(text))


def load_image_records(dataset_root: str | Path) -> list[ImageRecord]:
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

