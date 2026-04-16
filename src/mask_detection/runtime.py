"""Runtime helpers shared across scripts."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .model import MaskClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(12),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def checkpoint_to_jsonable(checkpoint: dict) -> dict:
    jsonable = dict(checkpoint)
    jsonable.pop("model_state_dict", None)
    jsonable.pop("optimizer_state_dict", None)
    return jsonable


def load_checkpoint(checkpoint_path: str | Path, map_location: torch.device | str = "cpu"):
    return torch.load(checkpoint_path, map_location=map_location)


def build_model_from_checkpoint(checkpoint: dict) -> MaskClassifier:
    class_names = checkpoint["class_names"]
    model = MaskClassifier(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def preprocess_pil_image(image: Image.Image, image_size: int) -> torch.Tensor:
    transform = build_transforms(image_size=image_size, train=False)
    return transform(image.convert("RGB")).unsqueeze(0)


def save_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
