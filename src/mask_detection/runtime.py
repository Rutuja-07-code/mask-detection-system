"""Runtime helpers shared across scripts."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def select_device() -> str:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        return f"GPU ({gpus[0].name})"
    return "CPU"


def normalize_images(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels


def build_image_datasets(
    data_dir: str | Path,
    image_size: int,
    batch_size: int,
    seed: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    data_dir = Path(data_dir)
    dataset_kwargs = {
        "label_mode": "int",
        "image_size": (image_size, image_size),
        "batch_size": batch_size,
    }

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir / "train",
        shuffle=True,
        seed=seed,
        **dataset_kwargs,
    )
    class_names = list(train_dataset.class_names)

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir / "val",
        shuffle=False,
        **dataset_kwargs,
    )
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir / "test",
        shuffle=False,
        **dataset_kwargs,
    )

    autotune = tf.data.AUTOTUNE
    train_dataset = train_dataset.map(normalize_images, num_parallel_calls=autotune).prefetch(autotune)
    val_dataset = val_dataset.map(normalize_images, num_parallel_calls=autotune).prefetch(autotune)
    test_dataset = test_dataset.map(normalize_images, num_parallel_calls=autotune).prefetch(autotune)

    return train_dataset, val_dataset, test_dataset, class_names


def checkpoint_to_jsonable(checkpoint: dict) -> dict:
    return dict(checkpoint)


def metadata_path_for_model(model_path: str | Path) -> Path:
    model_path = Path(model_path)
    return model_path.with_name(f"{model_path.stem}_metadata.json")


def load_checkpoint(checkpoint_path: str | Path) -> dict:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = json.loads(metadata_path_for_model(checkpoint_path).read_text(encoding="utf-8"))
    checkpoint["model_path"] = str(checkpoint_path.resolve())
    return checkpoint


def build_model_from_checkpoint(checkpoint: dict) -> tf.keras.Model:
    return tf.keras.models.load_model(checkpoint["model_path"])


def preprocess_pil_image(image: Image.Image, image_size: int) -> np.ndarray:
    resized = image.convert("RGB").resize((image_size, image_size))
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def save_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
