"""Keras model definitions."""

from __future__ import annotations

import tensorflow as tf

def _conv_block(filters: int) -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(filters, kernel_size=3, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=2),
        ]
    )


def build_mask_classifier(image_size: int, num_classes: int) -> tf.keras.Model:
    """Build a lightweight CNN for three-way face-mask classification."""

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))

    # Keep augmentation in the model so the CLI scripts share the same pipeline.
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.035)(x)
    x = tf.keras.layers.RandomZoom(height_factor=0.08, width_factor=0.08)(x)

    x = _conv_block(32)(x)
    x = _conv_block(64)(x)
    x = _conv_block(128)(x)
    x = _conv_block(256)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="mask_classifier")
