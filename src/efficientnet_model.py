"""
efficientnet_model.py
----------------------
Step 3: EfficientNet-B0 transfer learning model -- the proposed
architecture for this project.

Architecture:
  Input (224,224,3), pixels in [0,1] from preprocessing.py
    -> Rescaling(255.0)                         # see note below
    -> EfficientNetB0(weights="imagenet", include_top=False)   # frozen
    -> GlobalAveragePooling2D
    -> BatchNormalization
    -> Dropout
    -> Dense (ReLU)
    -> Dense (Softmax output, num_classes units)

Why Rescaling(255.0) here:
  preprocessing.py deliberately normalizes images to [0,1] (per the
  assignment spec, and to stay reusable for teammates' models). But
  EfficientNetB0 has its own built-in Rescaling(1/255) + ImageNet
  Normalization as the first stage of its architecture -- it expects raw
  [0,255] pixels. Feeding it [0,1] input directly would rescale twice,
  crushing pixel values to near-zero and destroying accuracy. Multiplying
  back by 255 here cancels out exactly: (x/255)*255 = x, so this model
  effectively receives the original pixel values, correctly normalized
  internally by EfficientNet itself.
"""

import os
import sys

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import IMG_SIZE

DEFAULT_DROPOUT = 0.30
DEFAULT_DENSE_UNITS = 256
DEFAULT_LEARNING_RATE = 1e-3

# Names of the custom classification head layers added in build_model(), used
# by train.py's fine-tuning phase to tell backbone layers apart from head
# layers after reloading a saved model from disk (where base_model is no
# longer a separate nested object -- see efficientnet_model.py's build_model
# input_tensor usage, which inlines EfficientNetB0's layers into the outer
# functional graph).
HEAD_LAYER_NAMES = {"gap", "head_bn", "head_dropout", "head_dense", "predictions"}


def build_model(num_classes: int, img_size=IMG_SIZE, dropout_rate: float = DEFAULT_DROPOUT,
                 dense_units: int = DEFAULT_DENSE_UNITS, learning_rate: float = DEFAULT_LEARNING_RATE,
                 freeze_backbone: bool = True):
    """Build and compile the EfficientNet-B0 transfer learning model.

    Returns (model, base_model) -- base_model is returned separately so
    train.py can freeze/unfreeze its layers directly in Step 5 (fine-tuning)
    without having to search for it inside the functional graph.
    """
    inputs = layers.Input(shape=img_size + (3,), name="input_image")
    x = layers.Rescaling(255.0, name="undo_0_1_normalization")(inputs)

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
        pooling=None,
    )
    base_model.trainable = not freeze_backbone

    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    x = layers.Dense(dense_units, activation="relu", name="head_dense")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="EfficientNetB0_PlantLeaf")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model, base_model


def count_trainable_params(model) -> tuple:
    """Return (trainable_count, non_trainable_count) for quick sanity checks."""
    trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    non_trainable = sum(tf.size(w).numpy() for w in model.non_trainable_weights)
    return int(trainable), int(non_trainable)


if __name__ == "__main__":
    from utils import REPORTS_DIR, ensure_output_dirs

    ensure_output_dirs()

    NUM_CLASSES = 8  # matches the 8 EgyPLI classes found in Step 1
    model, base_model = build_model(NUM_CLASSES)

    trainable, non_trainable = count_trainable_params(model)
    print(f"Base model (EfficientNetB0) layers: {len(base_model.layers)}")
    print(f"Base model frozen: {not base_model.trainable}")
    print(f"Trainable params:     {trainable:,}")
    print(f"Non-trainable params: {non_trainable:,}")

    summary_path = os.path.join(REPORTS_DIR, "model_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))
        f.write(f"\nBase model (EfficientNetB0) layers: {len(base_model.layers)}\n")
        f.write(f"Base model frozen: {not base_model.trainable}\n")
        f.write(f"Trainable params:     {trainable:,}\n")
        f.write(f"Non-trainable params: {non_trainable:,}\n")
    print(f"Saved model summary -> {summary_path}")
