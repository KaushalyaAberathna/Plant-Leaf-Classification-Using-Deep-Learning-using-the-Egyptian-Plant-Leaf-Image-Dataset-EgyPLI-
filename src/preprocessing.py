"""
preprocessing.py
-----------------
Step 2: reusable image preprocessing pipeline for the EfficientNet-B0
plant leaf classifier.

Pipeline stages (shared by train.py, evaluate.py and predict.py):
  1. Build a (filepath, label) table from the dataset folder.
  2. Stratified train/val/test split (70/15/15), persisted to CSV so every
     script -- and every re-run -- uses the *same* test set.
  3. tf.data.Dataset that decodes each JPEG, resizes to 224x224 and scales
     pixels to [0, 1].
  4. A Keras augmentation block (rotation, flip, zoom, width/height shift,
     brightness) applied only to the training split.

Note on EfficientNet-B0 and [0,1] normalization
------------------------------------------------
EfficientNetB0 (tf.keras.applications) already contains its own internal
Rescaling(1/255) + ImageNet Normalization layers as the first stage of the
architecture -- it expects raw [0,255] pixel input, unlike a plain CNN.
To satisfy the assignment's requirement that this pipeline normalize
pixels to [0,1] *and* remain numerically correct for EfficientNet,
efficientnet_model.py inserts a compensating `Rescaling(255.0)` layer
immediately before the EfficientNetB0 base. Net effect: (x/255)*255 = x,
identical to feeding EfficientNet raw pixels directly -- so this
pipeline's [0,1] output stays safe to reuse for other backbones
(MobileNetV2 / ResNet50), each of which should apply its own
model-specific adapter the same way.
"""

import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    DATASET_DIR,
    IMG_SIZE,
    REPORTS_DIR,
    SEED,
    VALID_EXTENSIONS,
    get_class_names,
)

SPLITS_DIR = os.path.join(REPORTS_DIR, "splits")
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
DEFAULT_BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# 1. File table + stratified split
# ---------------------------------------------------------------------------
def build_dataframe(dataset_dir: str = DATASET_DIR) -> pd.DataFrame:
    """Walk dataset_dir and return a DataFrame with filepath/label/label_idx.

    label_idx uses the same sorted-class-name ordering as utils.get_class_names,
    so it matches classes.json exactly once that is saved in Step 4.
    """
    class_names = get_class_names(dataset_dir)
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    rows = []
    for class_name in class_names:
        class_dir = os.path.join(dataset_dir, class_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(VALID_EXTENSIONS):
                rows.append({
                    "filepath": os.path.join(class_dir, fname),
                    "label": class_name,
                    "label_idx": class_to_idx[class_name],
                })
    return pd.DataFrame(rows)


def stratified_split(df: pd.DataFrame, train_ratio: float = TRAIN_RATIO,
                      val_ratio: float = VAL_RATIO, test_ratio: float = TEST_RATIO,
                      seed: int = SEED):
    """Split df into train/val/test, preserving per-class proportions.

    Stratifying matters here because EgyPLI is imbalanced (3.44x); a
    non-stratified split could leave the smallest class (Tomato, 159
    images) under-represented in the test set, making its precision/recall
    unreliable in Step 6 evaluation.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "ratios must sum to 1"

    train_df, temp_df = train_test_split(
        df, test_size=(val_ratio + test_ratio), stratify=df["label"], random_state=seed
    )
    relative_test_size = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test_size, stratify=temp_df["label"], random_state=seed
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_splits(train_df, val_df, test_df, splits_dir: str = SPLITS_DIR) -> None:
    os.makedirs(splits_dir, exist_ok=True)
    train_df.to_csv(os.path.join(splits_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(splits_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(splits_dir, "test.csv"), index=False)


def load_splits(splits_dir: str = SPLITS_DIR):
    """Load previously saved splits, or return None if they don't exist yet."""
    paths = {name: os.path.join(splits_dir, f"{name}.csv") for name in ("train", "val", "test")}
    if not all(os.path.exists(p) for p in paths.values()):
        return None
    return tuple(pd.read_csv(paths[name]) for name in ("train", "val", "test"))


def get_or_create_splits(dataset_dir: str = DATASET_DIR, splits_dir: str = SPLITS_DIR,
                          force_new: bool = False):
    """Load existing train/val/test CSVs, or build+save them if absent.

    This is the single entry point train.py/evaluate.py should call --
    it guarantees every script sees the same split without re-deriving it.
    """
    if not force_new:
        existing = load_splits(splits_dir)
        if existing is not None:
            return existing

    df = build_dataframe(dataset_dir)
    train_df, val_df, test_df = stratified_split(df)
    save_splits(train_df, val_df, test_df, splits_dir)
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# 2. tf.data pipeline
# ---------------------------------------------------------------------------
def _decode_and_resize(path: tf.Tensor) -> tf.Tensor:
    """Read a JPEG from disk, resize to IMG_SIZE, scale to [0,1] float32."""
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def get_augmentation_layer() -> tf.keras.Sequential:
    """Keras augmentation block covering every transform the spec requires.

    Applied only to the training split (never val/test), and only while
    `training=True`, so it is automatically inert during evaluation and
    prediction even if accidentally left inside a shared model graph.
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),                                  # horizontal flip
        layers.RandomRotation(0.10),                                      # rotation (~+-36 deg)
        layers.RandomZoom(0.15),                                          # zoom
        layers.RandomTranslation(height_factor=0.10, width_factor=0.10),  # height/width shift
        layers.RandomBrightness(0.20, value_range=(0.0, 1.0)),            # brightness variation
    ], name="augmentation")


def build_dataset(df: pd.DataFrame, num_classes: int, batch_size: int = DEFAULT_BATCH_SIZE,
                   augment: bool = False, shuffle: bool = False) -> tf.data.Dataset:
    """Turn a (filepath, label_idx) DataFrame into a batched tf.data.Dataset
    of (image[0,1], one_hot_label) pairs, ready to pass to model.fit/.evaluate.
    """
    paths = df["filepath"].values
    labels = df["label_idx"].values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)

    def _load(path, label):
        image = _decode_and_resize(path)
        one_hot = tf.one_hot(label, num_classes)
        return image, one_hot

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        aug_layer = get_augmentation_layer()
        ds = ds.map(lambda x, y: (aug_layer(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def get_datasets(dataset_dir: str = DATASET_DIR, batch_size: int = DEFAULT_BATCH_SIZE,
                  splits_dir: str = SPLITS_DIR):
    """Convenience wrapper: returns (train_ds, val_ds, test_ds, class_names)
    built from the persisted split, ready for train.py to consume directly.
    """
    class_names = get_class_names(dataset_dir)
    num_classes = len(class_names)
    train_df, val_df, test_df = get_or_create_splits(dataset_dir, splits_dir)

    train_ds = build_dataset(train_df, num_classes, batch_size, augment=True, shuffle=True)
    val_ds = build_dataset(val_df, num_classes, batch_size, augment=False, shuffle=False)
    test_ds = build_dataset(test_df, num_classes, batch_size, augment=False, shuffle=False)

    return train_ds, val_ds, test_ds, class_names


# ---------------------------------------------------------------------------
# 3. Single-image preprocessing (reused by predict.py and app.py)
# ---------------------------------------------------------------------------
def preprocess_for_inference(image) -> tf.Tensor:
    """Convert a filepath / PIL.Image / numpy array into a (1,224,224,3)
    float32 tensor scaled to [0,1] -- the exact same transform training
    images go through (minus augmentation), so inference matches training.
    """
    if isinstance(image, str):
        raw = tf.io.read_file(image)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    elif isinstance(image, Image.Image):
        img = tf.convert_to_tensor(np.array(image.convert("RGB")))
    elif isinstance(image, np.ndarray):
        arr = image[..., :3] if image.shape[-1] == 4 else image
        img = tf.convert_to_tensor(arr)
    else:
        raise TypeError(f"Unsupported image input type: {type(image)}")

    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)
    return img


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from utils import PLOTS_DIR, ensure_output_dirs

    ensure_output_dirs()

    print("Building stratified train/val/test split...")
    train_df, val_df, test_df = get_or_create_splits(force_new=True)
    print(f"  train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")
    print(f"  saved -> {SPLITS_DIR}")

    class_names = get_class_names()
    num_classes = len(class_names)

    # Visual sanity check: original vs. augmented versions of one training image
    sample_path = train_df.iloc[0]["filepath"]
    sample_label = train_df.iloc[0]["label"]
    original = _decode_and_resize(tf.constant(sample_path))
    aug_layer = get_augmentation_layer()

    fig, axes = plt.subplots(1, 6, figsize=(18, 3.5))
    axes[0].imshow(original.numpy())
    axes[0].set_title(f"Original\n({sample_label})")
    axes[0].axis("off")
    for i in range(1, 6):
        augmented = aug_layer(tf.expand_dims(original, 0), training=True)[0]
        axes[i].imshow(augmented.numpy())
        axes[i].set_title(f"Augmented {i}")
        axes[i].axis("off")
    plt.tight_layout()
    preview_path = os.path.join(PLOTS_DIR, "augmentation_preview.png")
    plt.savefig(preview_path, dpi=150)
    print(f"Saved augmentation preview -> {preview_path}")

    # Verify the batched tf.data pipeline produces correctly shaped tensors
    train_ds, val_ds, test_ds, _ = get_datasets()
    for images, labels in train_ds.take(1):
        print(f"Batch image shape: {images.shape}, dtype: {images.dtype}, "
              f"pixel range: [{tf.reduce_min(images):.3f}, {tf.reduce_max(images):.3f}]")
        print(f"Batch label shape: {labels.shape}")
