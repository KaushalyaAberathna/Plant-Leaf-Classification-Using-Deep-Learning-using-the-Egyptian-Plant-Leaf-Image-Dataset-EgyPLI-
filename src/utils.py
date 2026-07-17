"""
utils.py
--------
Shared, reusable helpers for the EfficientNet-B0 plant leaf classification
pipeline. Every other script (preprocessing, train, evaluate, predict, app)
imports from here instead of redefining paths or plotting code, so the
project has a single source of truth for constants and common routines.
"""

import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Project paths (all relative to the project root, resolved from this file's
# location so scripts work the same whether run from root or from src/).
# ---------------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")
CONFUSION_MATRIX_DIR = os.path.join(OUTPUTS_DIR, "confusion_matrix")
PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, "predictions")

MODEL_PATH = os.path.join(MODELS_DIR, "efficientnetb0_best.keras")
CLASSES_PATH = os.path.join(MODELS_DIR, "classes.json")

IMG_SIZE = (224, 224)
SEED = 42

# EgyPLI images ship as clean JPEGs; keep the allow-list explicit so any
# stray non-image file in a class folder is skipped rather than crashing
# a training run.
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def set_global_seed(seed: int = SEED) -> None:
    """Seed python/numpy/tensorflow RNGs for reproducible splits and runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def ensure_output_dirs() -> None:
    """Create every output directory if it doesn't already exist."""
    for d in (MODELS_DIR, PLOTS_DIR, REPORTS_DIR, CONFUSION_MATRIX_DIR, PREDICTIONS_DIR):
        os.makedirs(d, exist_ok=True)


def get_class_names(dataset_dir: str = DATASET_DIR) -> list:
    """Return sorted class names, one per subfolder of dataset_dir.

    Sorting makes the class-index <-> class-name mapping deterministic,
    which matters because that mapping is saved to classes.json and must
    stay identical across preprocessing, training and prediction.
    """
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    return sorted(
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    )


def count_images_per_class(dataset_dir: str = DATASET_DIR) -> dict:
    """Return {class_name: image_count} for every class folder."""
    counts = {}
    for class_name in get_class_names(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        counts[class_name] = sum(
            1 for f in os.listdir(class_dir) if f.lower().endswith(VALID_EXTENSIONS)
        )
    return counts


def save_classes_json(class_names: list, path: str = CLASSES_PATH) -> None:
    """Persist the index -> class-name mapping used by train/predict/app.

    Saved as {"0": "Apple", "1": "Berry", ...} so json.load keys are
    directly usable after casting to int, and the file is human-readable.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mapping = {str(i): name for i, name in enumerate(class_names)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)


def load_classes_json(path: str = CLASSES_PATH) -> dict:
    """Load the index -> class-name mapping saved by save_classes_json."""
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return {int(k): v for k, v in mapping.items()}


def plot_class_distribution(counts: dict, save_path: str = None, title: str = "EgyPLI Class Distribution"):
    """Bar chart of image counts per class, sorted descending.

    Sorting (rather than alphabetical order) makes class imbalance visible
    at a glance, which is the whole point of this plot.
    """
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    names, values = zip(*items)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, values, color="#4C9A2A", edgecolor="black")
    ax.set_xlabel("Plant Class")
    ax.set_ylabel("Number of Images")
    ax.set_title(title)
    ax.bar_label(bars, padding=3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    return fig


def plot_sample_images(dataset_dir: str = DATASET_DIR, samples_per_class: int = 1, save_path: str = None):
    """Grid of sample leaf images, one row of samples per class.

    Useful to visually sanity-check that class folders contain what their
    name claims, and to eyeball lighting/background variation before
    deciding on augmentation strength.
    """
    class_names = get_class_names(dataset_dir)
    rng = random.Random(SEED)

    fig, axes = plt.subplots(1, len(class_names), figsize=(3 * len(class_names), 3.5))
    if len(class_names) == 1:
        axes = [axes]

    for ax, class_name in zip(axes, class_names):
        class_dir = os.path.join(dataset_dir, class_name)
        files = [f for f in os.listdir(class_dir) if f.lower().endswith(VALID_EXTENSIONS)]
        chosen = rng.choice(files)
        img = plt.imread(os.path.join(class_dir, chosen))
        ax.imshow(img)
        ax.set_title(class_name, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    return fig


