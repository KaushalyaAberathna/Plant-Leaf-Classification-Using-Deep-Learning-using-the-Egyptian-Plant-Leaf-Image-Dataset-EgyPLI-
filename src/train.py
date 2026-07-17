"""
train.py
--------
Step 4: training pipeline for EfficientNet-B0 (initial phase -- backbone
frozen, only the classification head trains). Step 5 fine-tuning extends
this by unfreezing part of the backbone and continuing training from the
best checkpoint saved here.

Usage:
    python src/train.py --epochs 20 --batch_size 32 --learning_rate 1e-3
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from efficientnet_model import DEFAULT_DENSE_UNITS, DEFAULT_DROPOUT, build_model
from preprocessing import DEFAULT_BATCH_SIZE, build_dataset, get_or_create_splits
from utils import (
    MODEL_PATH,
    REPORTS_DIR,
    ensure_output_dirs,
    save_classes_json,
    set_global_seed,
)

DEFAULT_EPOCHS = 20
DEFAULT_LEARNING_RATE = 1e-3


def compute_balanced_class_weights(train_df, num_classes) -> dict:
    """Inverse-frequency class weights so the 3.44x imbalance found in
    Step 1 (Orange=547 vs Tomato=159) doesn't bias the model toward the
    majority classes.
    """
    classes = np.arange(num_classes)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=train_df["label_idx"].values
    )
    return {int(c): float(w) for c, w in zip(classes, weights)}


def get_callbacks(checkpoint_path: str, log_path: str,
                   early_stopping_patience: int = 8, reduce_lr_patience: int = 3):
    return [
        ModelCheckpoint(
            checkpoint_path, monitor="val_accuracy", mode="max",
            save_best_only=True, verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss", patience=early_stopping_patience,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=reduce_lr_patience,
            min_lr=1e-7, verbose=1,
        ),
        CSVLogger(log_path, append=False),
    ]


def train_initial(epochs: int = DEFAULT_EPOCHS, batch_size: int = DEFAULT_BATCH_SIZE,
                   learning_rate: float = DEFAULT_LEARNING_RATE,
                   dropout_rate: float = DEFAULT_DROPOUT, dense_units: int = DEFAULT_DENSE_UNITS,
                   use_class_weight: bool = True):
    """Phase 1 training: frozen EfficientNet-B0 backbone, trainable head only.

    Saves the best model to MODEL_PATH, per-epoch CSV logs, a JSON history
    dict (for Step 7 plotting) and a run-config JSON (for the report).
    Returns (model, base_model, history, class_names) so Step 5's
    fine-tuning phase can continue from here without re-loading anything.
    """
    set_global_seed()
    ensure_output_dirs()

    train_df, val_df, _test_df = get_or_create_splits()
    class_names = sorted(train_df["label"].unique())
    num_classes = len(class_names)
    save_classes_json(class_names)

    train_ds = build_dataset(train_df, num_classes, batch_size, augment=True, shuffle=True)
    val_ds = build_dataset(val_df, num_classes, batch_size, augment=False, shuffle=False)

    class_weight = None
    if use_class_weight:
        class_weight = compute_balanced_class_weights(train_df, num_classes)
        print(f"Using balanced class weights: {class_weight}")

    model, base_model = build_model(
        num_classes, dropout_rate=dropout_rate, dense_units=dense_units,
        learning_rate=learning_rate, freeze_backbone=True,
    )

    log_path = os.path.join(REPORTS_DIR, "training_log_initial.csv")
    callbacks = get_callbacks(MODEL_PATH, log_path)

    print(f"\nStarting initial training: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
    start_time = time.time()
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs,
        callbacks=callbacks, class_weight=class_weight, shuffle=False,
    )
    elapsed_minutes = (time.time() - start_time) / 60

    history_path = os.path.join(REPORTS_DIR, "history_initial.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

    # Both metrics below are read from the SAME epoch (the one ModelCheckpoint
    # actually saved, i.e. the val_accuracy argmax) so the report describes
    # one coherent snapshot rather than independently-mined best values.
    best_epoch_idx = int(np.argmax(history.history["val_accuracy"]))
    config = {
        "phase": "initial_frozen_backbone",
        "epochs_requested": epochs,
        "epochs_run": len(history.history["loss"]),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "dropout_rate": dropout_rate,
        "dense_units": dense_units,
        "class_weight_used": use_class_weight,
        "best_epoch": best_epoch_idx + 1,
        "best_val_accuracy": float(history.history["val_accuracy"][best_epoch_idx]),
        "best_val_loss": float(history.history["val_loss"][best_epoch_idx]),
        "training_time_minutes": round(elapsed_minutes, 2),
        "model_saved_to": MODEL_PATH,
    }
    config_path = os.path.join(REPORTS_DIR, "training_config_initial.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\nInitial training complete in {elapsed_minutes:.1f} min")
    print(f"Best epoch: {best_epoch}  val_accuracy={config['best_val_accuracy']:.4f}  val_loss={config['best_val_loss']:.4f}")
    print(f"Model checkpoint -> {MODEL_PATH}")
    print(f"History -> {history_path}")
    print(f"Config  -> {config_path}")
    print(f"Log     -> {log_path}")

    return model, base_model, history, class_names


def parse_args():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 (initial frozen-backbone phase)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--dense_units", type=int, default=DEFAULT_DENSE_UNITS)
    parser.add_argument("--no_class_weight", action="store_true",
                         help="Disable balanced class weighting")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_initial(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout,
        dense_units=args.dense_units,
        use_class_weight=not args.no_class_weight,
    )
