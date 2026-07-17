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
import shutil
import sys
import time

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from efficientnet_model import (
    DEFAULT_DENSE_UNITS,
    DEFAULT_DROPOUT,
    HEAD_LAYER_NAMES,
    build_model,
)
from preprocessing import DEFAULT_BATCH_SIZE, build_dataset, get_or_create_splits
from utils import (
    MODEL_PATH,
    MODELS_DIR,
    REPORTS_DIR,
    ensure_output_dirs,
    save_classes_json,
    set_global_seed,
)

DEFAULT_EPOCHS = 20
DEFAULT_LEARNING_RATE = 1e-3

FINE_TUNE_CANDIDATE_PATH = os.path.join(MODELS_DIR, "efficientnetb0_finetuned_candidate.keras")
DEFAULT_FINE_TUNE_EPOCHS = 15
DEFAULT_FINE_TUNE_LR = 1e-5
DEFAULT_UNFREEZE_LAYERS = 30


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


def unfreeze_top_backbone_layers(model, num_layers_to_unfreeze: int = DEFAULT_UNFREEZE_LAYERS):
    """Unfreeze the last N backbone layers for fine-tuning, in place.

    Works whether `model` still has its original base_model layers attached
    (right after train_initial) or was reloaded fresh from disk -- either
    way, HEAD_LAYER_NAMES tells backbone and head layers apart. BatchNorm
    layers inside the unfrozen block are deliberately kept frozen: updating
    their running statistics on a small dataset/batch size destabilizes
    training instead of helping it.
    """
    backbone_layers = [l for l in model.layers if l.name not in HEAD_LAYER_NAMES]

    for layer in backbone_layers:
        layer.trainable = False

    for layer in backbone_layers[-num_layers_to_unfreeze:]:
        layer.trainable = not isinstance(layer, BatchNormalization)

    for layer in model.layers:
        if layer.name in HEAD_LAYER_NAMES:
            layer.trainable = True

    return model


def fine_tune(epochs: int = DEFAULT_FINE_TUNE_EPOCHS, batch_size: int = DEFAULT_BATCH_SIZE,
              learning_rate: float = DEFAULT_FINE_TUNE_LR, num_layers_to_unfreeze: int = DEFAULT_UNFREEZE_LAYERS,
              use_class_weight: bool = True, base_model_path: str = MODEL_PATH):
    """Phase 2: unfreeze the top of the backbone and continue training at a
    low learning rate, starting from the Step 4 checkpoint at base_model_path.

    Trains into a separate candidate file rather than overwriting
    base_model_path directly, then compares "before" (the phase-1 best,
    read from training_config_initial.json) against "after" (this phase's
    best) and promotes whichever one actually wins to MODEL_PATH -- fine-
    tuning is not assumed to help; it is verified.
    """
    set_global_seed()
    ensure_output_dirs()

    initial_config_path = os.path.join(REPORTS_DIR, "training_config_initial.json")
    if not os.path.exists(initial_config_path):
        raise FileNotFoundError(
            "training_config_initial.json not found -- run train_initial() (Step 4) first."
        )
    with open(initial_config_path, "r", encoding="utf-8") as f:
        initial_config = json.load(f)
    before_val_accuracy = initial_config["best_val_accuracy"]
    before_val_loss = initial_config["best_val_loss"]

    train_df, val_df, _test_df = get_or_create_splits()
    class_names = sorted(train_df["label"].unique())
    num_classes = len(class_names)

    train_ds = build_dataset(train_df, num_classes, batch_size, augment=True, shuffle=True)
    val_ds = build_dataset(val_df, num_classes, batch_size, augment=False, shuffle=False)

    class_weight = None
    if use_class_weight:
        class_weight = compute_balanced_class_weights(train_df, num_classes)

    print(f"Loading Step 4 checkpoint from {base_model_path}")
    model = tf.keras.models.load_model(base_model_path)

    unfreeze_top_backbone_layers(model, num_layers_to_unfreeze)
    trainable = sum(int(tf.size(w)) for w in model.trainable_weights)
    non_trainable = sum(int(tf.size(w)) for w in model.non_trainable_weights)
    print(f"Unfroze top {num_layers_to_unfreeze} backbone layers "
          f"(trainable params: {trainable:,} / non-trainable: {non_trainable:,})")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    log_path = os.path.join(REPORTS_DIR, "training_log_finetune.csv")
    callbacks = get_callbacks(
        FINE_TUNE_CANDIDATE_PATH, log_path,
        early_stopping_patience=5, reduce_lr_patience=2,
    )

    print(f"\nStarting fine-tuning: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
    start_time = time.time()
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs,
        callbacks=callbacks, class_weight=class_weight, shuffle=False,
    )
    elapsed_minutes = (time.time() - start_time) / 60

    history_path = os.path.join(REPORTS_DIR, "history_finetune.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

    best_epoch_idx = int(np.argmax(history.history["val_accuracy"]))
    after_val_accuracy = history.history["val_accuracy"][best_epoch_idx]
    after_val_loss = history.history["val_loss"][best_epoch_idx]

    fine_tune_wins = after_val_accuracy > before_val_accuracy
    if fine_tune_wins:
        shutil.copyfile(FINE_TUNE_CANDIDATE_PATH, MODEL_PATH)
        winner = "fine_tuned"
    else:
        winner = "initial_frozen_backbone"
    os.remove(FINE_TUNE_CANDIDATE_PATH)

    comparison = {
        "before_fine_tuning": {
            "val_accuracy": before_val_accuracy,
            "val_loss": before_val_loss,
        },
        "after_fine_tuning": {
            "val_accuracy": after_val_accuracy,
            "val_loss": after_val_loss,
            "best_epoch": best_epoch_idx + 1,
            "epochs_run": len(history.history["loss"]),
            "unfrozen_layers": num_layers_to_unfreeze,
            "learning_rate": learning_rate,
        },
        "winner": winner,
        "improvement": after_val_accuracy - before_val_accuracy,
        "training_time_minutes": round(elapsed_minutes, 2),
        "model_saved_to": MODEL_PATH,
    }
    comparison_path = os.path.join(REPORTS_DIR, "finetune_comparison.json")
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nFine-tuning complete in {elapsed_minutes:.1f} min")
    print(f"Before: val_accuracy={before_val_accuracy:.4f}  val_loss={before_val_loss:.4f}")
    print(f"After:  val_accuracy={after_val_accuracy:.4f}  val_loss={after_val_loss:.4f}")
    print(f"Winner: {winner} (improvement: {comparison['improvement']:+.4f})")
    print(f"Comparison -> {comparison_path}")
    print(f"Best model  -> {MODEL_PATH}")

    return model, history, comparison


def parse_args():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 for plant leaf classification")
    parser.add_argument("--stage", choices=["initial", "finetune", "both"], default="both",
                         help="Which phase to run: initial (Step 4), finetune (Step 5), or both in sequence")

    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                         help="Epochs for the initial frozen-backbone phase")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE,
                         help="Learning rate for the initial phase")
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--dense_units", type=int, default=DEFAULT_DENSE_UNITS)
    parser.add_argument("--no_class_weight", action="store_true",
                         help="Disable balanced class weighting")

    parser.add_argument("--fine_tune_epochs", type=int, default=DEFAULT_FINE_TUNE_EPOCHS)
    parser.add_argument("--fine_tune_lr", type=float, default=DEFAULT_FINE_TUNE_LR,
                         help="Learning rate for the fine-tuning phase (should be << initial LR)")
    parser.add_argument("--unfreeze_layers", type=int, default=DEFAULT_UNFREEZE_LAYERS,
                         help="Number of top backbone layers to unfreeze during fine-tuning")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    use_class_weight = not args.no_class_weight

    if args.stage in ("initial", "both"):
        train_initial(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            dropout_rate=args.dropout,
            dense_units=args.dense_units,
            use_class_weight=use_class_weight,
        )

    if args.stage in ("finetune", "both"):
        fine_tune(
            epochs=args.fine_tune_epochs,
            batch_size=args.batch_size,
            learning_rate=args.fine_tune_lr,
            num_layers_to_unfreeze=args.unfreeze_layers,
            use_class_weight=use_class_weight,
        )
