"""
visualize_training.py
----------------------
Step 7: training visualization. Reads the history JSON files saved by
train.py (Step 4's history_initial.json and Step 5's history_finetune.json)
and plots accuracy/loss across the *whole* training journey -- both phases
concatenated into one timeline with a marker where fine-tuning began.

Kept as its own script (rather than being called inline from train.py) so
plots can be regenerated any time from saved history without retraining.

Usage:
    python src/visualize_training.py
"""

import json
import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import PLOTS_DIR, REPORTS_DIR, ensure_output_dirs


def load_history(phase_name: str):
    """Load a saved history_<phase_name>.json, or None if it doesn't exist
    (e.g. fine-tuning hasn't been run yet).
    """
    path = os.path.join(REPORTS_DIR, f"history_{phase_name}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def combine_histories(initial: dict, finetune: dict = None):
    """Concatenate the initial and fine-tune histories into one continuous
    per-epoch timeline. Returns (combined_dict, boundary_epoch), where
    boundary_epoch is the epoch count after which fine-tuning began (0 if
    there was no fine-tuning phase to plot).
    """
    combined = {key: list(initial[key]) for key in ("accuracy", "val_accuracy", "loss", "val_loss")}
    boundary_epoch = len(initial["accuracy"])

    if finetune is not None:
        for key in ("accuracy", "val_accuracy", "loss", "val_loss"):
            combined[key].extend(finetune[key])

    return combined, boundary_epoch


def _plot_metric_pair(epochs, train_values, val_values, boundary_epoch, ylabel, title, save_path):
    plt.figure(figsize=(9, 5.5))
    plt.plot(epochs, train_values, label=f"Training {ylabel}", marker=".")
    plt.plot(epochs, val_values, label=f"Validation {ylabel}", marker=".")
    if 0 < boundary_epoch < len(epochs):
        plt.axvline(x=boundary_epoch + 0.5, color="gray", linestyle="--",
                    label="Start Fine-Tuning")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_curves(combined: dict, boundary_epoch: int, save_dir: str = PLOTS_DIR):
    """Save accuracy.png and loss.png (Step 10's required filenames)."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = list(range(1, len(combined["accuracy"]) + 1))

    _plot_metric_pair(
        epochs, combined["accuracy"], combined["val_accuracy"], boundary_epoch,
        ylabel="Accuracy", title="EfficientNet-B0 -- Training vs Validation Accuracy",
        save_path=os.path.join(save_dir, "accuracy.png"),
    )
    _plot_metric_pair(
        epochs, combined["loss"], combined["val_loss"], boundary_epoch,
        ylabel="Loss", title="EfficientNet-B0 -- Training vs Validation Loss",
        save_path=os.path.join(save_dir, "loss.png"),
    )


def main():
    ensure_output_dirs()

    initial = load_history("initial")
    if initial is None:
        raise FileNotFoundError(
            "history_initial.json not found -- run train.py (Step 4) before visualizing."
        )
    finetune = load_history("finetune")

    combined, boundary_epoch = combine_histories(initial, finetune)
    plot_curves(combined, boundary_epoch)

    total_epochs = len(combined["accuracy"])
    finetune_epochs = total_epochs - boundary_epoch
    print(f"Plotted {total_epochs} total epochs "
          f"(initial: {boundary_epoch}, fine-tune: {finetune_epochs})")
    print(f"Saved -> {os.path.join(PLOTS_DIR, 'accuracy.png')}")
    print(f"Saved -> {os.path.join(PLOTS_DIR, 'loss.png')}")


if __name__ == "__main__":
    main()
