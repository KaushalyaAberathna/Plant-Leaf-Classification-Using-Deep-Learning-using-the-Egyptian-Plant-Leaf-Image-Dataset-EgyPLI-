"""
evaluate.py
-----------
Step 6: evaluate the trained EfficientNet-B0 model on the held-out test
split -- data that has never influenced training, validation-based
checkpointing, early stopping, or fine-tuning's before/after comparison.

Usage:
    python src/evaluate.py
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import build_dataset, get_or_create_splits
from utils import (
    CONFUSION_MATRIX_DIR,
    MODEL_PATH,
    REPORTS_DIR,
    ensure_output_dirs,
    get_class_names,
)


def load_test_data(batch_size: int = 32):
    """Load the persisted test split (see preprocessing.py) and build a
    non-shuffled tf.data.Dataset from it, so prediction order matches
    test_df row order and true labels can be read straight off the
    DataFrame.
    """
    _train_df, _val_df, test_df = get_or_create_splits()
    class_names = get_class_names()
    num_classes = len(class_names)

    test_ds = build_dataset(test_df, num_classes, batch_size, augment=False, shuffle=False)
    y_true = test_df["label_idx"].values
    return test_ds, y_true, class_names, test_df


def evaluate_model(model_path: str = MODEL_PATH, batch_size: int = 32):
    """Run the full Step 6 evaluation and save every required artifact.

    Returns a dict of the computed metrics for reuse (e.g. by the design
    report generator or a quick console check).
    """
    ensure_output_dirs()

    print(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)

    test_ds, y_true, class_names, test_df = load_test_data(batch_size)
    print(f"Evaluating on {len(test_df)} held-out test images "
          f"({test_df['session_id'].nunique()} photo sessions, never seen during training)")

    y_prob = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report_text = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0, output_dict=True
    )

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION -- EfficientNet-B0")
    print("=" * 60)
    print(f"Accuracy:            {accuracy:.4f}")
    print(f"Macro  Precision/Recall/F1: {macro_precision:.4f} / {macro_recall:.4f} / {macro_f1:.4f}")
    print(f"Weighted Precision/Recall/F1: {weighted_precision:.4f} / {weighted_recall:.4f} / {weighted_f1:.4f}")
    print("-" * 60)
    print(report_text)

    # --- Save classification_report.txt (Step 10 required filename) ---
    report_path = os.path.join(REPORTS_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("EfficientNet-B0 -- Test Set Classification Report\n")
        f.write(f"Test images: {len(test_df)}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(report_text)
    print(f"\nSaved classification report -> {report_path}")

    # --- Save machine-readable metrics summary (for cross-model comparison) ---
    metrics_summary = {
        "model": "EfficientNet-B0",
        "test_images": len(test_df),
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "per_class": {
            name: {
                "precision": report_dict[name]["precision"],
                "recall": report_dict[name]["recall"],
                "f1_score": report_dict[name]["f1-score"],
                "support": report_dict[name]["support"],
            }
            for name in class_names
        },
    }
    metrics_path = os.path.join(REPORTS_DIR, "evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved metrics summary -> {metrics_path}")

    # --- Confusion matrix (Step 10 required image) ---
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=True, values_format="d")
    ax.set_title("EfficientNet-B0 -- Test Set Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(CONFUSION_MATRIX_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix -> {cm_path}")

    return metrics_summary


if __name__ == "__main__":
    evaluate_model()
