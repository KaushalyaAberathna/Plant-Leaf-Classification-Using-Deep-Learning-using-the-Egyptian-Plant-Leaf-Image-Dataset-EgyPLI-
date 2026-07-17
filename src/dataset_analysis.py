"""
dataset_analysis.py
--------------------
Step 1: Dataset Analysis for the EgyPLI plant leaf dataset.

Run directly (`python src/dataset_analysis.py`) to print a summary and
save the class-distribution plot and a sample-image grid to outputs/.
This script only reads the dataset -- it performs no resizing or
augmentation, that belongs to preprocessing.py (Step 2).
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    DATASET_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
    count_images_per_class,
    ensure_output_dirs,
    get_class_names,
    plot_class_distribution,
    plot_sample_images,
)


def analyze_dataset(dataset_dir: str = DATASET_DIR) -> dict:
    """Compute and print the core dataset statistics.

    Returns the counts dict so callers (e.g. a notebook) can reuse it
    without re-scanning the filesystem.
    """
    class_names = get_class_names(dataset_dir)
    counts = count_images_per_class(dataset_dir)
    total_images = sum(counts.values())

    print("=" * 55)
    print("EgyPLI DATASET SUMMARY")
    print("=" * 55)
    print(f"Number of classes : {len(class_names)}")
    print(f"Class names       : {', '.join(class_names)}")
    print(f"Total images      : {total_images}")
    print("-" * 55)
    print(f"{'Class':<15}{'Images':>10}{'Share':>10}")
    for name, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        share = 100 * count / total_images
        print(f"{name:<15}{count:>10}{share:>9.1f}%")
    print("-" * 55)
    imbalance_ratio = max(counts.values()) / min(counts.values())
    print(f"Max/min class imbalance ratio: {imbalance_ratio:.2f}x")
    print("=" * 55)

    return counts


def save_report(counts: dict, path: str = None) -> str:
    """Write the same summary shown on screen to a text file for the report."""
    if path is None:
        path = os.path.join(REPORTS_DIR, "dataset_summary.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    total_images = sum(counts.values())
    lines = [
        "EgyPLI DATASET SUMMARY",
        "=" * 55,
        f"Number of classes : {len(counts)}",
        f"Class names       : {', '.join(sorted(counts.keys()))}",
        f"Total images      : {total_images}",
        "-" * 55,
        f"{'Class':<15}{'Images':>10}{'Share':>10}",
    ]
    for name, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        share = 100 * count / total_images
        lines.append(f"{name:<15}{count:>10}{share:>9.1f}%")
    lines.append("-" * 55)
    imbalance_ratio = max(counts.values()) / min(counts.values())
    lines.append(f"Max/min class imbalance ratio: {imbalance_ratio:.2f}x")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def main():
    ensure_output_dirs()
    counts = analyze_dataset()

    report_path = save_report(counts)
    print(f"\nSaved text summary -> {report_path}")

    dist_path = os.path.join(PLOTS_DIR, "class_distribution.png")
    plot_class_distribution(counts, save_path=dist_path)
    print(f"Saved distribution plot -> {dist_path}")

    samples_path = os.path.join(PLOTS_DIR, "sample_images.png")
    plot_sample_images(save_path=samples_path)
    print(f"Saved sample image grid -> {samples_path}")


if __name__ == "__main__":
    main()
