"""
predict.py
----------
Step 8: prediction module. Loads the trained EfficientNet-B0 model once and
classifies leaf images -- no retraining, no re-loading per image. Used
directly by app.py (Step 9) for the Gradio demo, and runnable standalone
from the command line for the evaluator's lab test.

Usage:
    python src/predict.py path/to/leaf1.jpg path/to/leaf2.jpg ...
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import IMG_SIZE, preprocess_for_inference
from utils import CLASSES_PATH, MODEL_PATH, PREDICTIONS_DIR, ensure_output_dirs, load_classes_json


class Predictor:
    """Loads the trained model + class labels once and serves predictions.

    Kept as a class (rather than a load-model-per-call function) so both
    this script's CLI loop and app.py's Gradio callback share one loaded
    model instance across many predictions.
    """

    def __init__(self, model_path: str = MODEL_PATH, classes_path: str = CLASSES_PATH):
        print(f"Loading model from {model_path} ...")
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = load_classes_json(classes_path)  # {0: "Apple", ...}
        self._warm_up()
        print(f"Model ready. Classes: {list(self.class_names.values())}")

    def _warm_up(self):
        """Run one dummy prediction so Keras' first-call graph tracing
        overhead doesn't inflate the timing of the evaluator's first
        real image.
        """
        dummy = tf.zeros((1,) + IMG_SIZE + (3,), dtype=tf.float32)
        self.model.predict(dummy, verbose=0)

    def predict(self, image, top_k: int = 3) -> dict:
        """Classify a single image (filepath / PIL.Image / numpy array).

        Returns predicted class, confidence, top-k predictions and the
        measured inference time -- exactly the fields Step 8 requires.
        """
        tensor = preprocess_for_inference(image)

        start = time.perf_counter()
        probs = self.model.predict(tensor, verbose=0)[0]
        elapsed_seconds = time.perf_counter() - start

        top_indices = np.argsort(probs)[::-1][:top_k]
        top_predictions = [
            {"class": self.class_names[int(i)], "confidence_percent": round(float(probs[i]) * 100, 2)}
            for i in top_indices
        ]

        return {
            "predicted_class": top_predictions[0]["class"],
            "confidence_percent": top_predictions[0]["confidence_percent"],
            "top_predictions": top_predictions,
            "prediction_time_seconds": round(elapsed_seconds, 4),
        }


def format_result(image_path: str, result: dict) -> str:
    """Render a prediction result in the exact style the assignment's
    example output specifies.
    """
    lines = [
        "-" * 50,
        f"Image: {os.path.basename(image_path)}",
        "-" * 50,
        "Prediction:",
        result["predicted_class"],
        "",
        "Confidence:",
        f"{result['confidence_percent']}%",
        "",
        "Top predictions:",
    ]
    for i, pred in enumerate(result["top_predictions"], start=1):
        lines.append(f"{i}. {pred['class']} - {pred['confidence_percent']}%")
    lines.append("")
    lines.append(f"Prediction time: {result['prediction_time_seconds']} seconds")
    lines.append("-" * 50)
    return "\n".join(lines)


def save_results(image_paths, results, out_dir: str = PREDICTIONS_DIR) -> str:
    """Persist a JSON record of a prediction run to outputs/predictions/."""
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"predictions_{timestamp}.json")

    payload = [
        {"image": path, **result}
        for path, result in zip(image_paths, results)
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description="Predict plant species from leaf image(s)")
    parser.add_argument("images", nargs="+", help="Path(s) to one or more leaf images")
    parser.add_argument("--top_k", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ensure_output_dirs()

    predictor = Predictor()

    results = []
    for image_path in args.images:
        if not os.path.exists(image_path):
            print(f"\nSkipping (file not found): {image_path}")
            continue
        result = predictor.predict(image_path, top_k=args.top_k)
        results.append(result)
        print()
        print(format_result(image_path, result))

    if results:
        saved_path = save_results(args.images[:len(results)], results)
        print(f"\nSaved prediction log -> {saved_path}")
