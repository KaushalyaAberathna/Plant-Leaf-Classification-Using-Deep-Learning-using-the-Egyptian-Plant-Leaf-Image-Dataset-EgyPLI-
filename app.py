"""
app.py
------
Step 9: Gradio web demo for the EfficientNet-B0 plant leaf classifier.

The trained model is loaded exactly once at startup (via predict.Predictor)
and reused for every prediction click -- no retraining, no reloading,
so the evaluator can test image after image with each click staying fast.

Usage:
    python app.py
Then open the printed local URL in a browser.
"""

import os
import sys

import gradio as gr

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from predict import Predictor  # noqa: E402

predictor = Predictor()  # loaded once when the app starts


def classify(image):
    """Gradio callback: run one prediction and format each required field
    as its own output (predicted class, confidence, top-3, timing).
    """
    if image is None:
        return "No image uploaded", "-", "-", "-"

    try:
        result = predictor.predict(image, top_k=3)
    except Exception as exc:
        return f"Error: {exc}", "-", "-", "-"

    predicted_class = result["predicted_class"]
    confidence = f"{result['confidence_percent']}%"
    top_predictions = "\n".join(
        f"{i}. {p['class']} - {p['confidence_percent']}%"
        for i, p in enumerate(result["top_predictions"], start=1)
    )
    processing_time = f"{result['prediction_time_seconds']} seconds"

    return predicted_class, confidence, top_predictions, processing_time


with gr.Blocks(title="EgyPLI Plant Leaf Classifier - EfficientNet-B0") as demo:
    gr.Markdown(
        "# Plant Leaf Classification -- EfficientNet-B0\n"
        "Upload a leaf image and click **Predict**. The model classifies it into one "
        f"of {len(predictor.class_names)} plant species from the EgyPLI dataset: "
        f"{', '.join(predictor.class_names.values())}."
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Leaf Image")
            predict_btn = gr.Button("Predict", variant="primary")
        with gr.Column():
            predicted_class_output = gr.Textbox(label="Predicted Plant Class")
            confidence_output = gr.Textbox(label="Confidence")
            top_predictions_output = gr.Textbox(label="Top 3 Predictions", lines=3)
            processing_time_output = gr.Textbox(label="Processing Time")

    predict_btn.click(
        fn=classify,
        inputs=image_input,
        outputs=[predicted_class_output, confidence_output, top_predictions_output, processing_time_output],
    )

    gr.Markdown(
        "---\n"
        "Model: EfficientNet-B0 (ImageNet transfer learning, fine-tuned) | "
        "Dataset: Egyptian Plant Leaf Image Dataset (EgyPLI)"
    )


if __name__ == "__main__":
    demo.launch()
