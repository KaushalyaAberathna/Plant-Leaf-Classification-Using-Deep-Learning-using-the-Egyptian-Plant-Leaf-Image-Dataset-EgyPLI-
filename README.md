# Plant Leaf Classification -- EfficientNet-B0 (EgyPLI)

Deep learning model that classifies plant leaf images into 8 species using
transfer learning on **EfficientNet-B0**, trained on the Egyptian Plant Leaf
Image Dataset (EgyPLI). This is the EfficientNet-B0 component of a larger
model-comparison project (alongside teammates' MobileNetV2 and ResNet50
implementations).

The model targets real-world conditions the dataset itself exhibits --
varying illumination, cluttered natural backgrounds, and partial occlusion
-- rather than clean, studio-lit leaf photos.

## Project Structure

```
Plant-Leaf-Classification/
|
├── dataset/                     # EgyPLI images, one subfolder per class (not tracked in git)
|
├── src/
|   ├── utils.py                 # shared paths/constants, plotting helpers
|   ├── dataset_analysis.py      # Step 1: dataset stats + visualizations
|   ├── preprocessing.py         # Step 2: resize/normalize/augment pipeline + leak-free split
|   ├── efficientnet_model.py    # Step 3: EfficientNet-B0 model definition
|   ├── train.py                 # Step 4+5: initial training + fine-tuning
|   ├── evaluate.py              # Step 6: test-set evaluation
|   ├── visualize_training.py    # Step 7: accuracy/loss curves
|   └── predict.py                # Step 8: prediction module (CLI + reusable Predictor class)
|
├── models/
|   ├── efficientnetb0_best.keras
|   └── classes.json
|
├── outputs/
|   ├── plots/                   # class_distribution, sample_images, augmentation_preview, accuracy, loss
|   ├── reports/                 # dataset summary, model summary, training logs/history, classification_report.txt, splits/
|   ├── confusion_matrix/
|   └── predictions/
|
├── app.py                       # Step 9: Gradio demo
├── requirements.txt
└── README.md
```

## Dataset

**Egyptian Plant Leaf Image Dataset (EgyPLI)** -- 8 plant species, 3,588 JPEG
images (256x256, RGB):

| Class | Images |
|---|---|
| Orange | 547 |
| Persimmon | 527 |
| Guava | 520 |
| Apple | 519 |
| Fig | 508 |
| Palm | 468 |
| Berry | 340 |
| Tomato | 159 |

The dataset is imbalanced (3.44x between the largest and smallest class),
handled during training via balanced `class_weight`s.

**Important methodology note -- burst-photo leakage:** EgyPLI filenames
encode a capture timestamp (`IMG_YYYYMMDD_HHMMSS.jpg`). Inspecting them
shows each physical leaf was photographed in a rapid burst (many frames a
few seconds apart, same leaf/background/angle). A naive random train/test
split would scatter near-duplicate frames of the *same leaf* across splits,
inflating accuracy in a way that would not hold up on genuinely new photos.
`preprocessing.py` instead clusters images into photo "sessions" (gap > 10s
= new session) and keeps every session entirely within one split
(`group_stratified_split`), with an automated leakage check
(`verify_no_session_leakage`) run on every split generation. Final split:
2,500 train / 553 val / 535 test images (~70/15/15), stratified by class,
zero session overlap between splits.

## Installation

```bash
git clone <repo-url>
cd Plant-Leaf-Classification
pip install -r requirements.txt
```

Requires Python 3.9+. Works on CPU; automatically uses a GPU if TensorFlow
detects one (no code changes needed). Compatible with Google Colab.

To retrain from scratch, place the EgyPLI dataset under `dataset/<ClassName>/*.jpg`
(one folder per class). This is **not required** to run prediction or the
demo -- the trained model is already included under `models/`.

## Training

Run dataset analysis (Step 1) and preprocessing self-checks (Step 2) first, if desired:

```bash
python src/dataset_analysis.py
python src/preprocessing.py
```

Train the model (Step 4: initial frozen-backbone phase, then Step 5: fine-tuning):

```bash
python src/train.py --stage both
```

Or run each phase separately:

```bash
python src/train.py --stage initial --epochs 20 --batch_size 32 --learning_rate 1e-3
python src/train.py --stage finetune --fine_tune_epochs 15 --fine_tune_lr 1e-5 --unfreeze_layers 30
```

All hyperparameters are configurable via CLI flags (`--epochs`, `--batch_size`,
`--learning_rate`, `--dropout`, `--dense_units`, `--fine_tune_epochs`,
`--fine_tune_lr`, `--unfreeze_layers`, `--no_class_weight`). Fine-tuning only
overwrites the saved model if it actually beats the initial phase's
validation accuracy (see `outputs/reports/finetune_comparison.json`).

Generate evaluation metrics and training curves:

```bash
python src/evaluate.py
python src/visualize_training.py
```

## Prediction

```bash
python src/predict.py path/to/leaf1.jpg path/to/leaf2.jpg
```

Loads the trained model once, then classifies each image, printing predicted
class, confidence, top-3 predictions, and prediction time. Results are also
logged to `outputs/predictions/`.

## Demo

```bash
python app.py
```

Opens a local Gradio web app (also renders inline automatically in Google
Colab). Upload a leaf image, click **Predict**, and view the predicted
class, confidence, top-3 predictions, and processing time. The model loads
once at startup and stays loaded across any number of predictions -- no
retraining during the demo.

## Model Architecture

```
Input (224, 224, 3)                    pixels in [0, 1]
  -> Rescaling(255.0)                  undoes [0,1] normalization for EfficientNet
  -> EfficientNetB0(weights="imagenet", include_top=False)
       - fully frozen during initial training (Step 4)
       - top 30 layers unfrozen during fine-tuning (Step 5), BatchNorm layers
         kept frozen throughout for training stability
  -> GlobalAveragePooling2D
  -> BatchNormalization
  -> Dropout(0.3)
  -> Dense(256, activation="relu")
  -> Dense(8, activation="softmax")    output
```

Compiled with Adam, categorical crossentropy loss, accuracy metric.
Callbacks: `ModelCheckpoint` (best val_accuracy), `EarlyStopping`
(val_loss, restores best weights), `ReduceLROnPlateau`, `CSVLogger`.

EfficientNetB0 has its own built-in `Rescaling(1/255)` + ImageNet
`Normalization` as the first stage of its architecture -- it expects raw
`[0,255]` pixels, unlike a plain CNN. The compensating `Rescaling(255.0)`
layer cancels out the pipeline's `[0,1]` normalization exactly
((x/255)*255 = x), satisfying both the assignment's normalization
requirement and EfficientNet's actual numerical needs.

## Results

**Training (Step 4 -> Step 5):**

| Phase | Best val_accuracy | val_loss |
|---|---|---|
| Initial (frozen backbone) | 91.68% | 0.314 |
| Fine-tuned (top 30 layers unfrozen) | **92.41%** | 0.298 |

**Test set evaluation (535 held-out images, never seen during training):**

| Metric | Score |
|---|---|
| Accuracy | **94.39%** |
| Macro Precision / Recall / F1 | 0.9425 / 0.9406 / 0.9389 |
| Weighted Precision / Recall / F1 | 0.9494 / 0.9439 / 0.9439 |

Per-class F1 ranges from 1.00 (Guava, Tomato) down to 0.76 (Berry). The
confusion matrix (`outputs/confusion_matrix/confusion_matrix.png`) shows
the main confusions are Apple->Berry (15 images) and Berry->Persimmon (6
images) -- visually similar leaf shapes/colors -- plus a smaller
Palm->Orange leak (5 images). Full breakdown in
`outputs/reports/classification_report.txt`.

## Future Improvements

- Collect more images for underrepresented/weaker classes (Tomato at 159
  images is the smallest; Berry has the lowest F1 at 0.76).
- Try test-time augmentation and/or ensembling with the teammates'
  MobileNetV2/ResNet50 models for a final comparison ensemble.
- Add Grad-CAM visualization to inspect *why* Apple and Berry leaves get
  confused (likely shared color/texture cues) and to build evaluator trust
  in the model's reasoning.
- Fine-tune deeper into the backbone (beyond the top 30 layers) now that a
  leak-free evaluation pipeline exists to measure the effect honestly.
- Expand the dataset with more photo sessions per class (different
  lighting, backgrounds, occlusion levels) to reduce dependence on the
  burst-photo grouping heuristic and improve robustness to real-world
  variation.
- Export to TensorFlow Lite for on-device / mobile field use.
