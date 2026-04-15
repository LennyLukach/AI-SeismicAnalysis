# AI-SeismicAnalysis

An end-to-end deep learning pipeline for automated fault detection in 2D marine seismic sections. The system combines classical signal processing for feature extraction with a Faster R-CNN object detector to identify and localize geological faults.

---

## Overview

Detecting faults in seismic data is traditionally a manual, time-intensive process. This project automates it using a three-stage pipeline:

1. **Feature extraction** — classical seismic attributes (NCC coherence, horizontal gradients, structure tensor, dip variance) are combined into a composite fault probability map
2. **Fault path tracing** — Simulated Annealing optimizes fault paths through the probability field to find geologically plausible traces
3. **CNN detection** — a fine-tuned Faster R-CNN (ResNet-50 + FPN backbone) trained on pipeline-generated annotations localizes faults with bounding boxes

Grad-CAM visualization is included to inspect what the model is actually attending to.

---

## Repository Structure

```
.
├── cnn/
│   ├── src/
│   │   ├── train_2d.py        # Model training script
│   │   └── predict_all.py     # Run inference on a set of images
│   └── models/
│       └── alpha/             # Trained model weights + training info
│           ├── alpha.pth
│           └── training_info.txt
├── pipeline/
│   ├── src/
│   │   ├── pipeline.py        # Main pipeline entry point
│   │   └── fault_detection.py # Feature extraction + SA fault tracing
│   └── results/               # Timestamped output runs
│       └── run_YYYYMMDD_HHMMSS/
│           ├── images/        # Rendered seismic section PNGs
│           ├── annotations/   # COCO-format JSON annotations
│           └── run.log
├── gradcam/
│   └── src/
│       └── gradcam.py         # EigenCAM visualization for detections
└── figures/                   # Sample outputs and CSVs
```

---

## Model

The current model is **alpha**, a Faster R-CNN with a ResNet-50 + FPN backbone.

| Metric     | Value |
|------------|-------|
| Epochs     | 90    |
| Best F1    | 0.596 |
| Precision  | 0.566 |
| Recall     | 0.625 |
| TP / FP / FN | 278 / 213 / 167 |
| Backbone   | ResNet-50 + FPN |
| Batch size | 4     |
| Learning rate | 0.0005 |

Faster R-CNN was chosen over YOLO/SSD because faults are small relative to the full seismic image — two-stage detectors handle small objects better. The FPN provides feature maps at multiple scales to catch faults ranging from ~20px to ~200px.

---

## Installation

```bash
pip install torch torchvision obspy scipy numpy pandas matplotlib pillow pytorch-grad-cam
```

---

## Usage

### Step 1 — Run the pipeline (feature extraction + annotation generation)

Edit the configuration block at the top of `pipeline/src/pipeline.py`:

```python
LINE = None                        # Set to a line name, or None to process all SGY files
AIRGUN_DATA_ROOT = "../../airgun_data"  # Path to your SGY seismic data
USE_AVERAGED_SA = False            # Average multiple SA runs per seed (slower, more stable)
```

Then run:

```bash
cd pipeline/src
python pipeline.py
```

Results are saved to a timestamped folder under `pipeline/results/`.

### Step 2 — Train the CNN

```bash
cd cnn/src
python train_2d.py \
    --data_dir ../../pipeline/merged \
    --images_dir ../../pipeline/results/run_YYYYMMDD_HHMMSS/images

# Resume from a checkpoint:
python train_2d.py --data_dir ./data --resume checkpoints/best_model.pth
```

Expected data structure:
```
data/
├── images/
│   └── T2010.100.mig.1500.png
├── train.json
└── val.json
```

### Step 3 — Run inference

Edit the paths at the top of `cnn/src/predict_all.py` to point to your images and model weights, then:

```bash
cd cnn/src
python predict_all.py
```

The confidence threshold can be adjusted:
```python
config["score_threshold"] = 0.6  # Lower = more detections, higher = fewer but more confident
```

### Step 4 — Visualize with Grad-CAM

```bash
cd gradcam/src
python gradcam.py
```

This runs EigenCAM on the predictions and saves heatmap overlays to `gradcam/results/`. EigenCAM is used over standard GradCAM because it doesn't require a classification target — it works directly from feature activations, which suits object detection models better.

---

## Configuration

Key parameters in `pipeline/src/pipeline.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NCC_FAULT_THRESH` | 0.6 | NCC coherence threshold for fault probability |
| `BANDPASS_LOW_HZ` | 5.0 | Low cut for bandpass filter |
| `BANDPASS_HIGH_HZ` | 200.0 | High cut for bandpass filter |
| `USE_AVERAGED_SA` | False | Average multiple SA runs per seed |
| `SA_RUNS` | 5 | Number of SA runs to average (if enabled) |
| `RANDOM_SEED` | 50 | Seed for single SA run reproducibility |

Key parameters in `pipeline/src/fault_detection.py` (`FAULT_CONFIG`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prob_threshold` | 0.45 | Minimum composite probability to consider a pixel |
| `min_fault_length_frac` | 0.15 | Minimum fault length as fraction of image height |
| `merge_distance_traces` | 100 | Merge faults within this many traces of each other |
| `sa_initial_temp` | 5.0 | SA starting temperature |
| `sa_cooling_rate` | 0.995 | SA cooling schedule |

---

## Data Format

The pipeline reads **SEG-Y (`.sgy`) files** from the `AIRGUN_DATA_ROOT` directory using ObsPy. It outputs:

- PNG images of each seismic section
- COCO-format JSON annotation files (`train.json`, `val.json`) for CNN training
- CSV files with detected fault locations

Every 5th line (sorted alphabetically) is assigned to the validation set; the rest go to training.