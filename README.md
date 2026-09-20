# Pareidolia Paradox — Lunar Surface Classification

## Problem
Binary classification of 256×256 grayscale lunar surface images:
- Class 0: Depth — craters, holes, depressions
- Class 1: Rise — mounds, hills, boulders

Key challenge: topographic inversion — same terrain looks
completely different under different Sun angles.

## Team
- Prithviraj Khedekar
- Arnav Pranil Gadge
- VIT Pune (Vishwakarma Institute of Technology)

## Approach

### Sun Angle Encoding
Sun azimuth angle encoded as sin/cos features:
- sin(azimuth) and cos(azimuth) passed alongside image features
- Preserves circular geometry (0° and 360° are same direction)
- Allows model to learn how lighting affects terrain appearance

### Models
- ConvNeXt-Base pretrained on ImageNet-22k (384×384 input)
- Swin-Base Transformer pretrained on ImageNet-22k (384×384 input)

Both models use a dedicated sun angle encoder network
that processes the sin/cos features separately before
combining with image features for final classification.

### Training Strategy
- Weighted random sampling to handle class imbalance
  (Class 1: 5000 samples, Class 0: 2854 samples)
- Weighted cross entropy loss
- Data augmentation: horizontal flip, vertical flip, color jitter
- Early stopping with patience=7
- Separate learning rates: backbone 5e-6, head 1e-4
- CosineAnnealingLR scheduler

### Inference
- Test Time Augmentation (TTA): 6 versions per image
  (original + hflip + vflip + 90° + 180° + 270°)
- Ensemble of both models
- Average probabilities across all versions and models

## Results
Validation Balanced Accuracy: 0.7743

## Code Structure

### train.py
Contains the complete training pipeline:

| Section | Description |
|---------|-------------|
| Imports & Paths | Libraries, dataset paths, device setup |
| Load Data | Read train CSV metadata |
| Dataset | LunarDataset class with sin/cos sun encoding and augmentation |
| Model Definitions | ConvNeXt-Base and Swin-Base with ImageNet-22k weights |
| Data Loaders | Weighted sampler for class imbalance handling |
| Training | Training loop for both models with early stopping |

Output: `best_convnext_in22k.pth` and `best_swin_in22k.pth`

### inference.py
Loads trained weights and generates predictions:

| Section | Description |
|---------|-------------|
| Load Models | Loads both .pth weight files |
| TTA Prediction | 6-version test time augmentation per image |
| Ensemble | Averages probabilities across both models |
| Output | Saves `submission.csv` with 2000 predictions |

## How to Reproduce

### Training
1. Install requirements: `pip install -r requirements.txt`
2. Upload datasets to Kaggle and attach them to a notebook
3. Enable GPU T4 x2 accelerator
4. Update the path variables at the top of `train.py`
5. Run `train.py`
6. Output: `best_convnext_in22k.pth` and `best_swin_in22k.pth`

### Inference
1. Update path variables and weight paths at the top of `inference.py`
2. Run `inference.py`
3. Output: `submission.csv` in `/kaggle/working/`

## Hardware
- Kaggle GPU T4 x2
- Training time: ~3 hours total
- Image size: 384×384
- Batch size: 16
