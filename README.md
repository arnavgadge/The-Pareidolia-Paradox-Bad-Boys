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
- Data augmentation: horizontal flip, vertical flip,
  color jitter
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

## Repository Structure
```
pareidolia-paradox/
├── README.md
├── notebook.ipynb
├── submission_timm.csv
└── requirements.txt
```

## How to Run
1. Install requirements: `pip install -r requirements.txt`
2. Set correct dataset paths in Cell 1
3. Run cells 1-7 sequentially
4. `submission_timm.csv` generated in working directory

## Hardware
- Kaggle GPU T4 x2
- Training time: ~3 hours total
