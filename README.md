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
`pareidolia_solution.py` contains the complete pipeline divided into 7 sections:

| Section | Description |
|---------|-------------|
| Cell 1 | Imports, library setup, dataset paths |
| Cell 2 | Load train/test CSV metadata |
| Cell 3 | Dataset class with sin/cos sun encoding + TTA function |
| Cell 4 | Model definitions — ConvNeXt-Base and Swin-Base (ImageNet-22k) |
| Cell 5 | Data loaders with weighted sampler for class imbalance |
| Cell 6 | Training loop — both models trained sequentially |
| Cell 7 | TTA ensemble inference — generates submission_timm.csv |

## How to Reproduce
1. Install requirements: `pip install -r requirements.txt`
2. Upload datasets to Kaggle and attach them to a notebook
3. Enable GPU T4 x2 accelerator
4. Update the 4 path variables in Cell 1 to match your dataset paths
5. Run `pareidolia_solution.py` sequentially
6. Output: `submission_timm.csv` in `/kaggle/working/`

## Hardware
- Kaggle GPU T4 x2
- Training time: ~3 hours total
- Image size: 384×384
- Batch size: 16
