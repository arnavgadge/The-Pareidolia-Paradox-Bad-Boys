# =============================================================
# Pareidolia Paradox — Inference Script
# Team: Prithviraj Khedekar & Arnav Pranil Gadge
# VIT Pune
# Loads trained model weights and generates submission.csv
# =============================================================

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import timm

# =============================================================
# PATHS — update these to match your dataset location
# =============================================================
EVAL_IMG_DIR  = '/kaggle/input/datasets/arnavgadgevit/eval-data/eval_images'
TEST_CSV      = '/kaggle/input/datasets/arnavgadgevit/test-meta/test_metadata.csv'

# Model weights — update paths if needed
WEIGHT_PATHS = {
    'convnext_in22k': '/kaggle/working/best_convnext_in22k.pth',
    'swin_in22k':     '/kaggle/working/best_swin_in22k.pth',
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

# =============================================================
# MODEL DEFINITIONS (must match train.py)
# =============================================================
def build_convnext_in22k():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            backbone = timm.create_model(
                'convnext_base.fb_in22k_ft_in1k_384',
                pretrained=False,
                num_classes=0
            )
            in_features = backbone.num_features
            self.backbone = backbone
            self.sun_encoder = nn.Sequential(
                nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64)
            )
            self.head = nn.Sequential(
                nn.Linear(in_features + 64, 256),
                nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(256, 2)
            )
        def forward(self, img, sun):
            f = self.backbone(img)
            s = self.sun_encoder(sun)
            return self.head(torch.cat([f, s], dim=1))
    return Model()


def build_swin_in22k():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            backbone = timm.create_model(
                'swin_base_patch4_window12_384.ms_in22k_ft_in1k',
                pretrained=False,
                num_classes=0
            )
            in_features = backbone.num_features
            self.backbone = backbone
            self.sun_encoder = nn.Sequential(
                nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64)
            )
            self.head = nn.Sequential(
                nn.Linear(in_features + 64, 256),
                nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(256, 2)
            )
        def forward(self, img, sun):
            f = self.backbone(img)
            s = self.sun_encoder(sun)
            return self.head(torch.cat([f, s], dim=1))
    return Model()


MODEL_BUILDERS = {
    'convnext_in22k': build_convnext_in22k,
    'swin_in22k':     build_swin_in22k,
}

# =============================================================
# TTA PREDICTION FUNCTION
# =============================================================
def tta_predict(model, img_pil, sun_azimuth, device):
    base = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    ])

    versions = [
        img_pil,
        TF.hflip(img_pil),
        TF.vflip(img_pil),
        TF.rotate(img_pil, 90),
        TF.rotate(img_pil, 180),
        TF.rotate(img_pil, 270),
    ]

    sun_rad     = np.deg2rad(sun_azimuth)
    sun_feature = torch.tensor(
        [[np.sin(sun_rad), np.cos(sun_rad)]],
        dtype=torch.float32
    ).to(device)

    probs = []
    model.eval()
    with torch.no_grad():
        for v in versions:
            t    = base(v).unsqueeze(0).to(device)
            out  = model(t, sun_feature)
            prob = torch.softmax(out, dim=1).cpu().numpy()[0]
            probs.append(prob)

    return np.mean(probs, axis=0)


# =============================================================
# LOAD MODELS
# =============================================================
loaded_models = []
for model_name, weight_path in WEIGHT_PATHS.items():
    builder = MODEL_BUILDERS[model_name]
    m = builder().to(DEVICE)
    m.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    m.eval()
    loaded_models.append((model_name, m))
    print(f'Loaded {model_name} from {weight_path}')

# =============================================================
# GENERATE PREDICTIONS
# =============================================================
test_df = pd.read_csv(TEST_CSV)
print(f'\nTest samples: {len(test_df)}')
print('Generating TTA ensemble predictions...')

all_image_ids = []
all_probs     = []

for i, row in test_df.iterrows():
    img_id      = row['image_id']
    sun_azimuth = float(row['sun_azimuth_angle'])

    img_path = os.path.join(EVAL_IMG_DIR, img_id)
    img      = Image.open(img_path).convert('RGB')

    model_probs = []
    for model_name, model in loaded_models:
        prob = tta_predict(model, img, sun_azimuth, DEVICE)
        model_probs.append(prob)

    ensemble_prob = np.mean(model_probs, axis=0)
    all_probs.append(ensemble_prob)
    all_image_ids.append(img_id)

    if (i + 1) % 200 == 0:
        print(f'  {i+1}/{len(test_df)}')

final_preds = [np.argmax(p) for p in all_probs]

# =============================================================
# SAVE SUBMISSION
# =============================================================
submission = pd.DataFrame({
    'image_id': all_image_ids,
    'label':    final_preds
})
submission.to_csv('/kaggle/working/submission.csv', index=False)
print(f'\nSaved {len(submission)} predictions to submission.csv')
print(submission['label'].value_counts())
