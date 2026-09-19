# =============================================================
# Pareidolia Paradox — Lunar Surface Classification
# Team: Prithviraj Khedekar & Arnav Pranil Gadge
# VIT Pune
# Validation Balanced Accuracy: 0.7743
# =============================================================

# =============================================================
# CELL 1 — Imports and Paths
# =============================================================

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.models as models
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

TRAIN_IMG_DIR = '/kaggle/input/datasets/arnavgadgevit/train-images/train_images'
EVAL_IMG_DIR  = '/kaggle/input/datasets/arnavgadgevit/eval-data/eval_images'
TRAIN_CSV     = '/kaggle/input/datasets/arnavgadgevit/train-meta/train_metadata.csv'
TEST_CSV      = '/kaggle/input/datasets/arnavgadgevit/test-meta/test_metadata.csv'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')


# =============================================================
# CELL 2 — Load Data
# =============================================================

train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

print(f'Train: {len(train_df)} | Test: {len(test_df)}')
print(f'Class distribution:\n{train_df["label"].value_counts()}')
print(f'Sun azimuth range: {train_df["sun_azimuth_angle"].min():.1f} to {train_df["sun_azimuth_angle"].max():.1f}')


# =============================================================
# CELL 3 — Dataset and TTA
# =============================================================

class LunarDataset(Dataset):
    def __init__(self, df, img_dir, augment=False, is_test=False):
        self.df      = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.augment = augment
        self.is_test = is_test

        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
        ])

        self.aug_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row         = self.df.iloc[idx]
        img_id      = row['image_id']
        sun_azimuth = float(row['sun_azimuth_angle'])

        img_path = os.path.join(self.img_dir, img_id)
        img      = Image.open(img_path).convert('RGB')

        if self.augment:
            img_tensor = self.aug_transform(img)
        else:
            img_tensor = self.transform(img)

        sun_rad     = np.deg2rad(sun_azimuth)
        sun_feature = torch.tensor(
            [np.sin(sun_rad), np.cos(sun_rad)],
            dtype=torch.float32
        )

        if self.is_test:
            return img_tensor, sun_feature, img_id

        label = torch.tensor(int(row['label']), dtype=torch.long)
        return img_tensor, sun_feature, label


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
# CELL 4 — Model Definitions (ConvNeXt + Swin, ImageNet-22k)
# =============================================================

def build_convnext_in22k():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            backbone = timm.create_model(
                'convnext_base.fb_in22k_ft_in1k_384',
                pretrained=True,
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
                pretrained=True,
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


MODEL_CONFIGS = [
    ('convnext_in22k', build_convnext_in22k),
    ('swin_in22k',     build_swin_in22k),
]

for name, builder in MODEL_CONFIGS:
    m = builder().to(DEVICE)
    params = sum(p.numel() for p in m.parameters())
    print(f'{name}: {params:,} parameters')
    del m
torch.cuda.empty_cache()


# =============================================================
# CELL 5 — Data Loaders
# =============================================================

train_meta, val_meta = train_test_split(
    train_df, test_size=0.15, random_state=42,
    stratify=train_df['label']
)
train_meta = train_meta.reset_index(drop=True)
val_meta   = val_meta.reset_index(drop=True)

class_counts  = train_meta['label'].value_counts().sort_index().values
class_weights = 1.0 / class_counts
sample_weights = train_meta['label'].map(
    {i: class_weights[i] for i in range(len(class_counts))}
).values
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_dataset = LunarDataset(train_meta, TRAIN_IMG_DIR, augment=True)
val_dataset   = LunarDataset(val_meta,   TRAIN_IMG_DIR, augment=False)
test_dataset  = LunarDataset(test_df,    EVAL_IMG_DIR,  is_test=True)

train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False,   num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False,   num_workers=0)

print(f'Train: {len(train_meta)} | Val: {len(val_meta)}')

weight_tensor = torch.tensor(
    class_weights / class_weights.sum() * 2,
    dtype=torch.float32
).to(DEVICE)


# =============================================================
# CELL 6 — Training
# =============================================================

def train_model(model, model_name, epochs=25):
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 5e-6},
        {'params': model.sun_encoder.parameters(), 'lr': 1e-4},
        {'params': model.head.parameters(), 'lr': 1e-4},
    ], weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-7
    )

    best_score = 0
    best_path  = f'/kaggle/working/best_{model_name}.pth'
    patience   = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, sun_angles, labels in train_loader:
            imgs       = imgs.to(DEVICE)
            sun_angles = sun_angles.to(DEVICE)
            labels     = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs, sun_angles)
            loss    = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        all_preds  = []
        all_labels = []
        with torch.no_grad():
            for imgs, sun_angles, labels in val_loader:
                imgs       = imgs.to(DEVICE)
                sun_angles = sun_angles.to(DEVICE)
                outputs    = model(imgs, sun_angles)
                preds      = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        score    = balanced_accuracy_score(all_labels, all_preds)
        avg_loss = train_loss / len(train_loader)

        print(f'[{model_name}] Epoch {epoch+1:02d}/{epochs} | '
              f'Loss: {avg_loss:.4f} | Balanced Acc: {score:.4f}')

        if score > best_score:
            best_score = score
            patience   = 0
            torch.save(model.state_dict(), best_path)
            print(f'  Best: {score:.4f}')
        else:
            patience += 1
            if patience >= 7:
                print(f'  Early stopping at epoch {epoch+1}')
                break

        scheduler.step()

    print(f'\n[{model_name}] Best: {best_score:.4f}\n')
    return best_score, best_path


trained_models = []
for model_name, builder in MODEL_CONFIGS:
    print(f'\n{"="*50}')
    print(f'Training {model_name}')
    print(f'{"="*50}')
    torch.cuda.empty_cache()
    model = builder()
    best_score, best_path = train_model(model, model_name)
    trained_models.append((model_name, builder, best_path, best_score))
    del model
    torch.cuda.empty_cache()

print('\nAll models trained:')
for name, _, path, score in trained_models:
    print(f'  {name}: {score:.4f}')


# =============================================================
# CELL 7 — TTA Ensemble Submission
# =============================================================

loaded_models = []
for model_name, builder, best_path, score in trained_models:
    m = builder().to(DEVICE)
    m.load_state_dict(torch.load(best_path))
    m.eval()
    loaded_models.append((model_name, m, score))
    print(f'Loaded {model_name} (val: {score:.4f})')

print('\nGenerating TTA ensemble predictions...')

all_image_ids = []
all_probs     = []

for i, row in test_df.iterrows():
    img_id      = row['image_id']
    sun_azimuth = float(row['sun_azimuth_angle'])

    img_path = os.path.join(EVAL_IMG_DIR, img_id)
    img      = Image.open(img_path).convert('RGB')

    model_probs = []
    for model_name, model, score in loaded_models:
        prob = tta_predict(model, img, sun_azimuth, DEVICE)
        model_probs.append(prob)

    ensemble_prob = np.mean(model_probs, axis=0)
    all_probs.append(ensemble_prob)
    all_image_ids.append(img_id)

    if (i + 1) % 200 == 0:
        print(f'  {i+1}/{len(test_df)}')

final_preds = [np.argmax(p) for p in all_probs]

submission = pd.DataFrame({
    'image_id': all_image_ids,
    'label':    final_preds
})
submission.to_csv('/kaggle/working/submission_timm.csv', index=False)
print(f'\nSaved {len(submission)} predictions')
print(submission['label'].value_counts())
