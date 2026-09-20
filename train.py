# =============================================================
# Pareidolia Paradox — Training Script
# Team: Prithviraj Khedekar & Arnav Pranil Gadge
# VIT Pune
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
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

# =============================================================
# PATHS — update these to match your dataset location
# =============================================================
TRAIN_IMG_DIR = '/kaggle/input/datasets/arnavgadgevit/train-images/train_images'
TRAIN_CSV     = '/kaggle/input/datasets/arnavgadgevit/train-meta/train_metadata.csv'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

# =============================================================
# LOAD DATA
# =============================================================
train_df = pd.read_csv(TRAIN_CSV)
print(f'Train samples: {len(train_df)}')
print(f'Class distribution:\n{train_df["label"].value_counts()}')

# =============================================================
# DATASET
# =============================================================
class LunarDataset(Dataset):
    def __init__(self, df, img_dir, augment=False):
        self.df      = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.augment = augment

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

        label = torch.tensor(int(row['label']), dtype=torch.long)
        return img_tensor, sun_feature, label


# =============================================================
# MODEL DEFINITIONS
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

# =============================================================
# DATA LOADERS
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

train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False,   num_workers=0)

print(f'Train: {len(train_meta)} | Val: {len(val_meta)}')

weight_tensor = torch.tensor(
    class_weights / class_weights.sum() * 2,
    dtype=torch.float32
).to(DEVICE)

# =============================================================
# TRAINING
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
