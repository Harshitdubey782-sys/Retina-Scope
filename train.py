import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split
from tqdm import tqdm
import timm


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = r"E:\SIH26038-DR\aptos2019-blindness-detection"

TRAIN_CSV = os.path.join(DATASET_DIR, "train.csv")
TRAIN_IMAGES = os.path.join(DATASET_DIR, "train_images")

# Mini-run only — later change to None for full dataset
MAX_IMAGES = 200

IMAGE_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 2
LEARNING_RATE = 1e-4

NUM_CLASSES = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print("=" * 60)
print("SIH26038 - DIABETIC RETINOPATHY MINI TRAINING")
print("=" * 60)

print(f"\nDevice: {DEVICE}")
print(f"Maximum images: {MAX_IMAGES}")
print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")


# ============================================================
# DATASET
# ============================================================

class APTOSDataset(Dataset):

    def __init__(self, dataframe, image_dir):

        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_id = row["id_code"]
        label = int(row["diagnosis"])

        image_path = os.path.join(
            self.image_dir,
            image_id + ".png"
        )

        # Read image using OpenCV
        image = cv2.imread(image_path)

        if image is None:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            )

        # BGR → RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # CLAHE
        # ----------------------------------------------------

        lab = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2LAB
        )

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        l = clahe.apply(l)

        lab = cv2.merge([l, a, b])

        image = cv2.cvtColor(
            lab,
            cv2.COLOR_LAB2RGB
        )

        # Convert to PIL for torchvision
        image = Image.fromarray(image)

        image = self.transform(image)

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label


# ============================================================
# LOAD CSV
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(TRAIN_CSV)

print(f"Total images in CSV: {len(df)}")


# ============================================================
# MINI DATASET
# ============================================================

if MAX_IMAGES is not None:

    # Stratified mini-sample
    df, _ = train_test_split(
        df,
        train_size=MAX_IMAGES,
        stratify=df["diagnosis"],
        random_state=42
    )

    df = df.reset_index(drop=True)

print(f"Images selected for this run: {len(df)}")


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

train_df, val_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["diagnosis"],
    random_state=42
)

print(f"Training images: {len(train_df)}")
print(f"Validation images: {len(val_df)}")


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = APTOSDataset(
    train_df,
    TRAIN_IMAGES
)

val_dataset = APTOSDataset(
    val_df,
    TRAIN_IMAGES
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = train_df["diagnosis"].value_counts().sort_index()

print("\nTraining class distribution:")

for cls in range(NUM_CLASSES):

    count = class_counts.get(cls, 0)

    print(f"Grade {cls}: {count}")


# Calculate balanced weights
total = len(train_df)

weights = []

for cls in range(NUM_CLASSES):

    count = class_counts.get(cls, 1)

    weight = total / (NUM_CLASSES * count)

    weights.append(weight)

class_weights = torch.tensor(
    weights,
    dtype=torch.float32
).to(DEVICE)

print("\nClass weights:")
print(class_weights)


# ============================================================
# MODEL
# ============================================================

print("\nLoading EfficientNet-B0...")

model = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)

print("✅ Model loaded")


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total_samples = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for images, labels in progress:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total_samples += images.size(0)

        progress.set_postfix(
            loss=loss.item()
        )

    train_loss = (
        running_loss / total_samples
    )

    train_accuracy = (
        correct / total_samples
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_samples = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += (
                loss.item() * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            val_correct += (
                predictions == labels
            ).sum().item()

            val_samples += images.size(0)

    val_loss /= val_samples

    val_accuracy = (
        val_correct / val_samples
    )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("\n")
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Train Loss      : {train_loss:.4f}")
    print(f"Train Accuracy  : {train_accuracy:.4f}")
    print(f"Validation Loss : {val_loss:.4f}")
    print(f"Validation Acc  : {val_accuracy:.4f}")


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

model_path = os.path.join(
    "models",
    "efficientnet_b0_mini.pth"
)

torch.save(
    model.state_dict(),
    model_path
)

print("\n" + "=" * 60)
print("🎉 MINI TRAINING COMPLETE")
print("=" * 60)

print(f"\nModel saved to:")
print(model_path)