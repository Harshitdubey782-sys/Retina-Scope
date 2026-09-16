import os
import pandas as pd
from PIL import Image

# ============================================
# SIH26038 - APTOS 2019 DATASET CHECK
# ============================================

DATASET_DIR = r"E:\SIH26038-DR\aptos2019-blindness-detection"

TRAIN_CSV = os.path.join(DATASET_DIR, "train.csv")
TRAIN_IMAGES = os.path.join(DATASET_DIR, "train_images")


# ============================================
# 1. CHECK DATASET PATH
# ============================================

print("=" * 60)
print("SIH26038 - APTOS 2019 DATASET VERIFICATION")
print("=" * 60)

if not os.path.exists(DATASET_DIR):
    print("\n❌ Dataset folder not found!")
    print(DATASET_DIR)
    exit()

print("\n✅ Dataset folder found")


# ============================================
# 2. CHECK REQUIRED FILES/FOLDERS
# ============================================

print("\nChecking required files and folders...\n")

required_items = [
    "train.csv",
    "test.csv",
    "train_images",
    "test_images"
]

for item in required_items:

    path = os.path.join(DATASET_DIR, item)

    if os.path.exists(path):
        print(f"✅ {item}")
    else:
        print(f"❌ {item} NOT FOUND")


# ============================================
# 3. LOAD TRAIN.CSV
# ============================================

print("\n" + "=" * 60)
print("TRAIN CSV INFORMATION")
print("=" * 60)

if not os.path.exists(TRAIN_CSV):
    print("❌ train.csv not found!")
    exit()

df = pd.read_csv(TRAIN_CSV)

print("\nColumns:")
print(list(df.columns))

print("\nNumber of training records:")
print(len(df))

print("\nFirst 5 rows:")
print(df.head())


# ============================================
# 4. CHECK CLASS DISTRIBUTION
# ============================================

print("\n" + "=" * 60)
print("DIABETIC RETINOPATHY CLASS DISTRIBUTION")
print("=" * 60)

if "diagnosis" in df.columns:

    class_counts = df["diagnosis"].value_counts().sort_index()

    print("\nDiagnosis counts:")

    for diagnosis, count in class_counts.items():
        print(f"Grade {diagnosis}: {count} images")

else:
    print("❌ 'diagnosis' column not found!")


# ============================================
# 5. CHECK IMAGE FILES
# ============================================

print("\n" + "=" * 60)
print("IMAGE FILE VERIFICATION")
print("=" * 60)

if not os.path.exists(TRAIN_IMAGES):
    print("❌ train_images folder not found!")
    exit()

missing_images = []
valid_images = 0

print("\nChecking first 20 images...\n")

for _, row in df.head(20).iterrows():

    image_id = row["id_code"]
    diagnosis = row["diagnosis"]

    image_path = os.path.join(
        TRAIN_IMAGES,
        image_id + ".png"
    )

    if os.path.exists(image_path):

        try:

            image = Image.open(image_path)

            print(
                f"✅ {image_id}.png | "
                f"Size: {image.size} | "
                f"Grade: {diagnosis}"
            )

            valid_images += 1

        except Exception as e:

            print(
                f"❌ {image_id}.png | "
                f"Could not open image | {e}"
            )

    else:

        print(f"❌ {image_id}.png | FILE NOT FOUND")
        missing_images.append(image_id)


# ============================================
# 6. FINAL RESULT
# ============================================

print("\n" + "=" * 60)
print("FINAL RESULT")
print("=" * 60)

print(f"\nTraining records in CSV : {len(df)}")
print(f"Images successfully checked : {valid_images}")
print(f"Missing images : {len(missing_images)}")

if len(missing_images) == 0:

    print("\n🎉 DATASET STRUCTURE LOOKS GOOD!")
    print("We can move to the ML training pipeline.")

else:

    print("\n⚠️ Some images are missing.")
    print("Missing IDs:")
    print(missing_images)

print("\n" + "=" * 60)