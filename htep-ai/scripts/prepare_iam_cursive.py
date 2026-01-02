import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# -------- PATHS (ADJUST ONLY BASE_DIR IF NEEDED) --------
BASE_DIR = Path("data/datasets/iam-handwriting-word-dataset")
WORDS_DIR = BASE_DIR / "words"
LABEL_FILE = BASE_DIR / "words.txt"

OUT_BASE = Path("data/icr_training/cursive")
TRAIN_DIR = OUT_BASE / "train"
TEST_DIR = OUT_BASE / "test"

TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)

samples = []

# -------- PARSE words.txt --------
with open(LABEL_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("#"):
            continue

        parts = line.strip().split()
        if len(parts) < 9:
            continue

        img_id = parts[0]
        status = parts[1]
        label = parts[-1].lower()

        # ---- FILTERING RULES ----
        if status != "ok":
            continue
        if not label.isalpha():
            continue
        if len(label) < 3 or len(label) > 15:
            continue

        # ---- IMAGE PATH ----
        # Example: a01-000u-00-00 → a01/a01-000u/a01-000u-00-00.png
        parts = img_id.split("-")

        writer = parts[0]              # a01
        form = f"{parts[0]}-{parts[1]}"  # a01-000u
        
        img_path = WORDS_DIR / writer / form / f"{img_id}.png"


        if img_path.exists():
            samples.append((img_path, label))

print(f"✅ Valid samples collected: {len(samples)}")

# -------- TRAIN / TEST SPLIT --------
train_samples, test_samples = train_test_split(
    samples, test_size=0.2, random_state=42
)

def copy_samples(sample_list, target_root):
    for img_path, label in sample_list:
        label_dir = target_root / label
        label_dir.mkdir(exist_ok=True)
        shutil.copy(img_path, label_dir / img_path.name)

copy_samples(train_samples, TRAIN_DIR)
copy_samples(test_samples, TEST_DIR)

print("🎉 IAM dataset successfully converted for cursive ICR")
