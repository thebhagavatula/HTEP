import random
import shutil
import json
from pathlib import Path

class DatasetSplitter:
    def __init__(
        self,
        input_dir="data/datasets/dataset_block",
        output_dir="data/icr_training",
        test_ratio=0.2
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.train_dir = self.output_dir / "train"
        self.test_dir = self.output_dir / "test"
        self.test_ratio = test_ratio

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Dataset not found: {self.input_dir}")

        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def split(self):
        print(f"\nSplitting dataset from: {self.input_dir}")
        print(f"Saving to: {self.output_dir}")
        print(f"Train/Test ratio: {int((1-self.test_ratio)*100)}% / {int(self.test_ratio*100)}%\n")

        metadata = {
            "train_samples": {},
            "test_samples": {},
            "image_width": 28,
            "image_height": 28
        }

        for class_dir in sorted(self.input_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            label = class_dir.name
            images = list(list(class_dir.glob("*.png")) +
                        list(class_dir.glob("*.jpg")) +
                        list(class_dir.glob("*.jpeg")))

            if not images:
                continue

            random.shuffle(images)
            split_idx = int(len(images) * (1 - self.test_ratio))

            train_imgs = images[:split_idx]
            test_imgs = images[split_idx:]

            train_label_dir = self.train_dir / label
            test_label_dir = self.test_dir / label
            train_label_dir.mkdir(exist_ok=True)
            test_label_dir.mkdir(exist_ok=True)

            for img in train_imgs:
                shutil.copy2(img, train_label_dir / img.name)

            for img in test_imgs:
                shutil.copy2(img, test_label_dir / img.name)

            metadata["train_samples"][label] = len(train_imgs)
            metadata["test_samples"][label] = len(test_imgs)

            print(f"✓ {label}: {len(train_imgs)} train, {len(test_imgs)} test")

        metadata["characters"] = sorted(metadata["train_samples"].keys())
        metadata["num_characters"] = len(metadata["characters"])

        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print("\n✓ Dataset split complete")
        print(f"✓ Metadata saved to: {metadata_file}")

        print("\nDataset Summary:")
        print(f"  Total characters: {metadata['num_characters']}")
        print(f"  Total train samples: {sum(metadata['train_samples'].values())}")
        print(f"  Total test samples: {sum(metadata['test_samples'].values())}")
        print(f"  Image size: 28x28")


if __name__ == "__main__":
    splitter = DatasetSplitter(
        input_dir="data/datasets/dataset_block",
        output_dir="data/icr_training",
        test_ratio=0.2
    )
    splitter.split()
