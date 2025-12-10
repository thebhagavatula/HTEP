
import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import json
from PIL import Image, ImageDraw, ImageFont
import random


class DatasetGenerator:
    """
    Generate training datasets for ICR (Block Letters).
    Creates synthetic data and processes real scanned images.
    """

    def __init__(self, output_dir: str = "data/icr_training"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.train_dir = self.output_dir / "train"
        self.test_dir = self.output_dir / "test"
        self.train_dir.mkdir(exist_ok=True)
        self.test_dir.mkdir(exist_ok=True)

        # Characters to recognize (alphanumeric + common medical symbols)
        self.characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

        # Image dimensions for training
        self.img_width = 28
        self.img_height = 28

    def generate_synthetic_data(self, samples_per_char: int = 100):
        """
        Generate synthetic block letter images for training.

        Args:
            samples_per_char: Number of samples to generate per character
        """
        print("Generating synthetic training data...")
        print(f"Creating {samples_per_char} samples per character")
        print(f"Total samples: {len(self.characters) * samples_per_char}")

        for char in self.characters:
            # Create directory for this character
            char_dir = self.train_dir / char
            char_dir.mkdir(exist_ok=True)

            for i in range(samples_per_char):
                # Generate image with variations
                img = self._create_character_image(
                    char,
                    variation=i % 10  # 10 different variations
                )

                # Save image
                filename = char_dir / f"{char}_{i:04d}.png"
                cv2.imwrite(str(filename), img)

            print(f"Generated {samples_per_char} samples for '{char}'")

        print(f"\nSynthetic data generation complete!")
        print(f"Data saved to: {self.train_dir}")

    def _create_character_image(self, char: str, variation: int = 0) -> np.ndarray:
        """
        Create a single character image with variations.

        Args:
            char: Character to generate
            variation: Variation style (0-9)

        Returns:
            Numpy array of image
        """
        # Create white background
        img = Image.new('L', (self.img_width, self.img_height), color=255)
        draw = ImageDraw.Draw(img)

        # Font size variations
        font_sizes = [16, 18, 20, 22, 24]
        font_size = font_sizes[variation % len(font_sizes)]

        # Try to load a font (fallback to default if not available)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

        # Calculate text position (centered with slight variations)
        bbox = draw.textbbox((0, 0), char, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Add position variations
        x_offset = random.randint(-2, 2)
        y_offset = random.randint(-2, 2)

        x = (self.img_width - text_width) // 2 + x_offset
        y = (self.img_height - text_height) // 2 + y_offset

        # Draw text in black
        draw.text((x, y), char, fill=0, font=font)

        # Convert to numpy array
        img_array = np.array(img)

        # Add noise variations
        if variation >= 5:
            img_array = self._add_noise(img_array, variation)

        # Add rotation variations
        if variation % 3 == 0:
            img_array = self._add_rotation(img_array, angle=random.randint(-5, 5))

        return img_array

    def _add_noise(self, img: np.ndarray, noise_level: int) -> np.ndarray:
        """Add random noise to image."""
        noise = np.random.normal(0, noise_level * 2, img.shape)
        noisy_img = img + noise
        return np.clip(noisy_img, 0, 255).astype(np.uint8)

    def _add_rotation(self, img: np.ndarray, angle: float) -> np.ndarray:
        """Add slight rotation to image."""
        h, w = img.shape
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, matrix, (w, h),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=255)
        return rotated

    def process_scanned_images(self, input_dir: str):
        """
        Process real scanned images of block letters.

        Args:
            input_dir: Directory containing scanned character images
        """
        input_path = Path(input_dir)

        if not input_path.exists():
            print(f"⚠ Input directory not found: {input_dir}")
            return

        print(f"Processing scanned images from: {input_dir}")

        # Expected structure: input_dir/A/*.png, input_dir/B/*.png, etc.
        for char_dir in input_path.iterdir():
            if not char_dir.is_dir():
                continue

            char = char_dir.name.upper()
            if char not in self.characters:
                print(f"⚠ Skipping unknown character: {char}")
                continue

            # Create output directory
            output_char_dir = self.train_dir / char
            output_char_dir.mkdir(exist_ok=True)

            # Process all images in this character directory
            image_files = list(char_dir.glob("*.png")) + list(char_dir.glob("*.jpg"))

            for img_file in image_files:
                processed_img = self._preprocess_scanned_image(str(img_file))

                # Save processed image
                output_file = output_char_dir / f"scanned_{img_file.name}"
                cv2.imwrite(str(output_file), processed_img)

            print(f"✓ Processed {len(image_files)} scanned images for '{char}'")

    def _preprocess_scanned_image(self, img_path: str) -> np.ndarray:
        """
        Preprocess a scanned image for training.

        Args:
            img_path: Path to image file

        Returns:
            Preprocessed image array
        """
        # Read image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Resize to standard size
        img = cv2.resize(img, (self.img_width, self.img_height))

        # Apply thresholding
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Denoise
        img = cv2.fastNlMeansDenoising(img)

        return img

    def create_train_test_split(self, test_ratio: float = 0.2):
        """
        Split training data into train and test sets.

        Args:
            test_ratio: Ratio of data to use for testing
        """
        print(f"\nCreating train/test split ({100 * (1 - test_ratio):.0f}% train, {100 * test_ratio:.0f}% test)...")

        for char in self.characters:
            char_dir = self.train_dir / char

            if not char_dir.exists():
                continue

            # Get all images for this character
            images = list(char_dir.glob("*.png"))

            if not images:
                continue

            # Shuffle and split
            random.shuffle(images)
            split_idx = int(len(images) * (1 - test_ratio))

            test_images = images[split_idx:]

            # Create test directory for this character
            test_char_dir = self.test_dir / char
            test_char_dir.mkdir(exist_ok=True)

            # Move test images
            for img_file in test_images:
                new_path = test_char_dir / img_file.name
                img_file.rename(new_path)

            print(f"✓ '{char}': {split_idx} train, {len(test_images)} test")

        print("✓ Train/test split complete!")

    def generate_metadata(self):
        """Generate metadata file with dataset information."""
        metadata = {
            'characters': list(self.characters),
            'num_characters': len(self.characters),
            'image_width': self.img_width,
            'image_height': self.img_height,
            'train_samples': {},
            'test_samples': {}
        }

        # Count training samples
        for char in self.characters:
            char_dir = self.train_dir / char
            if char_dir.exists():
                metadata['train_samples'][char] = len(list(char_dir.glob("*.png")))

        # Count test samples
        for char in self.characters:
            char_dir = self.test_dir / char
            if char_dir.exists():
                metadata['test_samples'][char] = len(list(char_dir.glob("*.png")))

        # Save metadata
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✓ Metadata saved to: {metadata_file}")

        # Print summary
        total_train = sum(metadata['train_samples'].values())
        total_test = sum(metadata['test_samples'].values())

        print(f"\nDataset Summary:")
        print(f"  Total training samples: {total_train}")
        print(f"  Total test samples: {total_test}")
        print(f"  Characters: {metadata['num_characters']}")
        print(f"  Image size: {metadata['image_width']}x{metadata['image_height']}")


def create_dataset(samples_per_char: int = 100):
    """
    Convenience function to create a complete dataset.

    Args:
        samples_per_char: Number of synthetic samples per character
    """
    generator = DatasetGenerator()

    # Generate synthetic data
    generator.generate_synthetic_data(samples_per_char)

    # Split into train/test
    generator.create_train_test_split(test_ratio=0.2)

    # Generate metadata
    generator.generate_metadata()

    print(f"\n{'=' * 60}")
    print("Dataset creation complete!")
    print(f"Location: {generator.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    # Create a dataset with 100 samples per character
    create_dataset(samples_per_char=100)