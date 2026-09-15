"""
src/food11/data.py

Prepares the Food-11 dataset for training.

Reads the raw Food-11 images from:
    ./data/food11_raw/{training,evaluation,validation}/

where every file is named "<class_id>_<image_id>.<ext>" (class_id is 0-10),
and produces two ImageFolder-style, resized copies:

    ./data/food11_processed/<split>/<category>/...
        - every image, resized to 128x128, sorted into per-category folders

    ./data/food11_processed_mini/<split>/<category>/...
        - the same, but capped at MAX_PER_CATEGORY images per category/split
        - meant for fast local development / sanity-checking your pipeline

Run with:
    uv run python ./src/food11/data.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

# Order matters: this must match the class_id encoded in each raw filename.
# e.g. a file named "3_128.jpg" has class_id 3 -> "Egg".
CATEGORIES = [
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
]

SPLITS = ["training", "evaluation", "validation"]

RAW_DIR = Path("data/food11_raw")
PROCESSED_DIR = Path("data/food11_processed")
MINI_DIR = Path("data/food11_processed_mini")

IMAGE_SIZE = (128, 128)
MAX_PER_CATEGORY = 100  # cap per category per split, for the mini dataset

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def category_from_filename(stem: str) -> str:
    """Food-11 raw files are named '<class_id>_<image_id>' (extension stripped)."""
    class_id_str = stem.split("_")[0]
    class_id = int(class_id_str)
    return CATEGORIES[class_id]


def process_split(split: str, max_per_category: int | None = None) -> dict[str, int]:
    """
    Resize and sort one split (training/evaluation/validation) into
    per-category folders under PROCESSED_DIR, and, for the first
    `max_per_category` images of each category, also under MINI_DIR.

    Returns a dict of category -> number of images processed, for reporting.
    """
    raw_split_dir = RAW_DIR / split
    counts = {category: 0 for category in CATEGORIES}

    if not raw_split_dir.exists():
        print(f"  [skip] {raw_split_dir} does not exist")
        return counts

    for image_path in sorted(raw_split_dir.iterdir()):
        if image_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        try:
            category = category_from_filename(image_path.stem)
        except (ValueError, IndexError):
            print(f"  [warn] could not parse category from '{image_path.name}', skipping")
            continue

        out_dirs = [PROCESSED_DIR / split / category]
        if max_per_category is None or counts[category] < max_per_category:
            out_dirs.append(MINI_DIR / split / category)

        with Image.open(image_path) as img:
            img = img.convert("RGB").resize(IMAGE_SIZE)
            for out_dir in out_dirs:
                out_dir.mkdir(parents=True, exist_ok=True)
                img.save(out_dir / image_path.name)

        counts[category] += 1

    return counts


def main() -> None:
    print(f"Reading raw data from: {RAW_DIR.resolve()}")

    for split in SPLITS:
        print(f"\nProcessing split: {split}")
        counts = process_split(split, max_per_category=MAX_PER_CATEGORY)
        for category, n in counts.items():
            print(f"  {category:<16} {n:>5} images")

    print("\nDone.")
    print(f"Full dataset: {PROCESSED_DIR.resolve()}")
    print(f"Mini dataset: {MINI_DIR.resolve()}  (<= {MAX_PER_CATEGORY} per category)")


if __name__ == "__main__":
    main()
