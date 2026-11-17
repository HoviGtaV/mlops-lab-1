"""
src/food11/validate_data.py

Checks a processed Food-11 dataset against its data contract and exits
with status 1 if any check fails.

    uv run python src/food11/validate_data.py data/food11_processed_mini
    uv run python src/food11/validate_data.py data/food11_processed --min-per-class 100
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

CLASSES = [
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
SPLITS = ["training", "validation", "evaluation"]
IMAGE_SIZE = (128, 128)
MAX_EXAMPLES = 5  # how many offending files to print per check


def list_images(root: Path, split: str, cls: str) -> list[Path]:
    folder = root / split / cls
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file())


def check_class_folders(root: Path) -> list[str]:
    errors = []
    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            errors.append(f"{split}: folder missing")
            continue
        found = {p.name for p in split_dir.iterdir() if p.is_dir()}
        missing = sorted(set(CLASSES) - found)
        extra = sorted(found - set(CLASSES))
        if missing:
            errors.append(f"{split}: missing classes {missing}")
        if extra:
            errors.append(f"{split}: unexpected folders {extra}")
    return errors


def check_images(root: Path) -> list[str]:
    errors = []
    for split in SPLITS:
        for cls in CLASSES:
            for path in list_images(root, split, cls):
                try:
                    with Image.open(path) as img:
                        img.load()
                        if img.mode != "RGB":
                            errors.append(f"{path}: mode {img.mode}, expected RGB")
                        if img.size != IMAGE_SIZE:
                            errors.append(f"{path}: size {img.size}, expected {IMAGE_SIZE}")
                except (OSError, ValueError, SyntaxError) as exc:  # unreadable or corrupted
                    errors.append(f"{path}: cannot open ({exc.__class__.__name__})")
    return errors


def check_min_per_class(root: Path, min_per_class: int) -> list[str]:
    errors = []
    for split in SPLITS:
        for cls in CLASSES:
            n = len(list_images(root, split, cls))
            if n < min_per_class:
                errors.append(f"{split}/{cls}: {n} images < {min_per_class}")
    return errors


def check_no_leakage(root: Path) -> list[str]:
    """No identical file content in two different splits."""
    seen: dict[str, set[str]] = defaultdict(set)
    where: dict[str, list[Path]] = defaultdict(list)
    for split in SPLITS:
        for cls in CLASSES:
            for path in list_images(root, split, cls):
                digest = hashlib.md5(path.read_bytes()).hexdigest()
                seen[digest].add(split)
                where[digest].append(path)
    return [
        "same content in " + ", ".join(str(p) for p in where[d])
        for d, splits in seen.items()
        if len(splits) > 1
    ]


def class_shares(root: Path, split: str) -> dict[str, float]:
    counts = {cls: len(list_images(root, split, cls)) for cls in CLASSES}
    total = sum(counts.values()) or 1
    return {cls: n / total for cls, n in counts.items()}


def check_class_balance(root: Path, tau: float) -> list[str]:
    train = class_shares(root, "training")
    val = class_shares(root, "validation")
    return [
        f"{cls}: training share {train[cls]:.3f} vs validation {val[cls]:.3f} "
        f"(diff {abs(train[cls] - val[cls]):.3f} > {tau})"
        for cls in CLASSES
        if abs(train[cls] - val[cls]) > tau
    ]


def validate(root: Path, min_per_class: int = 50, tau: float = 0.05) -> dict[str, list[str]]:
    """Runs every check; returns {check name: list of errors}."""
    return {
        "11 class folders per split": check_class_folders(root),
        "images open, RGB, 128x128": check_images(root),
        f"at least {min_per_class} images per class": check_min_per_class(root, min_per_class),
        "no image in two splits": check_no_leakage(root),
        f"train/val class shares within {tau}": check_class_balance(root, tau),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("dataset", type=Path, help="e.g. data/food11_processed_mini")
    parser.add_argument("--min-per-class", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.05)
    args = parser.parse_args()

    results = validate(args.dataset, args.min_per_class, args.tau)
    for name, errors in results.items():
        print(f"{'PASS' if not errors else 'FAIL'}  {name}")
        for err in errors[:MAX_EXAMPLES]:
            print(f"      - {err}")
        if len(errors) > MAX_EXAMPLES:
            print(f"      ... and {len(errors) - MAX_EXAMPLES} more")

    failed = [name for name, errors in results.items() if errors]
    if failed:
        print(f"\n{len(failed)} check(s) failed")
        sys.exit(1)
    print("\nAll checks passed")


if __name__ == "__main__":
    main()
