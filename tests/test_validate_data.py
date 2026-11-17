import shutil

from PIL import Image

from src.food11.validate_data import CLASSES, SPLITS, validate


def make_dataset(root, per_class=3):
    """A tiny valid dataset: every image is a unique 128x128 RGB colour."""
    i = 0
    for split in SPLITS:
        for cls in CLASSES:
            folder = root / split / cls
            folder.mkdir(parents=True)
            for _ in range(per_class):
                Image.new("RGB", (128, 128), (i % 256, i // 256, 7)).save(folder / f"{i}.png")
                i += 1
    return root


def failed(results):
    return [name for name, errors in results.items() if errors]


def test_valid_dataset_passes(tmp_path):
    root = make_dataset(tmp_path)
    assert failed(validate(root, min_per_class=3)) == []


def test_duplicate_across_splits_is_caught(tmp_path):
    root = make_dataset(tmp_path)
    src = next((root / "training" / "Bread").iterdir())
    shutil.copy(src, root / "evaluation" / "Bread" / "copy.png")
    assert failed(validate(root, min_per_class=3)) == ["no image in two splits"]


def test_wrong_size_and_mode_are_caught(tmp_path):
    root = make_dataset(tmp_path)
    Image.new("L", (64, 64)).save(root / "training" / "Egg" / "bad.png")
    assert failed(validate(root, min_per_class=3)) == ["images open, RGB, 128x128"]


def test_missing_class_folder_is_caught(tmp_path):
    root = make_dataset(tmp_path)
    shutil.rmtree(root / "validation" / "Soup")
    assert "11 class folders per split" in failed(validate(root, min_per_class=3))
