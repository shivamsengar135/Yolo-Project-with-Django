import argparse
from collections import Counter
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print a simple YOLO dataset summary.")
    parser.add_argument("--data", type=str, default="data/dataset.yaml", help="Path to dataset YAML file.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_split_dir(dataset_root: Path, split_value: str) -> Path:
    split_path = Path(split_value)
    if split_path.is_absolute():
        return split_path
    return dataset_root / split_path


def summarize_split(image_dir: Path, label_dir: Path, names: dict[int, str]) -> tuple[int, int, Counter]:
    image_count = len([path for path in image_dir.glob("*") if path.is_file()])
    label_files = list(label_dir.glob("*.txt"))
    class_counts: Counter = Counter()
    for label_file in label_files:
        with label_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                class_id = int(stripped.split()[0])
                class_counts[names[class_id]] += 1
    return image_count, len(label_files), class_counts


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.data))
    dataset_root = Path(config.get("path", "data"))
    names = config.get("names", {})
    names = {int(k): v for k, v in names.items()}

    print("Dataset summary")
    print(f"Classes: {', '.join(names.values())}")
    for split_name in ("train", "val", "test"):
        split_value = config.get(split_name)
        if not split_value:
            continue
        image_dir = resolve_split_dir(dataset_root, split_value)
        label_dir = Path(str(image_dir).replace("images", "labels"))
        if not image_dir.exists() or not label_dir.exists():
            print(f"{split_name}: missing")
            continue

        image_count, label_count, class_counts = summarize_split(image_dir, label_dir, names)
        print(f"{split_name}: {image_count} images, {label_count} labels")
        if class_counts:
            for class_name in names.values():
                print(f"  {class_name}: {class_counts.get(class_name, 0)} boxes")


if __name__ == "__main__":
    main()
