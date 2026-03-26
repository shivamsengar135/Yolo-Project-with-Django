import argparse
from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a YOLO-format dataset.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/dataset.yaml",
        help="Path to dataset YAML file.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_split_dir(dataset_root: Path, split_value: str) -> Path:
    split_path = Path(split_value)
    if split_path.is_absolute():
        return split_path
    return dataset_root / split_path


def get_images(split_dir: Path) -> list[Path]:
    return sorted([path for path in split_dir.glob("*") if path.suffix.lower() in IMAGE_EXTENSIONS])


def validate_label_file(label_path: Path, class_count: int) -> list[str]:
    errors = []
    if not label_path.exists():
        errors.append(f"Missing label file: {label_path}")
        return errors

    with label_path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle.readlines() if line.strip()]

    if not lines:
        errors.append(f"Empty label file: {label_path}")
        return errors

    for index, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{label_path} line {index}: expected 5 values, found {len(parts)}")
            continue

        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"{label_path} line {index}: non-numeric values found")
            continue

        if class_id < 0 or class_id >= class_count:
            errors.append(f"{label_path} line {index}: invalid class id {class_id}")
        if any(value < 0 or value > 1 for value in values):
            errors.append(f"{label_path} line {index}: coordinates must be normalized between 0 and 1")

    return errors


def main() -> None:
    args = parse_args()
    yaml_path = Path(args.data)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")

    config = load_yaml(yaml_path)
    dataset_root = Path(config.get("path", "data"))
    names = config.get("names", {})
    class_count = len(names)
    if class_count == 0:
        raise ValueError("No class names found in dataset YAML.")

    all_errors = []
    for split_name in ("train", "val", "test"):
        split_value = config.get(split_name)
        if not split_value:
            continue

        image_dir = resolve_split_dir(dataset_root, split_value)
        label_dir = Path(str(image_dir).replace("images", "labels"))
        if not image_dir.exists():
            all_errors.append(f"Missing image directory: {image_dir}")
            continue
        if not label_dir.exists():
            all_errors.append(f"Missing label directory: {label_dir}")
            continue

        images = get_images(image_dir)
        if not images:
            if split_name != "test":
                all_errors.append(f"No images found in {image_dir}")
            continue

        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"
            all_errors.extend(validate_label_file(label_path, class_count))

    if all_errors:
        print("Dataset validation failed:")
        for error in all_errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("Dataset validation passed.")


if __name__ == "__main__":
    main()
