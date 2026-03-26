import argparse
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz


DEFAULT_CLASSES = ["bottle", "cup", "cell phone"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a small labeled object detection dataset from COCO and export it in YOLO format."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="downloads/coco_subset",
        help="Where to store the downloaded raw dataset subset.",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="data",
        help="Where to export the YOLO-formatted dataset.",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=120,
        help="Maximum number of training images to download.",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=30,
        help="Maximum number of validation images to download.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of download workers to use. Lower is slower but more stable.",
    )
    return parser.parse_args()


def download_split(
    split: str,
    max_samples: int,
    classes: list[str],
    seed: int,
    num_workers: int,
) -> fo.Dataset:
    return foz.load_zoo_dataset(
        "coco-2017",
        split=split,
        label_types=["detections"],
        classes=classes,
        max_samples=max_samples,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
    )


def export_split(
    dataset: fo.Dataset,
    split: str,
    export_root: Path,
    classes: list[str],
) -> None:
    dataset.export(
        export_dir=str(export_root),
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        classes=classes,
        split=split,
        export_media=True,
    )


def ensure_test_split(export_root: Path) -> None:
    image_test_dir = export_root / "images" / "test"
    label_test_dir = export_root / "labels" / "test"
    image_test_dir.mkdir(parents=True, exist_ok=True)
    label_test_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    export_root = Path(args.export_dir)
    export_root.mkdir(parents=True, exist_ok=True)

    train_dataset = download_split(
        "train",
        args.train_samples,
        DEFAULT_CLASSES,
        args.seed,
        args.num_workers,
    )
    export_split(train_dataset, "train", export_root, DEFAULT_CLASSES)

    val_dataset = download_split(
        "validation",
        args.val_samples,
        DEFAULT_CLASSES,
        args.seed,
        args.num_workers,
    )
    export_split(val_dataset, "val", export_root, DEFAULT_CLASSES)

    ensure_test_split(export_root)
    print(f"Downloaded and exported dataset to {export_root.resolve()}")


if __name__ == "__main__":
    main()
