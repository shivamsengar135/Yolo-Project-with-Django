import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show readiness status for this object detection project.")
    parser.add_argument("--root", type=str, default=".", help="Project root directory.")
    return parser.parse_args()


def check(label: str, path: Path, is_dir: bool = False) -> tuple[str, bool]:
    exists = path.is_dir() if is_dir else path.exists()
    status = "OK" if exists else "MISSING"
    return f"[{status}] {label}: {path}", exists


def main() -> None:
    args = parse_args()
    root = Path(args.root)

    checks = [
        check("Dataset YAML", root / "data" / "dataset.yaml"),
        check("Train images", root / "data" / "images" / "train", is_dir=True),
        check("Train labels", root / "data" / "labels" / "train", is_dir=True),
        check("Val images", root / "data" / "images" / "val", is_dir=True),
        check("Val labels", root / "data" / "labels" / "val", is_dir=True),
        check("Training script", root / "src" / "train_yolo.py"),
        check("YOLO webcam script", root / "src" / "infer_webcam_yolo.py"),
        check("SSD webcam script", root / "src" / "infer_webcam_ssd.py"),
        check("ONNX webcam script", root / "src" / "infer_webcam_onnx.py"),
        check("Downloader script", root / "src" / "download_dataset.py"),
        check("Pretrained YOLO weights", root / "yolov8n.pt"),
    ]

    print("Project status")
    for line, _ in checks:
        print(line)

    train_images = len(list((root / "data" / "images" / "train").glob("*"))) if (root / "data" / "images" / "train").exists() else 0
    val_images = len(list((root / "data" / "images" / "val").glob("*"))) if (root / "data" / "images" / "val").exists() else 0
    train_labels = len(list((root / "data" / "labels" / "train").glob("*.txt"))) if (root / "data" / "labels" / "train").exists() else 0
    val_labels = len(list((root / "data" / "labels" / "val").glob("*.txt"))) if (root / "data" / "labels" / "val").exists() else 0

    print("")
    print(f"Data counts: train_images={train_images}, train_labels={train_labels}, val_images={val_images}, val_labels={val_labels}")
    ready = all(ok for _, ok in checks[:10]) and train_images > 0 and train_labels > 0 and val_images > 0 and val_labels > 0
    print(f"Training readiness: {'READY' if ready else 'NOT READY'}")


if __name__ == "__main__":
    main()
