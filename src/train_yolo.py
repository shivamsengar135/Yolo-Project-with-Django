import argparse
from pathlib import Path

from ultralytics import YOLO

from device_utils import resolve_ultralytics_device

def augmentation_preset(name: str) -> dict:
    presets = {
        "default": {},
        "balanced": {
            "mosaic": 1.0,
            "mixup": 0.15,
            "copy_paste": 0.1,
            "degrees": 5.0,
            "translate": 0.12,
            "scale": 0.55,
            "fliplr": 0.5,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
        },
        "minority_boost": {
            "mosaic": 1.0,
            "mixup": 0.2,
            "copy_paste": 0.2,
            "degrees": 7.0,
            "translate": 0.15,
            "scale": 0.6,
            "fliplr": 0.5,
            "hsv_h": 0.02,
            "hsv_s": 0.75,
            "hsv_v": 0.45,
        },
    }
    if name not in presets:
        raise ValueError(f"Unknown preset '{name}'. Choose one of: {', '.join(sorted(presets))}")
    return presets[name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO with transfer learning.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/dataset.yaml",
        help="Path to dataset YAML file.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="Pretrained YOLO weights to start from.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device to use: "auto", "cpu", "cuda", or explicit Ultralytics id like "0".',
    )
    parser.add_argument(
        "--project",
        type=str,
        default="results/yolo_train",
        help="Directory to store training runs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="exp",
        help="Run name under project directory.",
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=10,
        help="How many layers to freeze for transfer learning warm start.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience in epochs.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="default",
        choices=["default", "balanced", "minority_boost"],
        help="Augmentation preset. Use minority_boost when one class underperforms.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Dataloader workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow reuse of an existing run directory name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_ultralytics_device(args.device)
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_path}. Update data/dataset.yaml first."
        )

    model = YOLO(args.weights)
    train_args = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
        "project": args.project,
        "name": args.name,
        "freeze": args.freeze,
        "patience": args.patience,
        "workers": args.workers,
        "seed": args.seed,
        "exist_ok": args.exist_ok,
    }
    train_args.update(augmentation_preset(args.preset))
    results = model.train(**train_args)
    print(f"Training device used: {device}")

    save_dir = getattr(results, "save_dir", None)
    if save_dir is not None:
        print(f"Training completed. Check weights under: {Path(save_dir) / 'weights'}")
    else:
        print("Training completed.")


if __name__ == "__main__":
    main()
