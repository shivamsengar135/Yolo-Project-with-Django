import argparse
from pathlib import Path

from ultralytics import YOLO

from device_utils import resolve_ultralytics_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX.")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Path to trained YOLO .pt model.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size.")
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 where supported.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device to use for export: "auto", "cpu", "cuda", or explicit id like "0".',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_ultralytics_device(args.device)
    model_path = Path(args.model)
    if not model_path.exists() and args.model != "yolov8n.pt":
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(args.model)
    exported_path = model.export(format="onnx", imgsz=args.imgsz, half=args.half, opset=args.opset, device=device)
    print(f"Export device used: {device}")
    print(f"Exported ONNX model: {exported_path}")


if __name__ == "__main__":
    main()
