import argparse
import csv
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from device_utils import resolve_ultralytics_device
from webcam_utils import open_webcam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple webcam benchmark for YOLO models.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Model path.")
    parser.add_argument("--source", type=str, default="auto", help="Webcam index or 'auto'.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to test.")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device to use: "auto", "cpu", "cuda", or explicit Ultralytics id like "0".',
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/metrics.csv",
        help="CSV path for benchmark results.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        help='Backend label for CSV (auto, pytorch, onnx, etc.).',
    )
    return parser.parse_args()


def infer_backend_name(model_path: str, backend_arg: str) -> str:
    if backend_arg != "auto":
        return backend_arg
    return "onnx" if model_path.lower().endswith(".onnx") else "pytorch"


def main() -> None:
    args = parse_args()
    device = resolve_ultralytics_device(args.device)
    model = YOLO(args.model)
    cap, used_source = open_webcam(args.source)

    frame_times = []
    processed = 0
    while processed < args.frames:
        ok, frame = cap.read()
        if not ok:
            break

        start = time.perf_counter()
        model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=device, verbose=False)
        elapsed = time.perf_counter() - start
        frame_times.append(elapsed)
        processed += 1

    cap.release()

    if not frame_times:
        raise RuntimeError("No frames were processed during benchmark.")

    avg_latency_ms = (sum(frame_times) / len(frame_times)) * 1000.0
    fps = 1.0 / (sum(frame_times) / len(frame_times))
    backend = infer_backend_name(args.model, args.backend)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(["model", "backend", "frames", "avg_latency_ms", "fps", "imgsz"])
        writer.writerow([args.model, backend, len(frame_times), f"{avg_latency_ms:.2f}", f"{fps:.2f}", args.imgsz])

    print(f"Frames processed: {len(frame_times)}")
    print(f"Average latency: {avg_latency_ms:.2f} ms")
    print(f"Approx FPS: {fps:.2f}")
    print(f"Device: {device}")
    print(f"Source: {used_source}")
    print(f"Saved benchmark to {output_path}")


if __name__ == "__main__":
    main()
