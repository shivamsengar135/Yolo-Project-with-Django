import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from device_utils import resolve_onnx_providers
from webcam_utils import open_webcam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webcam benchmark for YOLO ONNX models.")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model.")
    parser.add_argument("--source", type=str, default="auto", help="Webcam index or 'auto'.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to benchmark.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Execution provider preference: "auto", "cpu", or "cuda".',
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/metrics.csv",
        help="CSV path for benchmark results.",
    )
    return parser.parse_args()


def letterbox(image: np.ndarray, new_shape: int = 640) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(new_shape / height, new_shape / width)
    resized_w = int(round(width * scale))
    resized_h = int(round(height * scale))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    pad_w = (new_shape - resized_w) // 2
    pad_h = (new_shape - resized_h) // 2
    canvas[pad_h : pad_h + resized_h, pad_w : pad_w + resized_w] = resized
    return canvas


def preprocess(frame: np.ndarray, imgsz: int) -> np.ndarray:
    image = letterbox(frame, imgsz)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    return np.expand_dims(image, axis=0)


def infer_backend_name(model_path: str) -> str:
    return "onnx" if model_path.lower().endswith(".onnx") else "unknown"


def append_metrics_row(
    output_path: Path, model: str, frames: int, avg_latency_ms: float, fps: float, imgsz: int, backend: str
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(["model", "backend", "frames", "avg_latency_ms", "fps", "imgsz"])
        writer.writerow([model, backend, frames, f"{avg_latency_ms:.2f}", f"{fps:.2f}", imgsz])


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    providers = resolve_onnx_providers(args.device, ort.get_available_providers())
    session = ort.InferenceSession(str(model_path), providers=providers)
    input_name = session.get_inputs()[0].name
    cap, used_source = open_webcam(args.source)

    frame_times = []
    processed = 0
    while processed < args.frames:
        ok, frame = cap.read()
        if not ok:
            break

        tensor = preprocess(frame, args.imgsz)
        start = time.perf_counter()
        session.run(None, {input_name: tensor})
        elapsed = time.perf_counter() - start
        frame_times.append(elapsed)
        processed += 1

    cap.release()

    if not frame_times:
        raise RuntimeError("No frames were processed during ONNX benchmark.")

    avg_latency_ms = (sum(frame_times) / len(frame_times)) * 1000.0
    fps = 1.0 / (sum(frame_times) / len(frame_times))
    backend = infer_backend_name(args.model)
    output_path = Path(args.output)
    append_metrics_row(output_path, args.model, len(frame_times), avg_latency_ms, fps, args.imgsz, backend)

    print(f"Frames processed: {len(frame_times)}")
    print(f"Average latency: {avg_latency_ms:.2f} ms")
    print(f"Approx FPS: {fps:.2f}")
    print(f"Providers: {session.get_providers()}")
    print(f"Source: {used_source}")
    print(f"Saved benchmark to {output_path}")


if __name__ == "__main__":
    main()
