import argparse
import csv
import time
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)

from device_utils import resolve_torch_device
from webcam_utils import open_webcam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webcam benchmark for SSD Lite.")
    parser.add_argument("--source", type=str, default="auto", help="Webcam index or 'auto'.")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to benchmark.")
    parser.add_argument("--device", type=str, default="auto", help='Device, e.g. "auto", "cpu" or "cuda".')
    parser.add_argument(
        "--output",
        type=str,
        default="results/metrics.csv",
        help="CSV path for benchmark results.",
    )
    return parser.parse_args()


def append_metrics_row(output_path: Path, frames: int, avg_latency_ms: float, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(["model", "backend", "frames", "avg_latency_ms", "fps", "imgsz"])
        writer.writerow(
            ["ssdlite320_mobilenet_v3_large", "ssd", frames, f"{avg_latency_ms:.2f}", f"{fps:.2f}", 320]
        )


def main() -> None:
    args = parse_args()
    device_str = resolve_torch_device(args.device)
    device = torch.device(device_str)
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    preprocess = weights.transforms()

    model = ssdlite320_mobilenet_v3_large(weights=weights)
    model.to(device)
    model.eval()

    cap, used_source = open_webcam(args.source)

    frame_times = []
    processed = 0
    while processed < args.frames:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        tensor = preprocess(pil_image).unsqueeze(0).to(device)

        start = time.perf_counter()
        with torch.no_grad():
            _ = model(tensor)
        elapsed = time.perf_counter() - start
        frame_times.append(elapsed)
        processed += 1

    cap.release()

    if not frame_times:
        raise RuntimeError("No frames were processed during SSD benchmark.")

    avg_latency_ms = (sum(frame_times) / len(frame_times)) * 1000.0
    fps = 1.0 / (sum(frame_times) / len(frame_times))
    output_path = Path(args.output)
    append_metrics_row(output_path, len(frame_times), avg_latency_ms, fps)

    print(f"Frames processed: {len(frame_times)}")
    print(f"Average latency: {avg_latency_ms:.2f} ms")
    print(f"Approx FPS: {fps:.2f}")
    print(f"Device: {device}")
    print(f"Source: {used_source}")
    print(f"Saved benchmark to {output_path}")


if __name__ == "__main__":
    main()
