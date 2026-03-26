import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from device_utils import resolve_ultralytics_device
from webcam_utils import open_webcam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time webcam inference with YOLO.")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLO model (.pt or .onnx).",
    )
    parser.add_argument("--source", type=str, default="auto", help="Webcam index or 'auto'.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device to use: "auto", "cpu", "cuda", or explicit Ultralytics id like "0".',
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save output video to results/webcam_yolo.mp4.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_ultralytics_device(args.device)
    model_path = Path(args.model)
    if not model_path.exists() and args.model != "yolov8n.pt":
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(args.model)
    cap, _ = open_webcam(args.source)

    writer = None
    if args.save:
        out_path = Path("results/webcam_yolo.mp4")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        writer = cv2.VideoWriter(str(out_path), fourcc, 20.0, (width, height))

    prev_time = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=device,
            verbose=False,
        )
        annotated = results[0].plot()

        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time
        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if writer is not None:
            writer.write(annotated)

        cv2.imshow("YOLO Webcam", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
