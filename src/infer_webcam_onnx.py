import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import yaml

from device_utils import resolve_onnx_providers
from webcam_utils import open_webcam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time webcam inference with a YOLO ONNX model.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to exported YOLO ONNX model.",
    )
    parser.add_argument("--source", type=str, default="auto", help="Webcam index or 'auto'.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Execution provider preference: "auto", "cpu", or "cuda".',
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/dataset.yaml",
        help="Dataset YAML used to load class names.",
    )
    return parser.parse_args()


def load_class_names(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    names = config.get("names", {})
    return [names[key] for key in sorted(names)]


def letterbox(image: np.ndarray, new_shape: int = 640) -> tuple[np.ndarray, float, tuple[float, float]]:
    height, width = image.shape[:2]
    scale = min(new_shape / height, new_shape / width)
    resized_w = int(round(width * scale))
    resized_h = int(round(height * scale))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    pad_w = (new_shape - resized_w) / 2
    pad_h = (new_shape - resized_h) / 2
    left = int(round(pad_w - 0.1))
    top = int(round(pad_h - 0.1))
    canvas[top : top + resized_h, left : left + resized_w] = resized
    return canvas, scale, (pad_w, pad_h)


def preprocess(frame: np.ndarray, imgsz: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    image, scale, pad = letterbox(frame, imgsz)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image, scale, pad


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = np.empty_like(boxes)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def scale_boxes(
    boxes: np.ndarray, scale: float, pad: tuple[float, float], frame_shape: tuple[int, int]
) -> np.ndarray:
    pad_w, pad_h = pad
    boxes[:, [0, 2]] -= pad_w
    boxes[:, [1, 3]] -= pad_h
    boxes /= scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, frame_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, frame_shape[0])
    return boxes


def postprocess(
    output: np.ndarray,
    scale: float,
    pad: tuple[float, float],
    frame_shape: tuple[int, int],
    conf_thres: float,
    iou_thres: float,
) -> list[tuple[np.ndarray, float, int]]:
    predictions = np.squeeze(output)
    if predictions.ndim == 3:
        predictions = predictions[0]
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    boxes = predictions[:, :4]
    class_scores = predictions[:, 4:]
    scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)

    keep = scores >= conf_thres
    boxes = boxes[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]
    if len(boxes) == 0:
        return []

    boxes = xywh_to_xyxy(boxes)
    boxes = scale_boxes(boxes, scale, pad, frame_shape)

    nms_boxes = boxes.astype(np.int32).tolist()
    indices = cv2.dnn.NMSBoxes(nms_boxes, scores.tolist(), conf_thres, iou_thres)
    if len(indices) == 0:
        return []

    results = []
    for idx in np.array(indices).flatten():
        results.append((boxes[idx], float(scores[idx]), int(class_ids[idx])))
    return results


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    class_names = load_class_names(Path(args.data))
    providers = resolve_onnx_providers(args.device, ort.get_available_providers())
    session = ort.InferenceSession(str(model_path), providers=providers)
    input_name = session.get_inputs()[0].name
    print(f"ONNX providers used: {session.get_providers()}")

    cap, _ = open_webcam(args.source)

    previous_time = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        input_tensor, scale, pad = preprocess(frame, args.imgsz)
        output = session.run(None, {input_name: input_tensor})[0]
        detections = postprocess(output, scale, pad, frame.shape[:2], args.conf, args.iou)

        for box, score, class_id in detections:
            x1, y1, x2, y2 = [int(v) for v in box]
            label = class_names[class_id] if class_id < len(class_names) else str(class_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
            cv2.putText(
                frame,
                f"{label}: {score:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 180, 0),
                2,
                cv2.LINE_AA,
            )

        current_time = time.time()
        fps = 1.0 / max(current_time - previous_time, 1e-6)
        previous_time = current_time
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("ONNX Webcam", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
