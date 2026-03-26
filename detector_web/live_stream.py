from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
import yaml
from PIL import Image
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)
from ultralytics import YOLO


COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def _parse_source(source: str) -> int | None:
    raw = str(source).strip().lower()
    if raw == "auto":
        return None
    return int(raw)


def _try_open(index: int) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        return cap
    cap.release()
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.isOpened():
        return cap
    cap.release()
    return None


def _open_webcam(source: str, max_index: int = 5) -> cv2.VideoCapture:
    parsed = _parse_source(source)
    if parsed is not None:
        cap = _try_open(parsed)
        if cap is None:
            raise RuntimeError(f"Could not open webcam index {parsed}")
        return cap
    for idx in range(max_index + 1):
        cap = _try_open(idx)
        if cap is not None:
            return cap
    raise RuntimeError("Could not open any webcam in range 0..5")


def _resolve_torch_device(device: str) -> str:
    requested = device.lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested in {"cuda", "gpu"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _resolve_ultra_device(device: str) -> str:
    requested = device.lower()
    if requested == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    if requested in {"cuda", "gpu"}:
        return "0" if torch.cuda.is_available() else "cpu"
    return requested


def _resolve_onnx_providers(device: str) -> list[str]:
    available = ort.get_available_providers()
    requested = device.lower()
    if requested in {"auto", "cuda", "gpu"} and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CPUExecutionProvider" in available:
        return ["CPUExecutionProvider"]
    return available


def _load_class_names(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    names = cfg.get("names", {})
    return [names[key] for key in sorted(names)]


def _letterbox(image: np.ndarray, new_shape: int = 640) -> tuple[np.ndarray, float, tuple[float, float]]:
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


def _preprocess_onnx(frame: np.ndarray, imgsz: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    image, scale, pad = _letterbox(frame, imgsz)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image, scale, pad


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = np.empty_like(boxes)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def _scale_boxes(
    boxes: np.ndarray, scale: float, pad: tuple[float, float], frame_shape: tuple[int, int]
) -> np.ndarray:
    pad_w, pad_h = pad
    boxes[:, [0, 2]] -= pad_w
    boxes[:, [1, 3]] -= pad_h
    boxes /= scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, frame_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, frame_shape[0])
    return boxes


def _postprocess_onnx(
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
    boxes = _xywh_to_xyxy(boxes)
    boxes = _scale_boxes(boxes, scale, pad, frame_shape)
    nms_boxes = boxes.astype(np.int32).tolist()
    indices = cv2.dnn.NMSBoxes(nms_boxes, scores.tolist(), conf_thres, iou_thres)
    if len(indices) == 0:
        return []
    out: list[tuple[np.ndarray, float, int]] = []
    for idx in np.array(indices).flatten():
        out.append((boxes[idx], float(scores[idx]), int(class_ids[idx])))
    return out


class LiveDetectionService:
    def __init__(self) -> None:
        self._stream_lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._yolo_models: dict[str, YOLO] = {}
        self._onnx_sessions: dict[tuple[str, str], tuple[ort.InferenceSession, str, list[str]]] = {}
        self._ssd_cache: dict[str, tuple[torch.nn.Module, object]] = {}
        self.project_root = Path(__file__).resolve().parent.parent

    def _yolo_model(self, model: str) -> YOLO:
        with self._model_lock:
            if model not in self._yolo_models:
                self._yolo_models[model] = YOLO(model)
            return self._yolo_models[model]

    def _onnx_session(self, model: str, device: str) -> tuple[ort.InferenceSession, str, list[str]]:
        key = (model, device.lower())
        with self._model_lock:
            if key not in self._onnx_sessions:
                providers = _resolve_onnx_providers(device)
                session = ort.InferenceSession(model, providers=providers)
                input_name = session.get_inputs()[0].name
                class_names = _load_class_names(self.project_root / "data" / "dataset.yaml")
                self._onnx_sessions[key] = (session, input_name, class_names)
            return self._onnx_sessions[key]

    def _ssd_model(self, device: str) -> tuple[torch.nn.Module, object, torch.device]:
        resolved = _resolve_torch_device(device)
        with self._model_lock:
            if resolved not in self._ssd_cache:
                weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
                model = ssdlite320_mobilenet_v3_large(weights=weights)
                model.eval()
                preprocess = weights.transforms()
                model = model.to(torch.device(resolved))
                self._ssd_cache[resolved] = (model, preprocess)
            model, preprocess = self._ssd_cache[resolved]
        return model, preprocess, torch.device(resolved)

    def _encode_frame(self, frame: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return b""
        return (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        )

    def stream(self, backend: str, *, model: str, source: str, device: str, imgsz: int, conf: float):
        if not self._stream_lock.acquire(blocking=False):
            raise RuntimeError("Another live stream is already active. Stop it first.")

        cap: cv2.VideoCapture | None = None
        try:
            cap = _open_webcam(source)
            prev = time.time()
            backend_name = backend.lower()

            if backend_name == "yolo":
                detector = self._yolo_model(model)
                device_name = _resolve_ultra_device(device)
            elif backend_name == "onnx":
                session, input_name, class_names = self._onnx_session(model, device)
            elif backend_name == "ssd":
                detector, preprocess, torch_device = self._ssd_model(device)
            else:
                raise RuntimeError(f"Unsupported backend '{backend}'.")

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if backend_name == "yolo":
                    results = detector.predict(
                        frame,
                        imgsz=imgsz,
                        conf=conf,
                        iou=0.45,
                        device=device_name,
                        verbose=False,
                    )
                    frame = results[0].plot()
                elif backend_name == "onnx":
                    input_tensor, scale, pad = _preprocess_onnx(frame, imgsz)
                    output = session.run(None, {input_name: input_tensor})[0]
                    detections = _postprocess_onnx(output, scale, pad, frame.shape[:2], conf, 0.45)
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
                else:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(rgb)
                    tensor = preprocess(pil).unsqueeze(0).to(torch_device)
                    with torch.no_grad():
                        pred = detector(tensor)[0]
                    for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
                        if float(score) < conf:
                            continue
                        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                        class_name = COCO_INSTANCE_CATEGORY_NAMES[int(label)]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                        cv2.putText(
                            frame,
                            f"{class_name}: {float(score):.2f}",
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 200, 255),
                            2,
                            cv2.LINE_AA,
                        )

                now = time.time()
                fps = 1.0 / max(now - prev, 1e-6)
                prev = now
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

                chunk = self._encode_frame(frame)
                if chunk:
                    yield chunk
        finally:
            if cap is not None:
                cap.release()
            self._stream_lock.release()


live_service = LiveDetectionService()
