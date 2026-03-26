import argparse
import time

import cv2
import torch
from PIL import Image
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)

from device_utils import resolve_torch_device
from webcam_utils import open_webcam


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time webcam inference with SSD Lite.")
    parser.add_argument("--source", type=str, default="auto", help="Webcam index or 'auto'.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Score threshold.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device to use: "auto", "cpu", or "cuda".',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device_str = resolve_torch_device(args.device)
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=weights)
    model.eval()

    device = torch.device(device_str)
    model.to(device)
    preprocess = weights.transforms()
    print(f"SSD device used: {device}")

    cap, _ = open_webcam(args.source)

    previous_time = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        tensor = preprocess(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(tensor)[0]

        for box, label, score in zip(
            prediction["boxes"], prediction["labels"], prediction["scores"]
        ):
            if float(score) < args.threshold:
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

        cv2.imshow("SSD Webcam", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
