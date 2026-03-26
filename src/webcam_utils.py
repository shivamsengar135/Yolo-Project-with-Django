from __future__ import annotations

import cv2


def parse_source(source: str) -> int | None:
    raw = str(source).strip().lower()
    if raw == "auto":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid source '{source}'. Use an integer index or 'auto'.") from exc


def _try_open(index: int) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        return cap
    cap.release()

    # Windows fallback backend can help when default backend fails.
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.isOpened():
        return cap
    cap.release()
    return None


def open_webcam(source: str, max_index: int = 5) -> tuple[cv2.VideoCapture, int]:
    parsed = parse_source(source)
    if parsed is not None:
        cap = _try_open(parsed)
        if cap is None:
            raise RuntimeError(f"Could not open webcam index {parsed}")
        print(f"Webcam source used: {parsed}")
        return cap, parsed

    for index in range(max_index + 1):
        cap = _try_open(index)
        if cap is not None:
            print(f"Webcam source auto-selected: {index}")
            return cap, index

    raise RuntimeError(f"Could not open any webcam in index range 0..{max_index}")
