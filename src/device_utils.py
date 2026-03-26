from __future__ import annotations

import torch


def has_cuda() -> bool:
    return torch.cuda.is_available()


def resolve_torch_device(device: str) -> str:
    requested = device.lower()
    if requested == "auto":
        resolved = "cuda" if has_cuda() else "cpu"
        if resolved == "cpu":
            print("CUDA not available, falling back to CPU.")
        return resolved
    if requested in {"cuda", "gpu"}:
        if has_cuda():
            return "cuda"
        print('Requested CUDA but CUDA is not available, falling back to CPU.')
        return "cpu"
    return device


def resolve_ultralytics_device(device: str) -> str:
    requested = device.lower()
    if requested == "auto":
        resolved = "0" if has_cuda() else "cpu"
        if resolved == "cpu":
            print("CUDA not available, falling back to CPU.")
        return resolved
    if requested in {"cuda", "gpu"}:
        if has_cuda():
            return "0"
        print('Requested CUDA but CUDA is not available, falling back to CPU.')
        return "cpu"
    return device


def resolve_onnx_providers(device: str, available: list[str]) -> list[str]:
    requested = device.lower()
    if requested in {"auto", "cuda", "gpu"} and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CPUExecutionProvider" in available:
        return ["CPUExecutionProvider"]
    return available
