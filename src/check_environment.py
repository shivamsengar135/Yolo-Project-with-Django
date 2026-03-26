import importlib
import platform
import sys


PACKAGES = [
    "cv2",
    "torch",
    "torchvision",
    "ultralytics",
    "onnx",
    "onnxruntime",
    "yaml",
]


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    for package_name in PACKAGES:
        try:
            module = importlib.import_module(package_name)
            version = getattr(module, "__version__", "unknown")
            print(f"{package_name}: {version}")
        except Exception as exc:
            print(f"{package_name}: not available ({exc})")

    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except Exception:
        pass

    try:
        import onnxruntime as ort

        print(f"ONNX providers: {ort.get_available_providers()}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
