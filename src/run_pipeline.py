import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run common project tasks with one command.")
    parser.add_argument(
        "task",
        choices=[
            "status",
            "validate",
            "report",
            "train",
            "infer_yolo",
            "infer_ssd",
            "export_onnx",
            "infer_onnx",
            "benchmark",
            "benchmark_onnx",
            "benchmark_ssd",
            "compare",
            "plot",
            "full_eval",
            "full_eval_fast",
            "quick_report",
            "publish_results",
            "gui",
        ],
        help="Task to run.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="auto",
        help='Model path for relevant tasks. Use "auto" to resolve latest trained model.',
    )
    parser.add_argument("--epochs", type=int, default=30, help="Epochs for train task.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold.")
    parser.add_argument("--frames", type=int, default=100, help="Frames for benchmark task.")
    parser.add_argument("--source", type=str, default="auto", help="Webcam source index or 'auto'.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device preference for tasks: "auto", "cpu", "cuda", or explicit id where supported.',
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="default",
        choices=["default", "balanced", "minority_boost"],
        help="Augmentation preset for train task.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="results/metrics.csv",
        help="Metrics CSV path used by benchmark and compare tasks.",
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default="auto",
        help='Training results.csv path for plot task. Use "auto" to pick latest.',
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/full_eval",
        help="Output directory used by full_eval task.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> int:
    print("Running:", " ".join(command))
    result = subprocess.run(command)
    return result.returncode


def latest_file(project_root: Path, patterns: list[str]) -> Path | None:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(project_root.glob(pattern))
    files = [path for path in candidates if path.is_file()]
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)


def resolve_pt_model(project_root: Path, model_arg: str) -> str:
    if model_arg != "auto":
        return model_arg
    latest = latest_file(
        project_root,
        [
            "runs/detect/results/yolo_train/**/weights/best.pt",
            "results/yolo_train/**/weights/best.pt",
            "runs/detect/**/weights/best.pt",
        ],
    )
    return str(latest) if latest else "yolov8n.pt"


def resolve_onnx_model(project_root: Path, model_arg: str) -> str:
    if model_arg != "auto":
        return model_arg
    latest = latest_file(
        project_root,
        [
            "runs/detect/results/yolo_train/**/weights/best.onnx",
            "results/yolo_train/**/weights/best.onnx",
            "runs/detect/**/weights/best.onnx",
        ],
    )
    return str(latest) if latest else "best.onnx"


def resolve_results_csv(project_root: Path, results_csv_arg: str) -> str:
    if results_csv_arg != "auto":
        return results_csv_arg
    latest = latest_file(
        project_root,
        [
            "runs/detect/results/yolo_train/**/results.csv",
            "results/yolo_train/**/results.csv",
            "runs/detect/**/results.csv",
        ],
    )
    if latest:
        return str(latest)
    return "results/yolo_train/exp/results.csv"


def main() -> None:
    args = parse_args()
    py = args.python
    root = Path(__file__).resolve().parent
    project_root = root.parent
    pt_model = resolve_pt_model(project_root, args.model)
    onnx_model = resolve_onnx_model(project_root, args.model)
    results_csv = resolve_results_csv(project_root, args.results_csv)
    train_model = "yolov8n.pt" if args.model == "auto" else args.model

    tasks = {
        "status": [py, str(root / "project_status.py")],
        "validate": [py, str(root / "validate_dataset.py"), "--data", "data/dataset.yaml"],
        "report": [py, str(root / "dataset_report.py"), "--data", "data/dataset.yaml"],
        "train": [
            py,
            str(root / "train_yolo.py"),
            "--data",
            "data/dataset.yaml",
            "--weights",
            train_model,
            "--epochs",
            str(args.epochs),
            "--imgsz",
            str(args.imgsz),
            "--batch",
            "8",
            "--device",
            args.device,
            "--preset",
            args.preset,
        ],
        "infer_yolo": [
            py,
            str(root / "infer_webcam_yolo.py"),
            "--model",
            pt_model,
            "--imgsz",
            str(args.imgsz),
            "--conf",
            str(args.conf),
            "--source",
            str(args.source),
            "--device",
            args.device,
        ],
        "infer_ssd": [
            py,
            str(root / "infer_webcam_ssd.py"),
            "--source",
            str(args.source),
            "--device",
            args.device,
            "--threshold",
            "0.5",
        ],
        "export_onnx": [
            py,
            str(root / "export_yolo_onnx.py"),
            "--model",
            pt_model,
            "--imgsz",
            str(args.imgsz),
            "--device",
            args.device,
        ],
        "infer_onnx": [
            py,
            str(root / "infer_webcam_onnx.py"),
            "--model",
            onnx_model,
            "--imgsz",
            str(args.imgsz),
            "--conf",
            str(args.conf),
            "--source",
            str(args.source),
            "--device",
            args.device,
        ],
        "benchmark": [
            py,
            str(root / "benchmark.py"),
            "--model",
            pt_model,
            "--frames",
            str(args.frames),
            "--imgsz",
            str(args.imgsz),
            "--conf",
            str(args.conf),
            "--source",
            str(args.source),
            "--device",
            args.device,
            "--output",
            args.metrics,
        ],
        "benchmark_onnx": [
            py,
            str(root / "benchmark_onnx.py"),
            "--model",
            onnx_model,
            "--frames",
            str(args.frames),
            "--imgsz",
            str(args.imgsz),
            "--source",
            str(args.source),
            "--device",
            args.device,
            "--output",
            args.metrics,
        ],
        "benchmark_ssd": [
            py,
            str(root / "benchmark_ssd.py"),
            "--frames",
            str(args.frames),
            "--source",
            str(args.source),
            "--device",
            args.device,
            "--output",
            args.metrics,
        ],
        "compare": [py, str(root / "generate_comparison_report.py"), "--metrics", args.metrics],
        "plot": [py, str(root / "plot_training_results.py"), "--results-csv", results_csv],
        "full_eval": [
            py,
            str(root / "full_eval.py"),
            "--python",
            py,
            "--epochs",
            str(args.epochs),
            "--imgsz",
            str(args.imgsz),
            "--frames",
            str(args.frames),
            "--source",
            str(args.source),
            "--preset",
            args.preset,
            "--device",
            args.device,
            "--output-dir",
            args.output_dir,
        ],
        "full_eval_fast": [
            py,
            str(root / "full_eval.py"),
            "--python",
            py,
            "--epochs",
            str(args.epochs),
            "--imgsz",
            str(args.imgsz),
            "--frames",
            str(args.frames),
            "--source",
            str(args.source),
            "--preset",
            args.preset,
            "--device",
            args.device,
            "--output-dir",
            args.output_dir,
            "--fast",
        ],
        "quick_report": [
            py,
            str(root / "full_eval.py"),
            "--python",
            py,
            "--imgsz",
            str(args.imgsz),
            "--frames",
            str(args.frames),
            "--source",
            str(args.source),
            "--device",
            args.device,
            "--output-dir",
            args.output_dir,
            "--fast",
            "--skip-train",
            "--skip-export",
        ],
        "publish_results": [
            py,
            str(root / "publish_final_results.py"),
            "--full-eval-root",
            args.output_dir,
            "--output",
            "results/final_results.md",
            "--readme",
            "README.md",
        ],
        "gui": [py, str(root / "app_gui.py")],
    }

    code = run_command(tasks[args.task])
    raise SystemExit(code)


if __name__ == "__main__":
    main()
