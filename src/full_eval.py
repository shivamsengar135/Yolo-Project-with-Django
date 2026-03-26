import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full evaluation pipeline: train, export, benchmark, compare, and summarize."
    )
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable to use.")
    parser.add_argument("--data", type=str, default="data/dataset.yaml", help="Dataset YAML path.")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Starting weights for training.")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--frames", type=int, default=100, help="Frames per benchmark run.")
    parser.add_argument("--source", type=str, default="auto", help="Webcam source index or 'auto'.")
    parser.add_argument(
        "--preset",
        type=str,
        default="minority_boost",
        choices=["default", "balanced", "minority_boost"],
        help="Augmentation preset for training.",
    )
    parser.add_argument("--device", type=str, default="auto", help='Device: "auto", "cpu", "cuda", or id.')
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: uses shorter defaults for quicker iteration.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and reuse latest best.pt.",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip ONNX export and reuse latest best.onnx.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional fixed run name. If empty, a timestamped name is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/full_eval",
        help="Directory for full-eval outputs and summary file.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    result = subprocess.run(command)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")


def latest_file(project_root: Path, patterns: list[str]) -> Path | None:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(project_root.glob(pattern))
    files = [path for path in candidates if path.is_file()]
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)


def read_last_row(csv_path: Path) -> dict[str, str]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    return rows[-1]


def read_metrics_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_summary(
    summary_path: Path,
    run_name: str,
    best_pt: Path,
    best_onnx: Path,
    training_csv: Path,
    metrics_csv: Path,
    compare_md: Path,
    curves_png: Path,
) -> None:
    last_train = read_last_row(training_csv) if training_csv.exists() else {}
    metrics_rows = read_metrics_rows(metrics_csv) if metrics_csv.exists() else []

    lines: list[str] = []
    lines.append("# Full Evaluation Summary")
    lines.append("")
    lines.append(f"Run name: `{run_name}`")
    lines.append(f"Best PT model: `{best_pt}`")
    lines.append(f"Best ONNX model: `{best_onnx}`")
    lines.append(f"Training CSV: `{training_csv}`")
    lines.append(f"Benchmark CSV: `{metrics_csv}`")
    lines.append(f"Comparison report: `{compare_md}`")
    lines.append(f"Training curves image: `{curves_png}`")
    lines.append("")

    if last_train:
        lines.append("## Final Training Metrics")
        lines.append("")
        lines.append(f"- epoch: `{last_train.get('epoch', 'n/a')}`")
        lines.append(f"- precision: `{last_train.get('metrics/precision(B)', 'n/a')}`")
        lines.append(f"- recall: `{last_train.get('metrics/recall(B)', 'n/a')}`")
        lines.append(f"- mAP50: `{last_train.get('metrics/mAP50(B)', 'n/a')}`")
        lines.append(f"- mAP50-95: `{last_train.get('metrics/mAP50-95(B)', 'n/a')}`")
        lines.append("")

    if metrics_rows:
        lines.append("## Benchmark Rows")
        lines.append("")
        lines.append("| backend | model | fps | avg_latency_ms | frames | imgsz |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for row in metrics_rows:
            lines.append(
                f"| {row.get('backend', '')} | {row.get('model', '')} | {row.get('fps', '')} | {row.get('avg_latency_ms', '')} | {row.get('frames', '')} | {row.get('imgsz', '')} |"
            )
        lines.append("")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved full-eval summary to {summary_path}")


def main() -> None:
    args = parse_args()
    py = args.python
    root = Path(__file__).resolve().parent
    project_root = root.parent

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name.strip() or f"full_eval_{timestamp}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = args.epochs
    frames = args.frames
    preset = args.preset
    if args.fast:
        if epochs == 40:
            epochs = 10
        if frames == 100:
            frames = 30
        if preset == "minority_boost":
            preset = "balanced"

    best_pt: Path | None = None
    if args.skip_train:
        best_pt = latest_file(
            project_root,
            ["runs/detect/results/yolo_train/**/weights/best.pt", "runs/detect/**/weights/best.pt"],
        )
        if best_pt is None:
            raise FileNotFoundError("Could not locate existing best.pt while --skip-train is enabled.")
        print(f"Skipping training, using latest model: {best_pt}")
    else:
        train_cmd = [
            py,
            str(root / "train_yolo.py"),
            "--data",
            args.data,
            "--weights",
            args.weights,
            "--epochs",
            str(epochs),
            "--imgsz",
            str(args.imgsz),
            "--batch",
            "8",
            "--device",
            args.device,
            "--preset",
            preset,
            "--name",
            run_name,
        ]
        run_command(train_cmd)
        best_pt = latest_file(
            project_root,
            [
                f"runs/detect/results/yolo_train/{run_name}/weights/best.pt",
                "runs/detect/results/yolo_train/**/weights/best.pt",
            ],
        )
        if best_pt is None:
            raise FileNotFoundError("Could not locate trained best.pt after training.")

    best_onnx: Path | None = None
    if args.skip_export:
        best_onnx = latest_file(
            project_root,
            ["runs/detect/results/yolo_train/**/weights/best.onnx", "runs/detect/**/weights/best.onnx"],
        )
        if best_onnx is None:
            raise FileNotFoundError("Could not locate existing best.onnx while --skip-export is enabled.")
        print(f"Skipping ONNX export, using latest model: {best_onnx}")
    else:
        export_cmd = [
            py,
            str(root / "export_yolo_onnx.py"),
            "--model",
            str(best_pt),
            "--imgsz",
            str(args.imgsz),
            "--device",
            args.device,
        ]
        run_command(export_cmd)
        best_onnx = best_pt.with_suffix(".onnx")
        if not best_onnx.exists():
            fallback_onnx = latest_file(
                project_root,
                [f"runs/detect/results/yolo_train/{run_name}/weights/best.onnx", "runs/detect/**/weights/best.onnx"],
            )
            if fallback_onnx is None:
                raise FileNotFoundError("Could not locate best.onnx after export.")
            best_onnx = fallback_onnx

    metrics_csv = output_dir / "metrics.csv"
    benchmark_cmd = [
        py,
        str(root / "benchmark.py"),
        "--model",
        str(best_pt),
        "--source",
        str(args.source),
        "--frames",
        str(frames),
        "--imgsz",
        str(args.imgsz),
        "--conf",
        "0.4",
        "--device",
        args.device,
        "--output",
        str(metrics_csv),
    ]
    run_command(benchmark_cmd)

    benchmark_onnx_cmd = [
        py,
        str(root / "benchmark_onnx.py"),
        "--model",
        str(best_onnx),
        "--source",
        str(args.source),
        "--frames",
        str(frames),
        "--imgsz",
        str(args.imgsz),
        "--device",
        args.device,
        "--output",
        str(metrics_csv),
    ]
    run_command(benchmark_onnx_cmd)

    benchmark_ssd_cmd = [
        py,
        str(root / "benchmark_ssd.py"),
        "--source",
        str(args.source),
        "--frames",
        str(frames),
        "--device",
        args.device,
        "--output",
        str(metrics_csv),
    ]
    run_command(benchmark_ssd_cmd)

    compare_md = output_dir / "comparison_report.md"
    compare_cmd = [
        py,
        str(root / "generate_comparison_report.py"),
        "--metrics",
        str(metrics_csv),
        "--output",
        str(compare_md),
    ]
    run_command(compare_cmd)

    training_csv = best_pt.parent.parent / "results.csv"
    curves_png = output_dir / "training_curves.png"
    plot_cmd = [
        py,
        str(root / "plot_training_results.py"),
        "--results-csv",
        str(training_csv),
        "--output",
        str(curves_png),
    ]
    run_command(plot_cmd)

    summary_md = output_dir / "full_eval_summary.md"
    write_summary(summary_md, run_name, best_pt, best_onnx, training_csv, metrics_csv, compare_md, curves_png)


if __name__ == "__main__":
    main()
