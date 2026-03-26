import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a markdown comparison report from benchmark CSV.")
    parser.add_argument(
        "--metrics",
        type=str,
        default="results/metrics.csv",
        help="Path to benchmark CSV.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/comparison_report.md",
        help="Path to generated markdown report.",
    )
    return parser.parse_args()


def safe_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def normalize_row(row: dict[str, str]) -> dict[str, str]:
    normalized = dict(row)
    if "backend" not in normalized:
        model = normalized.get("model", "")
        normalized["backend"] = "onnx" if model.lower().endswith(".onnx") else "pytorch"
    return normalized


def read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [normalize_row(row) for row in reader]


def best_by_backend(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    for row in rows:
        backend = row.get("backend", "unknown")
        fps = safe_float(row.get("fps", "nan"))
        if backend not in best or fps > safe_float(best[backend].get("fps", "nan")):
            best[backend] = row
    return best


def to_markdown(rows: list[dict[str, str]], best_rows: dict[str, dict[str, str]]) -> str:
    lines = []
    lines.append("# Benchmark Comparison Report")
    lines.append("")
    lines.append(f"Total benchmark runs: {len(rows)}")
    lines.append("")
    lines.append("## Best Run Per Backend")
    lines.append("")
    lines.append("| Backend | Model | FPS | Avg Latency (ms) | Frames | Image Size |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for backend, row in sorted(best_rows.items()):
        lines.append(
            f"| {backend} | {row.get('model', '')} | {row.get('fps', '')} | {row.get('avg_latency_ms', '')} | {row.get('frames', '')} | {row.get('imgsz', '')} |"
        )
    lines.append("")
    lines.append("## All Runs")
    lines.append("")
    lines.append("| Backend | Model | FPS | Avg Latency (ms) | Frames | Image Size |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row.get('backend', '')} | {row.get('model', '')} | {row.get('fps', '')} | {row.get('avg_latency_ms', '')} | {row.get('frames', '')} | {row.get('imgsz', '')} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        print(f"Metrics CSV not found: {metrics_path}")
        print("Run benchmark tasks first to generate metrics.")
        raise SystemExit(0)

    rows = read_metrics(metrics_path)
    if not rows:
        print(f"No rows found in {metrics_path}")
        print("Run benchmark tasks first to generate rows.")
        raise SystemExit(0)

    best_rows = best_by_backend(rows)
    report = to_markdown(rows, best_rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Saved comparison report to {output_path}")


if __name__ == "__main__":
    main()
