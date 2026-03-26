import argparse
import csv
from datetime import datetime
from pathlib import Path


START_MARKER = "<!-- FINAL_RESULTS_START -->"
END_MARKER = "<!-- FINAL_RESULTS_END -->"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish latest full-eval results into markdown outputs.")
    parser.add_argument(
        "--full-eval-root",
        type=str,
        default="results/full_eval",
        help="Root folder containing full_eval_<timestamp> runs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/final_results.md",
        help="Output markdown file for final results.",
    )
    parser.add_argument(
        "--readme",
        type=str,
        default="README.md",
        help="README file to patch with final results section.",
    )
    return parser.parse_args()


def find_latest_eval_run(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Full-eval root not found: {root}")
    candidates = [path for path in root.iterdir() if path.is_dir() and path.name.startswith("full_eval_")]
    if not candidates:
        raise FileNotFoundError(f"No full_eval_* directories found under: {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def safe_read(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def best_by_backend(metrics_path: Path) -> list[dict[str, str]]:
    if not metrics_path.exists():
        return []
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    best: dict[str, dict[str, str]] = {}
    for row in rows:
        backend = row.get("backend", "unknown")
        fps = float(row.get("fps", "nan"))
        if backend not in best or fps > float(best[backend].get("fps", "nan")):
            best[backend] = row
    return [best[key] for key in sorted(best.keys())]


def build_final_results_md(
    run_dir: Path,
    full_summary_path: Path,
    comparison_path: Path,
    curves_path: Path,
    metrics_path: Path,
) -> str:
    lines: list[str] = []
    lines.append("# Final Results")
    lines.append("")
    lines.append(f"Generated: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append(f"Latest full-eval run: `{run_dir}`")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Full eval summary: `{full_summary_path}`")
    lines.append(f"- Comparison report: `{comparison_path}`")
    lines.append(f"- Training curves: `{curves_path}`")
    lines.append(f"- Metrics CSV: `{metrics_path}`")
    lines.append("")

    best_rows = best_by_backend(metrics_path)
    if best_rows:
        lines.append("## Best Benchmark Per Backend")
        lines.append("")
        lines.append("| Backend | Model | FPS | Avg Latency (ms) | Frames | Image Size |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for row in best_rows:
            lines.append(
                f"| {row.get('backend', '')} | {row.get('model', '')} | {row.get('fps', '')} | {row.get('avg_latency_ms', '')} | {row.get('frames', '')} | {row.get('imgsz', '')} |"
            )
        lines.append("")

    full_summary = safe_read(full_summary_path)
    if full_summary:
        lines.append("## Full Eval Summary Copy")
        lines.append("")
        lines.append(full_summary.strip())
        lines.append("")

    return "\n".join(lines)


def patch_readme(readme_path: Path, section_md: str) -> None:
    original = safe_read(readme_path)
    block = f"{START_MARKER}\n{section_md}\n{END_MARKER}"
    if START_MARKER in original and END_MARKER in original:
        start = original.index(START_MARKER)
        end = original.index(END_MARKER) + len(END_MARKER)
        updated = original[:start] + block + original[end:]
    else:
        suffix = "\n\n## Published Final Results\n\n" + block + "\n"
        updated = original.rstrip() + suffix
    readme_path.write_text(updated, encoding="utf-8")


def main() -> None:
    args = parse_args()
    full_eval_root = Path(args.full_eval_root)
    run_dir = find_latest_eval_run(full_eval_root)

    full_summary_path = run_dir / "full_eval_summary.md"
    comparison_path = run_dir / "comparison_report.md"
    curves_path = run_dir / "training_curves.png"
    metrics_path = run_dir / "metrics.csv"

    final_results_md = build_final_results_md(
        run_dir,
        full_summary_path,
        comparison_path,
        curves_path,
        metrics_path,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(final_results_md, encoding="utf-8")
    print(f"Saved final results markdown: {output_path}")

    readme_path = Path(args.readme)
    patch_readme(readme_path, final_results_md)
    print(f"Patched README final results section: {readme_path}")


if __name__ == "__main__":
    main()
