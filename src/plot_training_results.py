import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot YOLO training metrics from results.csv.")
    parser.add_argument(
        "--results-csv",
        type=str,
        default="results/yolo_train/exp/results.csv",
        help="Path to YOLO results.csv file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/training_curves.png",
        help="Output PNG path.",
    )
    return parser.parse_args()


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def parse_float_column(rows: list[dict[str, str]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key, "").strip()
        if not value:
            values.append(float("nan"))
            continue
        values.append(float(value))
    return values


def pick_column(fieldnames: list[str], preferred: str) -> str | None:
    if preferred in fieldnames:
        return preferred
    for key in fieldnames:
        if key.strip() == preferred:
            return key
    return None


def main() -> None:
    args = parse_args()
    csv_path = Path(args.results_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Training results not found: {csv_path}")

    rows = load_rows(csv_path)
    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")

    fieldnames = list(rows[0].keys())
    epoch_key = pick_column(fieldnames, "epoch")
    precision_key = pick_column(fieldnames, "metrics/precision(B)")
    recall_key = pick_column(fieldnames, "metrics/recall(B)")
    map50_key = pick_column(fieldnames, "metrics/mAP50(B)")
    map5095_key = pick_column(fieldnames, "metrics/mAP50-95(B)")
    box_loss_key = pick_column(fieldnames, "train/box_loss")
    cls_loss_key = pick_column(fieldnames, "train/cls_loss")
    dfl_loss_key = pick_column(fieldnames, "train/dfl_loss")

    epochs = parse_float_column(rows, epoch_key) if epoch_key else list(range(len(rows)))
    precision = parse_float_column(rows, precision_key) if precision_key else []
    recall = parse_float_column(rows, recall_key) if recall_key else []
    map50 = parse_float_column(rows, map50_key) if map50_key else []
    map5095 = parse_float_column(rows, map5095_key) if map5095_key else []
    box_loss = parse_float_column(rows, box_loss_key) if box_loss_key else []
    cls_loss = parse_float_column(rows, cls_loss_key) if cls_loss_key else []
    dfl_loss = parse_float_column(rows, dfl_loss_key) if dfl_loss_key else []

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    if precision:
        plt.plot(epochs, precision, label="Precision")
    if recall:
        plt.plot(epochs, recall, label="Recall")
    if map50:
        plt.plot(epochs, map50, label="mAP50")
    if map5095:
        plt.plot(epochs, map5095, label="mAP50-95")
    plt.title("Detection Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 1, 2)
    if box_loss:
        plt.plot(epochs, box_loss, label="box_loss")
    if cls_loss:
        plt.plot(epochs, cls_loss, label="cls_loss")
    if dfl_loss:
        plt.plot(epochs, dfl_loss, label="dfl_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved training curves to {output_path}")


if __name__ == "__main__":
    main()
