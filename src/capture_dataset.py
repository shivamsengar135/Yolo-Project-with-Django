import argparse
from pathlib import Path

import cv2

from webcam_utils import open_webcam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture labeled images from webcam.")
    parser.add_argument(
        "--class-name",
        type=str,
        required=True,
        help="Class name used in the output filename, e.g. bottle.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to save images into.",
    )
    parser.add_argument("--source", type=str, default="auto", help="Webcam index or 'auto'.")
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Number of images to capture.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/images",
        help="Base output image directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    cap, _ = open_webcam(args.source)

    saved = 0
    print("Press space to save a frame. Press q to quit.")
    while saved < args.count:
        ok, frame = cap.read()
        if not ok:
            break

        preview = frame.copy()
        cv2.putText(
            preview,
            f"{args.class_name} | saved {saved}/{args.count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Capture Dataset", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            file_path = output_dir / f"{args.class_name}_{saved:04d}.jpg"
            cv2.imwrite(str(file_path), frame)
            saved += 1
            print(f"Saved {file_path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished with {saved} images saved to {output_dir}")


if __name__ == "__main__":
    main()
