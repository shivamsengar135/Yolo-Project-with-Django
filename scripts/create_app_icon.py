from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    out_dir = Path("assets")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "app_icon.ico"

    size = 256
    img = Image.new("RGBA", (size, size), (20, 28, 40, 255))
    draw = ImageDraw.Draw(img)

    # Rounded card background
    draw.rounded_rectangle((18, 18, 238, 238), radius=36, fill=(35, 120, 210, 255))
    draw.rounded_rectangle((34, 34, 222, 222), radius=30, fill=(13, 24, 45, 255))

    # Detection box motif
    draw.rectangle((56, 58, 198, 200), outline=(0, 220, 160, 255), width=10)
    draw.rectangle((86, 88, 168, 170), outline=(255, 190, 60, 255), width=8)

    # Corner accents
    draw.line((56, 58, 90, 58), fill=(255, 255, 255, 255), width=8)
    draw.line((56, 58, 56, 92), fill=(255, 255, 255, 255), width=8)
    draw.line((198, 200, 164, 200), fill=(255, 255, 255, 255), width=8)
    draw.line((198, 200, 198, 166), fill=(255, 255, 255, 255), width=8)

    img.save(out_path, format="ICO", sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
    print(f"Created icon: {out_path}")


if __name__ == "__main__":
    main()
