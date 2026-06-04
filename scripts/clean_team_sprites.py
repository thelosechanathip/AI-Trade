from pathlib import Path

from PIL import Image


SRC_DIR = Path("static/assets/team-sprites")
OUT_DIR = Path("static/assets/team-sprites-clean")
FRAME_W = 96
FRAME_H = 112
COLS = 4
ROWS = 6
EDGE_PAD = 8


def clean_sheet(name: str) -> None:
    src = SRC_DIR / f"{name}.png"
    out = OUT_DIR / f"{name}.png"
    sheet = Image.open(src).convert("RGBA")
    clean = Image.new("RGBA", sheet.size, (0, 0, 0, 0))

    for row in range(ROWS):
        for col in range(COLS):
            left = col * FRAME_W
            top = row * FRAME_H
            frame = sheet.crop((left, top, left + FRAME_W, top + FRAME_H))
            pixels = frame.load()

            for x in range(FRAME_W):
                for y in range(FRAME_H):
                    if (
                        x < EDGE_PAD
                        or y < EDGE_PAD
                        or x >= FRAME_W - EDGE_PAD
                        or y >= FRAME_H - EDGE_PAD
                    ):
                        pixels[x, y] = (0, 0, 0, 0)

            clean.alpha_composite(frame, (left, top))

    clean.save(out)
    print(f"cleaned {name}: {clean.size}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("dex", "jame", "nova", "risk", "vector"):
        clean_sheet(name)


if __name__ == "__main__":
    main()
