"""Remove the black matte from a supplied lockup, producing a transparent PNG.

`frontend/public/assets/new.png` arrived as RGB with no alpha: the artwork is white and brand
colours composited onto solid black. Placing it on the sign-in panel as-is would paste a black
rectangle over the background, so the matte has to come out.

The artwork is bright on black, which makes this exact rather than a guess: for a source pixel
composited as `src = colour * alpha` over black, the alpha is recoverable as the brightest
channel, and dividing it back out restores the original colour. Anti-aliased edges and the soft
outer glow therefore come through as genuine partial transparency instead of a hard cut-out.

    python server/scripts/make_logo_transparent.py

Input : frontend/public/assets/new.png
Output: frontend/public/assets/adviSU-logo-dark.png   (for dark grounds; white type is baked in)
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT_ROOT / "frontend" / "public" / "assets" / "new.png"
TARGET = PROJECT_ROOT / "frontend" / "public" / "assets" / "adviSU-logo-dark.png"

# Below this, a pixel is matte rather than artwork. Kept low so the outer glow survives; raising
# it starts biting into the soft edges of the descriptor type.
ALPHA_FLOOR = 8


def remove_black_matte(source: Path = SOURCE, target: Path = TARGET) -> Path:
    image = Image.open(source).convert("RGB")
    width, height = image.size
    pixels = image.load()

    out = Image.new("RGBA", (width, height))
    out_pixels = out.load()
    cleared = 0

    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            alpha = max(r, g, b)
            if alpha <= ALPHA_FLOOR:
                out_pixels[x, y] = (0, 0, 0, 0)
                cleared += 1
                continue
            # Un-premultiply: src = colour * (alpha/255)  ->  colour = src * 255 / alpha.
            scale = 255.0 / alpha
            out_pixels[x, y] = (
                min(255, int(r * scale)),
                min(255, int(g * scale)),
                min(255, int(b * scale)),
                alpha,
            )

    # Trim the transparent margin so the lockup fills its box at whatever size it is rendered.
    bbox = out.getbbox()
    if bbox:
        out = out.crop(bbox)

    target.parent.mkdir(parents=True, exist_ok=True)
    out.save(target, optimize=True)
    print(
        f"{target.relative_to(PROJECT_ROOT)}: {out.size[0]}x{out.size[1]}, "
        f"cleared {cleared} matte pixels of {width * height}"
    )
    return target


# The full lockup stacks three bands with clear transparent gutters between them:
#   y 0-463    book mark + "AdviSU" wordmark (+ cap)
#   y 536-649  "SABANCI UNIVERSITY"
#   y 697-791  "ACADEMIC ADVISOR"
# and inside the top band the book ends at x=592 before the "A" starts at x=644. Those
# boundaries are measured from the alpha profile of the generated file, not eyeballed, so
# re-running after new artwork should re-check them.
WORDMARK_BOTTOM = 500
MARK_RIGHT = 618

WORDMARK_TARGET = PROJECT_ROOT / "frontend" / "public" / "assets" / "adviSU-wordmark-dark.png"
MARK_TARGET = PROJECT_ROOT / "frontend" / "public" / "assets" / "adviSU-mark-dark.png"


def derive_compact_variants(source: Path = TARGET) -> None:
    """Cut the two smaller lockups the app needs from the full one.

    The sidebar renders its logo about 48px tall, at which height the stacked descriptor is a
    few pixels of illegible grey — so it needs the horizontal wordmark, not the full lockup.
    The chat empty state needs the book mark alone.
    """
    full = Image.open(source).convert("RGBA")

    wordmark = full.crop((0, 0, full.width, WORDMARK_BOTTOM))
    wordmark = wordmark.crop(wordmark.getbbox())
    wordmark.save(WORDMARK_TARGET, optimize=True)
    print(f"{WORDMARK_TARGET.relative_to(PROJECT_ROOT)}: {wordmark.size[0]}x{wordmark.size[1]}")

    mark = full.crop((0, 0, MARK_RIGHT, WORDMARK_BOTTOM))
    mark = mark.crop(mark.getbbox())
    mark.save(MARK_TARGET, optimize=True)
    print(f"{MARK_TARGET.relative_to(PROJECT_ROOT)}: {mark.size[0]}x{mark.size[1]}")


if __name__ == "__main__":
    remove_black_matte()
    derive_compact_variants()
