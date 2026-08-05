"""Regenerate the dark-ground (knockout) variant of the adviSU lockup.

The supplied lockup (`frontend/public/assets/big.png`) is drawn in navy ink for a light
ground. Its background is already transparent, but on the navy login panel the descriptor
type ("SABANCI UNIVERSITY / ACADEMIC ADVISOR") and the "SU" glyphs disappear into the
background. Painting a white plate behind the logo hides the problem and looks cheap.

This script produces the correct answer instead: remap only the *navy* ink to near-white and
leave the gold / teal / light-blue accents alone, so the mark keeps its identity on a dark
surface. Alpha is preserved untouched, so anti-aliased edges stay clean.

    python server/scripts/make_logo_knockout.py

Input : frontend/public/assets/big.png
Output: frontend/public/assets/adviSU-logo-knockout.png
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT_ROOT / "frontend" / "public" / "assets" / "big.png"
TARGET = PROJECT_ROOT / "frontend" / "public" / "assets" / "adviSU-logo-knockout.png"

# A pixel is "navy ink" when it is dark AND at least as blue as it is red. That keeps the
# brand accents intact: gold (214,161,58) is red-dominant, teal (96,144,144) and the light
# blue (48,144,192) are both above the luminance cut.
LUMINANCE_CUT = 120.0
MIN_BLUE_OVER_RED = 0


def _luminance(r: int, g: int, b: int) -> float:
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def build_knockout(source: Path = SOURCE, target: Path = TARGET) -> Path:
    image = Image.open(source).convert("RGBA")
    pixels = image.load()
    width, height = image.size
    remapped = 0

    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if a == 0:
                continue
            luminance = _luminance(r, g, b)
            if luminance >= LUMINANCE_CUT or (b - r) < MIN_BLUE_OVER_RED:
                continue
            # Keep a little of the original modelling instead of flattening to pure white:
            # the darkest ink becomes the brightest, so the mark still reads as drawn.
            value = int(255 - (luminance / LUMINANCE_CUT) * 22)
            pixels[x, y] = (value, value, value, a)
            remapped += 1

    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, optimize=True)
    total = width * height
    print(f"{target.relative_to(PROJECT_ROOT)}: remapped {remapped} of {total} pixels")
    return target


if __name__ == "__main__":
    build_knockout()
