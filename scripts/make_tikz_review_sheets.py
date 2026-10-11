#!/usr/bin/env python3
"""Compose side-by-side PNG comparisons of TikZ previews vs. canonical figures.

For each declared pair, the standalone TikZ PDF (``figures_tikz/*.pdf``) is
rasterized with ``pdftoppm`` at 150 dpi and placed next to the canonical
manuscript raster (``figures/*.png``), scaled to a common height, under a
labeled header. The output sheets are *visual review aids only*: they are not
manuscript figures, carry no physics content, and are gitignored under
``figures_tikz/review/``.

Pairs are declared with verified caption anchors so a reviewer can check,
panel by panel, that the vector reproduction matches the canonical raster in
content (labels, junction counts, operating-point notes). A structural sanity
gate verifies that every declared pair input exists before rendering.

The script is deterministic: the rendered pixels depend only on the input PDF
and raster (both byte-deterministic artifacts), and no timestamps or random
seeds enter the composition.

Args:
    None (no CLI arguments by design; configuration is centralized below).

Returns:
    None. Writes PNG comparison sheets to ``figures_tikz/review/`` and prints
    a per-pair summary line (exit code 1 if any pair failed).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TIKZ_DIR = PACKAGE_ROOT / "figures_tikz"
FIG_DIR = PACKAGE_ROOT / "figures"
OUT_DIR = TIKZ_DIR / "review"

RENDER_DPI = 150
TARGET_HEIGHT = 700  # common pixel height for both panels of a sheet
HEADER_HEIGHT = 60
MARGIN = 20
GUTTER = 30

HEADER_TEXT_COLOR = (20, 20, 20)
HEADER_BG_COLOR = (238, 238, 238)
BORDER_COLOR = (180, 180, 180)

# (tikz_pdf_stem, canonical_png_name, header_label)
PAIRS: list[tuple[str, str, str]] = [
    (
        "fig_setup_schematic",
        "rabi_setup_schematic.png",
        "Main FIG. 1 setup schematic: TikZ preview (left) vs canonical raster (right)",
    ),
    (
        "fig_platform_flux_qubit",
        "platform_flux_qubit.png",
        "SI FIG. S2 flux-qubit platform: TikZ preview (left) vs canonical raster (right)",
    ),
    (
        "fig_platform_transmon",
        "platform_transmon.png",
        "SI FIG. S3 transmon platform: TikZ preview (left) vs canonical raster (right)",
    ),
    (
        "fig_platform_fluxonium",
        "platform_fluxonium.png",
        "SI FIG. S4 fluxonium platform: TikZ preview (left) vs canonical raster (right)",
    ),
]


def render_pdf_to_png(pdf_path: Path, out_png: Path, dpi: int) -> None:
    """Rasterize a single-page PDF to PNG using pdftoppm.

    Args:
        pdf_path: Path to the single-page PDF to rasterize.
        out_png: Destination path for the rendered PNG (150 dpi default).
        dpi: Rasterization resolution in dots per inch.

    Returns:
        None. Raises RuntimeError if pdftoppm fails.
    """
    cmd = [
        "pdftoppm",
        "-png",
        "-r",
        str(dpi),
        "-singlefile",
        str(pdf_path),
        str(out_png.with_suffix("")),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not out_png.exists():
        raise RuntimeError(f"pdftoppm failed for {pdf_path.name}: {result.stderr.strip()}")


def load_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a TrueType font for sheet headers, with a bitmap fallback.

    Returns:
        ImageFont.FreeTypeFont: DejaVu Sans Bold 22 pt when available.
        ImageFont.ImageFont: Default bitmap font otherwise.
    """
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, 22)
    return ImageFont.load_default()


def scaled_to_height(image: Image.Image, height: int) -> Image.Image:
    """Return a copy of ``image`` resized to the requested height, keeping aspect.

    Args:
        image: Source image.
        height: Target pixel height.

    Returns:
        Image.Image: Resized copy (LANCZOS resampling).
    """
    width = round(image.width * height / image.height)
    return image.resize((width, height), Image.LANCZOS)


def compose_sheet(
    left: Image.Image,
    right: Image.Image,
    left_label: str,
    right_label: str,
    header: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    out_path: Path,
) -> None:
    """Compose one side-by-side sheet with a header and per-panel labels.

    Args:
        left: Left panel image (TikZ preview render).
        right: Right panel image (canonical raster).
        left_label: Small caption printed under the left panel.
        right_label: Small caption printed under the right panel.
        header: One-line sheet title drawn in the header band.
        font: Font used for the header text.
        out_path: Destination PNG path.

    Returns:
        None.
    """
    left_s = scaled_to_height(left, TARGET_HEIGHT)
    right_s = scaled_to_height(right, TARGET_HEIGHT)

    sheet_w = MARGIN + left_s.width + GUTTER + right_s.width + MARGIN
    sheet_h = HEADER_HEIGHT + MARGIN + TARGET_HEIGHT + 30 + MARGIN
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")

    draw = ImageDraw.Draw(sheet)
    draw.rectangle([0, 0, sheet_w, HEADER_HEIGHT], fill=HEADER_BG_COLOR)
    draw.text((MARGIN, 16), header, fill=HEADER_TEXT_COLOR, font=font)
    draw.line([0, HEADER_HEIGHT, sheet_w, HEADER_HEIGHT], fill=BORDER_COLOR, width=2)

    x = MARGIN
    sheet.paste(left_s, (x, HEADER_HEIGHT + MARGIN))
    draw.rectangle(
        [x - 1, HEADER_HEIGHT + MARGIN - 1, x + left_s.width, HEADER_HEIGHT + MARGIN + TARGET_HEIGHT],
        outline=BORDER_COLOR,
    )
    draw.text((x, HEADER_HEIGHT + MARGIN + TARGET_HEIGHT + 4), left_label, fill=HEADER_TEXT_COLOR)

    x += left_s.width + GUTTER
    sheet.paste(right_s, (x, HEADER_HEIGHT + MARGIN))
    draw.rectangle(
        [x - 1, HEADER_HEIGHT + MARGIN - 1, x + right_s.width, HEADER_HEIGHT + MARGIN + TARGET_HEIGHT],
        outline=BORDER_COLOR,
    )
    draw.text((x, HEADER_HEIGHT + MARGIN + TARGET_HEIGHT + 4), right_label, fill=HEADER_TEXT_COLOR)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> int:
    """Render all declared pairs and write the comparison sheets.

    Returns:
        int: 0 when every pair succeeds, 1 otherwise.
    """
    font = load_font()
    failures: list[str] = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Structural sanity gate: verify every declared input exists up front.
    missing = []
    for stem, png_name, _ in PAIRS:
        if not (TIKZ_DIR / f"{stem}.pdf").exists():
            missing.append(f"{stem}.pdf")
        if not (FIG_DIR / png_name).exists():
            missing.append(png_name)
    if missing:
        print(f"FAIL: missing inputs: {', '.join(missing)}")
        return 1

    for stem, png_name, header in PAIRS:
        out_path = OUT_DIR / f"review_{stem}.png"
        try:
            render_png = OUT_DIR / f"_{stem}_render.png"
            render_pdf_to_png(TIKZ_DIR / f"{stem}.pdf", render_png, RENDER_DPI)
            tikz_img = Image.open(render_png).convert("RGB")
            canonical = Image.open(FIG_DIR / png_name).convert("RGB")
            compose_sheet(
                tikz_img,
                canonical,
                "TikZ preview (figures_tikz/)",
                f"canonical ({png_name})",
                header,
                font,
                out_path,
            )
            render_png.unlink()
            print(f"OK  {out_path.relative_to(PACKAGE_ROOT)}  ({out_path.stat().st_size} bytes)")
        except Exception as exc:  # noqa: BLE001 - report per-pair and continue
            failures.append(stem)
            print(f"FAIL {stem}: {exc}")

    print(f"{len(PAIRS) - len(failures)}/{len(PAIRS)} sheets written to {OUT_DIR.relative_to(PACKAGE_ROOT)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
