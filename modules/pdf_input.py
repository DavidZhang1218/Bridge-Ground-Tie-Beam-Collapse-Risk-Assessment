"""Render a drawing PDF for the profile and section workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

import pypdfium2 as pdfium


MIN_DPI = 300


def render_drawing_pdf(pdf_path: Path, output_dir: Path, dpi: int = MIN_DPI) -> tuple[Path, list[Path]]:
    """Save page 1 as the profile and every later page as a section view."""
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Drawing PDF not found: {pdf_path}")
    if dpi < MIN_DPI:
        raise ValueError(f"Rendering resolution must be at least {MIN_DPI} dpi.")

    document = pdfium.PdfDocument(str(pdf_path))
    try:
        if len(document) < 2:
            raise ValueError(f"A drawing PDF must contain at least two pages; found {len(document)}.")
        output_dir.mkdir(parents=True, exist_ok=True)
        profile_path = output_dir / f"{pdf_path.stem}.jpg"
        section_paths = []
        for page_number in range(2, len(document) + 1):
            suffix = "" if page_number == 2 else f"_{page_number}"
            section_paths.append(output_dir / f"{pdf_path.stem}_sectionview{suffix}.jpg")
        for page_index, image_path in enumerate([profile_path, *section_paths]):
            page = document[page_index]
            try:
                rendered = page.render(scale=dpi / 72).to_pil().convert("RGB")
                rendered.save(image_path, format="JPEG", quality=95, subsampling=0, dpi=(dpi, dpi))
            finally:
                page.close()
        return profile_path, section_paths
    finally:
        document.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a drawing PDF at 300 dpi or higher.")
    parser.add_argument("pdf", type=Path, help="Drawing PDF with a profile page followed by section pages")
    parser.add_argument("output", type=Path, help="Directory for the rendered JPG files")
    parser.add_argument("--dpi", type=int, default=MIN_DPI, help="Rendering resolution (minimum: 300)")
    arguments = parser.parse_args()
    profile, sections = render_drawing_pdf(arguments.pdf, arguments.output, arguments.dpi)
    for image_path in [profile, *sections]:
        print(image_path)
