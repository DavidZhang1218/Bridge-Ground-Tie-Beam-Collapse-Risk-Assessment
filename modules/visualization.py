"""Draw the final beam selection and recognized section symbols."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont


SECTION_COLOR = (150, 65, 245)
BEAM_COLOR = (235, 30, 35)
BEAM_LABEL_BACKGROUND = (255, 222, 222)
DETAIL_BACKGROUND = (255, 255, 180)
TABLE_COLOR = (160, 160, 160)
GROUND_ROW_COLOR = (59, 130, 246)
GROUND_CELL_COLOR = (255, 170, 0)


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _box(value: object) -> tuple[int, int, int, int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (round(float(point)) for point in value)
    except (TypeError, ValueError):
        return None
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont,
           fill: tuple[int, int, int], background: tuple[int, int, int], width: int) -> None:
    text_box = draw.textbbox((0, 0), text, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    x = max(0, min(xy[0], width - text_width - 8))
    y = max(0, xy[1])
    draw.rectangle((x, y, x + text_width + 8, y + text_height + 8), fill=background)
    draw.text((x + 4, y + 4), text, fill=fill, font=font)


def _number(value: object, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}".rstrip("0").rstrip(".")


def draw_assessment_visualization(
    image_path: Path,
    records: Sequence[Mapping],
    ocr_summary_path: Path | str | None,
    output_path: Path,
    ground_summary_path: Path | str | None = None,
) -> Path:
    """Draw confirmed beams, section symbols, and the ground elevation table."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as original:
        image = original.convert("RGB")
    draw = ImageDraw.Draw(image)
    large_font = _font(max(18, image.width // 185))
    small_font = _font(max(16, image.width // 245))

    selected_boxes = {
        box for record in records
        if record.get("excavation_type") in ("full", "partial")
        if (box := _box(record.get("candidate_bbox"))) is not None
    }
    if ground_summary_path and Path(ground_summary_path).is_file():
        ground_summary = json.loads(Path(ground_summary_path).read_text(encoding="utf-8"))
        table_box = _box((ground_summary.get("table_debug") or {}).get("grid", {}).get("table_bbox"))
        if table_box is not None:
            draw.rectangle(table_box, outline=TABLE_COLOR, width=2)
        ground_row = ground_summary.get("detected_ground_row") or {}
        row_box = _box(ground_row.get("bbox"))
        if row_box is not None:
            draw.rectangle(row_box, outline=GROUND_ROW_COLOR, width=3)
            _label(draw, (row_box[0] + 4, row_box[1] - 30), "ground row", small_font,
                   (0, 0, 0), DETAIL_BACKGROUND, image.width)
        drawn_cells = set()
        for candidate in ground_summary.get("candidate_beams", []):
            if _box(candidate.get("candidate_bbox")) not in selected_boxes:
                continue
            ground_level = candidate.get("ground_level") or {}
            cell_box = _box(ground_level.get("cell_bbox"))
            if cell_box is None or cell_box in drawn_cells:
                continue
            drawn_cells.add(cell_box)
            draw.rectangle(cell_box, outline=GROUND_CELL_COLOR, width=3)
            _label(draw, (cell_box[0] + 2, cell_box[1] - 30),
                   f"G={_number(ground_level.get('value'))}", small_font,
                   (0, 0, 0), DETAIL_BACKGROUND, image.width)

    if ocr_summary_path and Path(ocr_summary_path).is_file():
        summary = json.loads(Path(ocr_summary_path).read_text(encoding="utf-8"))
        for symbol in summary.get("section_symbols", []):
            box = _box(symbol.get("bbox_in_full"))
            if box is None:
                continue
            text = symbol.get("symbol_text") or symbol.get("ocr_text") or "?"
            orientation = symbol.get("orientation")
            suffix = "L" if orientation == "left" else "R" if orientation == "right" else "?"
            draw.rectangle(box, outline=SECTION_COLOR, width=5)
            _label(draw, (box[0], box[1] - 38), f"S={text}-{suffix}", small_font,
                   SECTION_COLOR, (250, 245, 255), image.width)

    selected_index = 0
    for record in records:
        if record.get("excavation_type") not in ("full", "partial"):
            continue
        box = _box(record.get("candidate_bbox"))
        if box is None:
            continue
        lane = selected_index % 3
        selected_index += 1
        x1, y1, x2, y2 = box
        draw.rectangle(box, outline=BEAM_COLOR, width=7)
        _label(draw, (x2 + 5, y1), "ground_tie_beam", small_font,
               BEAM_COLOR, BEAM_LABEL_BACKGROUND, image.width)
        detail = (
            f"GB{record.get('pier_id') or '?'} S={record.get('section') or '?'} "
            f"B={_number(record.get('h1_m'))} G={_number(record.get('h2_m'))} "
            f"D={_number(record.get('excavation_depth_m'))}"
        )
        _label(draw, (x1 - 2, y1 - 42 - lane * 45), detail, large_font,
               (35, 35, 0), DETAIL_BACKGROUND, image.width)

    image.save(output_path)
    return output_path
