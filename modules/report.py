"""Create a dynamic assessment report with result tables and drawing figures."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from html import escape
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import HRFlowable, Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


PAGE_WIDTH, PAGE_HEIGHT = A4
CONTENT_WIDTH = PAGE_WIDTH - 88


def _font(name: str, paths: tuple[str, ...], fallback: str) -> str:
    for path in paths:
        if Path(path).is_file():
            if name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(name, path))
            return name
    return fallback


SERIF = _font("ReportSerif", (
    "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"), "Times-Roman")
SERIF_BOLD = _font("ReportSerifBold", (
    "/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"), "Times-Bold")
SANS = _font("ReportSans", (
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), "Helvetica")
SANS_BOLD = _font("ReportSansBold", (
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"), "Helvetica-Bold")

TITLE = ParagraphStyle("Title", fontName=SANS_BOLD, fontSize=17, leading=20, alignment=TA_CENTER, spaceAfter=17)
BODY = ParagraphStyle("Body", fontName=SERIF, fontSize=9.5, leading=14.8, alignment=TA_JUSTIFY, spaceAfter=10)
HEADING = ParagraphStyle("Heading", fontName=SANS_BOLD, fontSize=12, leading=16, spaceAfter=10)
CAPTION = ParagraphStyle("Caption", fontName=SERIF, fontSize=10.5, leading=14, alignment=TA_CENTER, spaceBefore=10, spaceAfter=10)
CELL = ParagraphStyle("Cell", fontName=SERIF, fontSize=8.7, leading=10.5, alignment=TA_CENTER)
CELL_HEAD = ParagraphStyle("CellHead", parent=CELL, fontName=SERIF_BOLD)
NOTE = ParagraphStyle("Note", fontName=SERIF, fontSize=8.4, leading=11.5, spaceBefore=8)


class NumberedCanvas(canvas.Canvas):
    """Draw a border and the total page count after layout is complete."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._page_states = []

    def showPage(self):
        self._page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._page_states)
        for state in self._page_states:
            self.__dict__.update(state)
            self.setStrokeColor(colors.black)
            self.setLineWidth(0.8)
            self.rect(5, 5, PAGE_WIDTH - 10, PAGE_HEIGHT - 10)
            self.setFillColor(colors.HexColor("#777777"))
            self.setFont(SANS, 8)
            self.drawCentredString(PAGE_WIDTH / 2, 21, f"Page {self._pageNumber} of {total}")
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)


def _value(value: object, digits: int = 3) -> str:
    if value is None or value == "":
        return "N/A"
    try:
        return f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(value)


def _table(rows: list[list[object]], widths: list[float], risk_column: int | None = None) -> Table:
    cells = [
        [Paragraph(escape(str(value)), CELL_HEAD if index == 0 else CELL) for value in row]
        for index, row in enumerate(rows)
    ]
    table = Table(cells, colWidths=widths, repeatRows=1, hAlign="CENTER")
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#4c4c4c")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9d9d9")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, 0), 7),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 7),
        ("TOPPADDING", (0, 1), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 3),
    ]
    if risk_column is not None:
        for index, row in enumerate(rows[1:], start=1):
            level = str(row[risk_column]).strip()
            color = "#f4dad3" if level in ("4", "5") else "#fff2cc" if level in ("1", "2", "3") else None
            if color:
                commands.append(("BACKGROUND", (risk_column, index), (risk_column, index), colors.HexColor(color)))
    table.setStyle(TableStyle(commands))
    return table


def _figure(path: Path | str | None, max_height: float) -> Image | None:
    if path is None or not Path(path).is_file():
        return None
    with PILImage.open(path) as original:
        width, height = original.size
    scale = min(CONTENT_WIDTH / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale, hAlign="CENTER")


def generate_report(
    summary: dict,
    output_path: Path,
    profile_image: Path | None = None,
    section_images: Sequence[Path] | None = None,
    result_image: Path | None = None,
) -> Path:
    """Render assessment tables and figures from one run, without demonstration rows."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = [row for row in summary.get("records", []) if row.get("excavation_type") != "excluded"]
    introduction = (
        "According to the Guideline for Safety Risk Assessment of Highway and Waterway Engineering "
        "Construction - Part 2: Bridge Engineering (JT/T 1375.2-2025), the possibility of bridge "
        "foundation pit construction risk events is evaluated using the score P. P combines selected "
        "indicator scores, their weights, and the safety management factor. The calculated P value is "
        "classified into five likelihood levels: Level 5 when P &gt; 60, Level 4 when 45 &lt; P ≤ 60, "
        "Level 3 when 30 &lt; P ≤ 45, Level 2 when 15 &lt; P ≤ 30, and Level 1 when 0 ≤ P ≤ 15. "
        "Table 1 presents the final assessment results. Table 2 summarizes the matched ground tie beam "
        "information for each pier, including the associated section, beam height h, beam elevation H1, "
        "ground elevation H2, and excavation depth d."
    )
    story: list[object] = [
        Spacer(1, 4),
        Paragraph("Bridge Ground Tie Beam Construction Collapse Risk<br/>Level Assessment Report", TITLE),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#24576a")),
        Spacer(1, 17),
        Paragraph(introduction, BODY),
        Paragraph("Table 1: Final assessment results.", CAPTION),
    ]
    result_rows: list[list[object]] = [["Ground tie beam and pier number", "P value", "Risk level"]]
    detail_rows: list[list[object]] = [["Ground tie beam and pier number", "Section", "h (cm)", "H1 (m)", "H2 (m)", "d (m)"]]
    for row in records:
        level = row.get("likelihood_level")
        result_rows.append([
            row.get("pier_id") or "N/A",
            _value(row.get("p_value"), 2) if row.get("p_value") is not None else "Not calculated",
            int(level) if level is not None else "Not calculated",
        ])
        detail_rows.append([
            row.get("pier_id") or "N/A", row.get("section") or "N/A",
            _value(row.get("h_cm"), 1), _value(row.get("h1_m")),
            _value(row.get("h2_m")), _value(row.get("excavation_depth_m")),
        ])
    if not records:
        result_rows.append(["No confirmed ground tie beam", "N/A", "N/A"])
        detail_rows.append(["No confirmed ground tie beam", "N/A", "N/A", "N/A", "N/A", "N/A"])
    story.extend([
        _table(result_rows, [CONTENT_WIDTH * ratio for ratio in (0.43, 0.27, 0.30)], risk_column=2),
        Spacer(1, 15),
        Paragraph("Table 2: Ground tie beam information.", CAPTION),
        _table(detail_rows, [CONTENT_WIDTH * ratio for ratio in (0.30, 0.12, 0.12, 0.15, 0.16, 0.15)]),
    ])
    if summary.get("scoring_note"):
        story.append(Paragraph(escape(str(summary["scoring_note"])), NOTE))
    scores = summary.get("indicator_scores") or {}
    if scores:
        settings = "; ".join(f"{escape(str(name))}: {_value(value, 2)}" for name, value in scores.items())
        story.append(Paragraph("Selected indicator Ri values: " + settings + ".", NOTE))
    weights = summary.get("weights")
    depth_scores = summary.get("depth_scores")
    if depth_scores:
        story.append(Paragraph(
            "Depth Ri by ascending band: " + ", ".join(_value(value, 2) for value in depth_scores)
            + ". Safety management factor λ: " + _value(summary.get("lambda_value"), 2) + ".",
            NOTE,
        ))
    if weights:
        story.append(Paragraph(
            "Weights γ in indicator order: " + ", ".join(_value(value, 4) for value in weights) + ".",
            NOTE,
        ))
    if summary.get("weight_note"):
        story.append(Paragraph(escape(str(summary["weight_note"])), NOTE))
    story.extend([
        Paragraph(
            f"Drawing: {escape(Path(summary.get('source_pdf') or 'Unknown').name)}. "
            f"Generated: {datetime.now():%Y-%m-%d %H:%M}.", NOTE
        ),
        PageBreak(),
        Paragraph("Appendix A. Detailed visualized output and section-view drawings", HEADING),
        Paragraph(
            "Figure 1 is the final visualized output. Recognized section symbols are marked in purple, "
            "and confirmed ground tie beams are highlighted in red with their associated section, H1, "
            "H2, and excavation depth. Candidates excluded by the excavation rule are not shown.", BODY
        ),
    ])
    visualization = _figure(result_image or profile_image, max_height=535)
    if visualization is not None:
        story.extend([Spacer(1, 10), visualization, Paragraph("Figure 1: Visualized output of bridge profile arrangement drawing.", CAPTION)])
    story.extend([PageBreak(), Paragraph("Appendix A. Corresponding section-view drawings", HEADING)])
    for index, section_image in enumerate(section_images or []):
        if index:
            story.extend([PageBreak(), Paragraph("Appendix A. Corresponding section-view drawings", HEADING)])
        section_figure = _figure(section_image, max_height=640)
        if section_figure is not None:
            story.extend([
                Spacer(1, 22), section_figure,
                Paragraph(f"Figure {index + 2}: Section-view drawing, PDF page {index + 2}.", CAPTION),
            ])
    document = SimpleDocTemplate(
        str(output_path), pagesize=A4, leftMargin=44, rightMargin=44,
        topMargin=38, bottomMargin=46,
        title="Bridge Ground Tie Beam Construction Collapse Risk Level Assessment Report",
        author="Bridge Ground Tie Beam Assessment",
    )
    document.build(story, canvasmaker=NumberedCanvas)
    return output_path
