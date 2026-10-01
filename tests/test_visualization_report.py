"""Verify the final figure and report use the assessed beam selection."""

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pypdfium2 as pdfium
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.report import generate_report
from modules.visualization import (
    BEAM_COLOR, GROUND_CELL_COLOR, GROUND_ROW_COLOR, SECTION_COLOR, TABLE_COLOR,
    draw_assessment_visualization,
)


class FinalOutputTests(unittest.TestCase):
    def test_excluded_candidate_is_absent_from_figure_and_report(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            drawing = root / "drawing.png"
            Image.new("RGB", (800, 500), "white").save(drawing)
            ocr = root / "ocr.json"
            ocr.write_text(json.dumps({"section_symbols": [{
                "bbox_in_full": [20, 20, 70, 70],
                "symbol_text": "II", "orientation": "right",
            }]}), encoding="utf-8")
            selected = {
                "pier_id": 3, "section": "II", "candidate_bbox": [140, 200, 180, 240],
                "h_cm": 150, "h1_m": 109.9, "h2_m": 110.193,
                "excavation_depth_m": 1.793, "excavation_type": "full",
                "p_value": None, "likelihood_level": None,
            }
            excluded = {
                **selected, "pier_id": 99, "candidate_bbox": [600, 200, 640, 240],
                "excavation_type": "excluded", "excavation_depth_m": None,
            }
            ground = root / "ground.json"
            ground.write_text(json.dumps({
                "table_debug": {"grid": {"table_bbox": [100, 300, 700, 480]}},
                "detected_ground_row": {"bbox": [100, 350, 700, 400]},
                "candidate_beams": [
                    {"candidate_bbox": selected["candidate_bbox"],
                     "ground_level": {"cell_bbox": [150, 350, 180, 400], "value": 110.193}},
                    {"candidate_bbox": excluded["candidate_bbox"],
                     "ground_level": {"cell_bbox": [610, 350, 640, 400], "value": 112.0}},
                ],
            }), encoding="utf-8")
            figure = draw_assessment_visualization(drawing, [selected, excluded], ocr, root / "final.png", ground)
            with Image.open(figure) as image:
                self.assertEqual(image.getpixel((20, 40)), SECTION_COLOR)
                self.assertEqual(image.getpixel((140, 220)), BEAM_COLOR)
                self.assertEqual(image.getpixel((600, 220)), (255, 255, 255))
                self.assertEqual(image.getpixel((100, 320)), TABLE_COLOR)
                self.assertEqual(image.getpixel((100, 375)), GROUND_ROW_COLOR)
                self.assertEqual(image.getpixel((150, 375)), GROUND_CELL_COLOR)
                self.assertEqual(image.getpixel((610, 375)), (255, 255, 255))
            report = generate_report(
                {"source_pdf": "drawing.pdf", "records": [selected, excluded],
                 "indicator_scores": {"Groundwater condition": 55}},
                root / "assessment_report.pdf", drawing, [drawing], figure,
            )
            document = pdfium.PdfDocument(str(report))
            try:
                self.assertEqual(len(document), 3)
                page = document[0]
                text = page.get_textpage().get_text_range()
                self.assertIn("Not calculated", text)
                self.assertNotIn("99", text)
                page.close()
            finally:
                document.close()

    def test_report_includes_every_section_page(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            drawing = root / "drawing.png"
            Image.new("RGB", (800, 500), "white").save(drawing)
            report = generate_report(
                {"source_pdf": "drawing.pdf", "records": []},
                root / "assessment_report.pdf",
                profile_image=drawing,
                section_images=[drawing, drawing],
                result_image=drawing,
            )
            document = pdfium.PdfDocument(str(report))
            try:
                self.assertEqual(len(document), 4)
                for page_index, figure_number in ((2, 2), (3, 3)):
                    page = document[page_index]
                    try:
                        text = page.get_textpage().get_text_range()
                        self.assertIn(f"Figure {figure_number}", text)
                    finally:
                        page.close()
            finally:
                document.close()


if __name__ == "__main__":
    unittest.main()
