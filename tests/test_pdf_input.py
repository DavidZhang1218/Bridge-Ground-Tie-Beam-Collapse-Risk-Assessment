"""Check profile and section-page rendering for multi-page PDFs."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image
from reportlab.pdfgen import canvas

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.pdf_input import render_drawing_pdf


class PdfInputTests(unittest.TestCase):
    def test_all_pages_after_profile_are_rendered_as_sections(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "drawing.pdf"
            document = canvas.Canvas(str(pdf_path), pagesize=(72, 72))
            for label in ("Profile", "Section A", "Section B"):
                document.drawString(5, 35, label)
                document.showPage()
            document.save()

            profile, sections = render_drawing_pdf(pdf_path, root / "rendered")
            self.assertEqual(profile.name, "drawing.jpg")
            self.assertEqual([path.name for path in sections], [
                "drawing_sectionview.jpg", "drawing_sectionview_3.jpg",
            ])
            for path in [profile, *sections]:
                with Image.open(path) as image:
                    self.assertEqual(image.size, (300, 300))

    def test_one_page_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "drawing.pdf"
            document = canvas.Canvas(str(pdf_path), pagesize=(72, 72))
            document.drawString(5, 35, "Profile")
            document.save()
            with self.assertRaisesRegex(ValueError, "at least two pages"):
                render_drawing_pdf(pdf_path, root / "rendered")


if __name__ == "__main__":
    unittest.main()
