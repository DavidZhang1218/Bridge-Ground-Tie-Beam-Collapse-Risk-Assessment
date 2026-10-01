"""Check table-divider alignment for ground elevation extraction."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import cv_ground


class GroundElevationTests(unittest.TestCase):
    def test_ground_value_is_read_left_of_nearest_table_divider(self):
        image = Image.new("RGB", (400, 300), "white")
        ImageDraw.Draw(image).rectangle((90, 120, 95, 180), fill="red")
        matched = {"results": [{
            "source_pier_id": 2,
            "candidate_id": 1,
            "candidate_bbox": [140, 30, 160, 50],
            "candidate_center": [150, 40],
            "matched_level_value": 110.0,
            "match_status": "matched",
        }]}
        ground_row = {"bbox": [0, 100, 400, 200]}

        def recognize(crop, *_args, **_kwargs):
            has_value = any(pixel == (255, 0, 0) for pixel in crop.getdata())
            return {"best": {
                "numeric_value": 105.0 if has_value else None,
                "numeric_text": "105.0" if has_value else "",
                "raw_score": 0.99 if has_value else 0.0,
            }}

        with patch.object(cv_ground, "recognize_text_variants", side_effect=recognize):
            beams, _groups = cv_ground.build_ground_tie_beams(
                image, matched, ground_row,
                {"vertical_line_segments": [
                    {"x": 148, "y1": 0, "y2": 90},
                    {"x": 120, "y1": 100, "y2": 200},
                ]},
                object(), 165,
            )
        self.assertEqual(beams[0]["ground_level"]["value"], 105.0)
        self.assertEqual(beams[0]["ground_level"]["cell_bbox"], [82, 100, 113, 200])
        self.assertEqual(beams[0]["status"], "ready")


if __name__ == "__main__":
    unittest.main()
