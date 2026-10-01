"""Regression checks for drawing geometry without model inference."""

import unittest

from PIL import Image, ImageDraw

from modules.cv_ground import detect_table_grid, row_left_label_crop
from modules.cv_match import build_match_score, correct_level_decimal_omissions, greedy_match


class DrawingGeometryTests(unittest.TestCase):
    def test_table_header_crop_follows_shifted_table(self) -> None:
        image = Image.new("RGB", (1000, 700), "white")
        draw = ImageDraw.Draw(image)
        for y in (500, 530, 560, 590, 620):
            draw.line((350, y, 850, y), fill="black", width=3)
        draw.line((350, 500, 350, 620), fill="black", width=3)
        draw.line((850, 500, 850, 620), fill="black", width=3)
        draw.rectangle((370, 540, 390, 550), fill="red")

        grid = detect_table_grid(image, 165)
        self.assertAlmostEqual(grid["table_left_x"], 350, delta=3)
        row = next(item for item in grid["rows"] if item["bbox"][1] <= 540 < item["bbox"][3])
        self.assertGreater(row["label_start_x"], image.width * 0.25)
        label_crop = row_left_label_crop(image, row)
        self.assertIn((255, 0, 0), label_crop.getdata())

    def test_level_bottom_to_candidate_top_is_one_to_one_within_pier(self) -> None:
        self.assertEqual(build_match_score([0, 10, 10, 20], [0, 0, 10, 10]), 0)
        self.assertEqual(build_match_score([20, 10, 30, 20], [0, 0, 10, 10]), 20)
        candidates = [
            {"candidate_id": 1, "bbox_in_full": [0, 10, 10, 20], "source_pier_id": 1, "source_crop": "p1"},
            {"candidate_id": 2, "bbox_in_full": [0, 10, 10, 20], "source_pier_id": 2, "source_crop": "p2"},
        ]
        levels = [
            {"level_id": 1, "bbox_in_full": [0, 0, 10, 10], "source_pier_id": 1, "source_crop": "p1"},
            {"level_id": 2, "bbox_in_full": [0, 0, 10, 10], "source_pier_id": 2, "source_crop": "p2"},
        ]
        matches, pairs = greedy_match(candidates, levels)
        self.assertEqual(matches, {1: 1, 2: 2})
        self.assertEqual(len(pairs), 2)

    def test_missing_decimal_uses_ocr_alternative_and_vertical_order(self) -> None:
        def level(y, value, text, candidates=None):
            return {
                "bbox_in_full": [0, y - 15, 60, y + 15],
                "center_in_full": [30, y],
                "recognized_value": value,
                "ocr_text": text,
                "ocr_score": 0.9,
                "ocr_debug": {"candidates": candidates or []},
            }

        levels = [
            level(80, 120.0, "120.0"),
            level(100, 118.0, "118.0"),
            level(140, 1123.0, "1123", [{
                "numeric_text": "112.3", "numeric_value": 112.3,
                "raw_text": "112.3", "raw_score": 0.75, "variant": "dark_binary_up",
            }]),
            level(170, 109.0, "109.0"),
            level(190, 107.0, "107.0"),
        ]
        correct_level_decimal_omissions(levels)
        self.assertEqual(levels[2]["recognized_value"], 112.3)
        self.assertEqual(levels[2]["context_correction"]["source"], "ocr_variant")
        self.assertEqual([item["recognized_value"] for item in levels if item is not levels[2]],
                         [120.0, 118.0, 109.0, 107.0])

        levels[2] = level(140, 1123.0, "1123")
        correct_level_decimal_omissions(levels)
        self.assertEqual(levels[2]["recognized_value"], 112.3)
        self.assertEqual(levels[2]["context_correction"]["source"], "decimal_insertion")

    def test_decimal_is_not_inferred_without_both_vertical_neighbors(self) -> None:
        levels = [
            {"bbox_in_full": [0, 0, 60, 30], "center_in_full": [30, 15],
             "recognized_value": 1123.0, "ocr_text": "1123", "ocr_score": 0.9,
             "ocr_debug": {"candidates": []}},
            {"bbox_in_full": [0, 40, 60, 70], "center_in_full": [30, 55],
             "recognized_value": 109.0, "ocr_text": "109.0", "ocr_score": 0.9,
             "ocr_debug": {"candidates": []}},
        ]
        correct_level_decimal_omissions(levels)
        self.assertEqual(levels[0]["recognized_value"], 1123.0)


if __name__ == "__main__":
    unittest.main()
