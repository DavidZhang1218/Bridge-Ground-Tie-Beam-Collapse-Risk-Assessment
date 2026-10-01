"""Boundary checks for section height parsing and deterministic assessment."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.risk import (
    assess_records,
    depth_band_index,
    likelihood_level,
    score_excavation,
    validate_indicator_scores,
    validate_scoring,
)
from modules.vlm import SectionHeightError, extract_section_heights, normalize_section_label, parse_height_response


SCORING = {
    "weights": [0.125] * 8,
    "lambda_factor": 1.0,
    "depth_scores": [20, 40, 60, 80],
}


class HeightParsingTests(unittest.TestCase):
    def test_normalizes_roman_labels_and_nulls(self):
        self.assertEqual(normalize_section_label("Ⅱ–Ⅱ"), "II")
        self.assertIsNone(normalize_section_label("I-II"))
        response = '[{"roman_numeral":"I-I","grounding_beam_thickness":"150 cm"},' \
                   '{"roman_numeral":"II-II","grounding_beam_thickness":null}]'
        self.assertEqual(parse_height_response(response), {"I": 150.0, "II": None})

    def test_rejects_conflicting_height_for_same_section(self):
        response = '[{"roman_numeral":"I-I","grounding_beam_thickness":100},' \
                   '{"roman_numeral":"I","grounding_beam_thickness":150}]'
        with self.assertRaises(SectionHeightError):
            parse_height_response(response)

    def test_single_request_includes_reference_without_app_timeout(self):
        requests = []
        statuses = []

        class FakeClient:
            def __init__(self, **kwargs):
                self.options = kwargs
                self.models = self
                self.closed = False
                requests.append(self)

            def generate_content(self, **kwargs):
                self.request = kwargs
                return SimpleNamespace(text='[{"roman_numeral":"I-I","grounding_beam_thickness":150}]')

            def close(self):
                self.closed = True

        with TemporaryDirectory() as directory:
            image = Path(directory) / "image.jpg"
            image.write_bytes(b"image")
            with (
                patch("google.genai.Client", FakeClient),
                patch("modules.vlm._image_part", side_effect=["reference", "section", "section2"]),
            ):
                heights = extract_section_heights([image, image], image, "test-key", status=statuses.append)

        self.assertEqual(heights, {"I": 150.0})
        self.assertEqual(len(requests), 1)
        self.assertNotIn("http_options", requests[0].options)
        self.assertEqual(requests[0].request["contents"][1], "reference")
        self.assertEqual(requests[0].request["contents"][4], "section")
        self.assertEqual(requests[0].request["contents"][6], "section2")
        self.assertTrue(requests[0].closed)
        self.assertEqual(len(statuses), 3)

    def test_transient_connection_error_retries_once(self):
        from httpx import ConnectError

        calls = []
        statuses = []

        class IntermittentClient:
            def __init__(self, **kwargs):
                self.models = self
                calls.append(kwargs)

            def generate_content(self, **_kwargs):
                if len(calls) == 1:
                    raise ConnectError("connection interrupted")
                return SimpleNamespace(text='[{"roman_numeral":"II-II","grounding_beam_thickness":120}]')

            def close(self):
                pass

        with TemporaryDirectory() as directory:
            image = Path(directory) / "image.jpg"
            image.write_bytes(b"image")
            with (
                patch("google.genai.Client", IntermittentClient),
                patch("modules.vlm._image_part", return_value="image"),
                patch("modules.vlm.time.sleep"),
            ):
                heights = extract_section_heights([image], image, "test-key", status=statuses.append)

        self.assertEqual(heights, {"II": 120.0})
        self.assertEqual(len(calls), 2)
        self.assertTrue(all("http_options" not in call for call in calls))
        self.assertTrue(any("retrying once" in message for message in statuses))


class RiskTests(unittest.TestCase):
    def test_excavation_classification_and_missing_scores(self):
        records = [
            {"pier_id": 1, "section": "I-I", "h1_m": 109.9, "h2_m": 110.2},
            {"pier_id": 2, "section": "I", "h1_m": 110.5, "h2_m": 110.2},
            {"pier_id": 3, "section": "I", "h1_m": 111.2, "h2_m": 110.2},
            {"pier_id": 4, "section": "II-II", "h1_m": 110.0, "h2_m": 110.0},
        ]
        result = assess_records(records, {"I": 100, "II": None}, SCORING)
        self.assertEqual([row["excavation_type"] for row in result],
                         ["full", "partial", "excluded", "height_unavailable"])
        self.assertAlmostEqual(result[0]["excavation_depth_m"], 1.3)
        self.assertAlmostEqual(result[1]["excavation_depth_m"], 0.7)
        self.assertIsNone(result[2]["excavation_depth_m"])
        self.assertIsNone(result[2]["p_value"])
        self.assertIsNone(result[3]["p_value"])
        self.assertIsNone(assess_records(records[:1], {"I": 100}, None)[0]["p_value"])

    def test_depth_and_possibility_boundaries(self):
        self.assertEqual([depth_band_index(v) for v in (2.999, 3, 4.999, 5, 10, 10.001)],
                         [0, 1, 1, 2, 2, 3])
        self.assertEqual([likelihood_level(v) for v in (0, 15, 15.01, 30, 30.01, 45, 45.01, 60, 60.01)],
                         [1, 1, 2, 2, 3, 3, 4, 4, 5])
        p_value, level, depth_score, band = score_excavation(2, SCORING)
        self.assertAlmostEqual(p_value, sum((55, 45, 80, 40, 25, 35, 20, 20)) / 8)
        self.assertEqual((level, depth_score, band), (3, 20, "<3 m"))

    def test_rejects_invalid_scores_but_allows_incomplete_configuration(self):
        self.assertIsNone(validate_scoring(None))
        self.assertIsNone(validate_scoring({"weights": None}))
        with self.assertRaises(ValueError):
            validate_scoring({**SCORING, "weights": [0.1] * 8})
        with self.assertRaises(ValueError):
            validate_scoring({**SCORING, "depth_scores": [26, 40, 60, 80]})
        with self.assertRaises(ValueError):
            validate_scoring({**SCORING, "lambda_factor": 1.2})

    def test_user_selected_ri_changes_p_and_rejects_out_of_range_ri(self):
        chosen = [40, 45, 80, 40, 25, 35, 20]
        settings = {**SCORING, "indicator_scores": chosen}
        p_value, _, _, _ = score_excavation(2, settings)
        self.assertAlmostEqual(p_value, sum((*chosen, 20)) / 8)
        with self.assertRaisesRegex(ValueError, "Ri score"):
            validate_indicator_scores([101, *chosen[1:]])


if __name__ == "__main__":
    unittest.main()
