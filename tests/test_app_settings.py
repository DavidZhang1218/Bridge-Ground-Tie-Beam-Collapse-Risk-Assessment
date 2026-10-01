"""Check the interface's Ri and advanced scoring input handling."""

import sys
import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app
from app import DEFAULT_DEPTH_RI, DEFAULT_RI, DEFAULT_WEIGHTS, LAMBDA_CHOICES, _scoring_values, build_interface


class AppSettingsTests(unittest.TestCase):
    def test_gemini_progress_timer_updates_elapsed_time_and_stops(self):
        updates = []

        def work():
            time.sleep(0.06)
            return {"I": 150.0}

        result = app._run_with_gemini_timer(
            work,
            lambda _fraction, desc: updates.append(desc),
            time.perf_counter(),
            interval_seconds=0.01,
        )
        self.assertEqual(result, {"I": 150.0})
        self.assertTrue(any("Gemini API processing" in message and "elapsed" in message for message in updates))
        count = len(updates)
        time.sleep(0.02)
        self.assertEqual(len(updates), count)

    def test_validation_error_is_shown_in_english_status(self):
        with TemporaryDirectory() as directory, patch.object(app, "SAMPLE_DIR", Path(directory)):
            (Path(directory) / "example.pdf").touch()
            result = app.run_assessment(
                "example.pdf", None, str(app.EXAMPLE), "", "gemini-3-flash-preview", "cpu",
                *DEFAULT_RI, *([None] * 8), 1.0, *DEFAULT_DEPTH_RI,
                progress=lambda *_args, **_kwargs: None,
            )
        self.assertTrue(result[0].startswith("Error: Enter a Gemini API key"))

    def test_sample_pdfs_are_discovered_and_can_be_refreshed(self):
        with TemporaryDirectory() as directory, patch.object(app, "SAMPLE_DIR", Path(directory)):
            root = Path(directory)
            for name in ("z.pdf", "a.PDF", "ICLprompt.png"):
                (root / name).touch()
            (root / "nested").mkdir()
            (root / "nested" / "hidden.pdf").touch()
            self.assertEqual(list(app.sample_pdf_paths()), ["a.PDF", "z.pdf"])
            self.assertEqual(app.refresh_sample_choices("z.pdf")["value"], "z.pdf")
            self.assertEqual(app.refresh_sample_choices("missing.pdf")["value"], "a.PDF")
            (root / "b.pdf").touch()
            self.assertEqual(app.refresh_sample_choices("a.PDF")["choices"], ["a.PDF", "b.pdf", "z.pdf"])

    def test_default_ri_run_does_not_require_gamma(self):
        scores, scoring, notice = _scoring_values((*DEFAULT_RI, *([None] * 13)))
        self.assertEqual(scores, list(map(float, DEFAULT_RI)))
        self.assertIsNone(scoring)
        self.assertIn("P was not calculated", notice)

    def test_complete_defaults_calculate_p(self):
        scores, scoring, notice = _scoring_values(
            (*DEFAULT_RI, *DEFAULT_WEIGHTS, 1.0, *DEFAULT_DEPTH_RI)
        )
        self.assertEqual(scores, list(map(float, DEFAULT_RI)))
        self.assertEqual(scoring["weights"], list(DEFAULT_WEIGHTS))
        self.assertEqual(scoring["depth_scores"], list(DEFAULT_DEPTH_RI))
        self.assertEqual(notice, "")

    def test_invalid_gamma_is_rejected_before_pipeline(self):
        with self.assertRaisesRegex(ValueError, "weight must be between 0 and 1"):
            _scoring_values((*DEFAULT_RI, 55, *([0.125] * 7), 1.0, 20, 40, 60, 80))

    def test_visible_ri_fields_use_source_defaults(self):
        config = build_interface().get_config_file()
        numbers = [component["props"] for component in config["components"] if component["type"] == "number"]
        self.assertEqual([number["value"] for number in numbers[:7]], list(DEFAULT_RI))
        self.assertTrue(all(number["label"] for number in numbers[:7]))
        self.assertEqual([number["value"] for number in numbers[-4:]], list(DEFAULT_DEPTH_RI))
        self.assertEqual([number["value"] for number in numbers[7:15]], list(DEFAULT_WEIGHTS))
        factor = next(component["props"] for component in config["components"]
                      if component["type"] == "dropdown" and component["props"].get("label") == "Safety management factor λ")
        self.assertEqual(factor["value"], 1.0)
        self.assertEqual([choice[1] for choice in factor["choices"]], list(LAMBDA_CHOICES))

    def test_complete_defaults_keep_depth_and_calculate_p(self):
        with TemporaryDirectory() as directory:
            output_root = Path(directory)
            sample_name = "new_sample.PDF"
            (output_root / sample_name).touch()
            reference = app.EXAMPLE
            progress_messages = []
            record = {"pier_id": 2, "section": "I", "h1_m": 109.5, "h2_m": 110.0}
            excluded = {"pier_id": 3, "section": "I", "h1_m": 112.0, "h2_m": 110.0}
            with (
                patch.object(app, "OUTPUT_DIR", output_root),
                patch.object(app, "SAMPLE_DIR", output_root),
                patch.object(app, "render_drawing_pdf", return_value=(reference, [reference, reference])),
                patch.object(app, "run_cv_pipeline", return_value={"records": [record, excluded], "artifacts": {}}),
                patch.object(app, "extract_section_heights", return_value={"I": 100}),
                patch.object(app, "generate_report", side_effect=lambda _summary, path, **_kwargs: path),
            ):
                result = app.run_assessment(
                    sample_name, None, str(reference), "test-key", "gemini-3-flash-preview", "cpu",
                    60, *DEFAULT_RI[1:], *DEFAULT_WEIGHTS, 1.0, *DEFAULT_DEPTH_RI,
                    progress=lambda *_args, **kwargs: progress_messages.append(kwargs.get("desc", "")),
                )
            summary = json.loads(Path(result[5]).read_text(encoding="utf-8"))
            self.assertEqual(summary["indicator_scores"]["Groundwater condition"], 60)
            self.assertEqual(summary["section_images"], [reference.name, reference.name])
            self.assertEqual(set(summary["stage_timings_s"]), {"pdf_render", "cv_ocr", "gemini"})
            self.assertTrue(any("CV/OCR complete" in message for message in progress_messages))
            self.assertTrue(any("Gemini finished" in message for message in progress_messages))
            self.assertIn("CV/OCR", result[0])
            self.assertEqual(result[3], [str(reference), str(reference)])
            self.assertEqual(len(summary["records"]), 1)
            self.assertEqual(summary["excluded_candidate_count"], 1)
            self.assertAlmostEqual(summary["records"][0]["excavation_depth_m"], 1.5)
            self.assertAlmostEqual(summary["records"][0]["p_value"], 39.6875)
            self.assertEqual(summary["records"][0]["likelihood_level"], 3)
            self.assertIn("Equal weights", summary["weight_note"])


if __name__ == "__main__":
    unittest.main()
