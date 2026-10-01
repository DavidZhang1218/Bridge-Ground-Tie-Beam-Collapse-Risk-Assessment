"""Run bridge ground tie beam extraction and assessment from a drawing PDF."""

from __future__ import annotations

import json
import os
import time
from contextvars import copy_context
from datetime import datetime
from pathlib import Path
from threading import Event, Thread
from typing import Callable
from uuid import uuid4

import gradio as gr

from modules.cv_pipeline import run_cv_pipeline
from modules.pdf_input import render_drawing_pdf
from modules.report import generate_report
from modules.risk import assess_records, validate_indicator_scores, validate_scoring
from modules.visualization import draw_assessment_visualization
from modules.vlm import extract_section_heights


ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "sample_image"
OUTPUT_DIR = ROOT / "outputs"
STAGE1 = ROOT / "weights" / "stage1.pt"
STAGE2 = ROOT / "weights" / "stage2.pt"
EXAMPLE = SAMPLE_DIR / "ICLprompt.png"
INDICATORS = (
    "Groundwater condition",
    "Foundation pit surroundings",
    "Construction-period rainfall condition",
    "Geological ground condition",
    "Excavation method",
    "Monitoring condition",
    "Technical applicability",
    "Excavation depth",
)
DEFAULT_RI = (55, 45, 80, 40, 25, 35, 20)
DEFAULT_WEIGHTS = (0.125,) * 8
DEFAULT_DEPTH_RI = (12.5, 37.5, 62.5, 87.5)
LAMBDA_CHOICES = (0.9, 0.95, 1.0, 1.05, 1.1)
DEPTH_BANDS = ("Below 3 m", "3 to below 5 m", "5 to 10 m", "Above 10 m")
MODEL_CHOICES = (
    ("Gemini 3 Flash", "gemini-3-flash-preview"),
    ("Gemini 3 Pro", "gemini-3.1-pro-preview"),
    ("Gemini 3.1 Flash Lite", "gemini-3.1-flash-lite"),
)
CUSTOM_CSS = """
#title {text-align: center; margin-bottom: 1.5rem;}
#title h1 {font-size: 2rem;}
#assessment-run {background: #f97316; color: white; border-color: #f97316;}
#assessment-run:hover {background: #ea580c; border-color: #ea580c;}
"""
TABLE_HEADERS = (
    "Pier",
    "Section",
    "H1 (m)",
    "H2 (m)",
    "h (cm)",
    "Depth (m)",
    "Excavation",
    "P",
    "Likelihood level",
)
CV_PROGRESS = {
    "Detecting piers and section symbols": 0.16,
    "Reading section symbols": 0.27,
    "Detecting beam candidates and level boxes": 0.38,
    "Matching level boxes to beam candidates": 0.50,
    "Reading ground elevation values": 0.64,
}


def sample_pdf_paths() -> dict[str, Path]:
    """Discover PDF samples in the sample image directory."""
    if not SAMPLE_DIR.is_dir():
        return {}
    paths = sorted(
        (path for path in SAMPLE_DIR.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: path.name.casefold(),
    )
    return {path.name: path for path in paths}


def refresh_sample_choices(selected: str | None) -> dict:
    choices = list(sample_pdf_paths())
    return gr.update(choices=choices, value=selected if selected in choices else (choices[0] if choices else None))


def _run_with_gemini_timer(
    work: Callable[[], dict[str, float | None]],
    progress: Callable[..., object],
    started: float,
    interval_seconds: float = 1.0,
) -> dict[str, float | None]:
    stop = Event()
    context = copy_context()

    def update_elapsed() -> None:
        while not stop.wait(interval_seconds):
            elapsed = int(time.perf_counter() - started)
            context.run(
                progress,
                0.83,
                desc=f"CV/OCR complete; Gemini API processing ({elapsed}s elapsed)",
            )

    timer = Thread(target=update_elapsed, daemon=True)
    timer.start()
    try:
        return work()
    finally:
        stop.set()
        timer.join()


def _number(value: object, digits: int = 3) -> object:
    if value is None or value == "":
        return ""
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def _display_rows(records: list[dict]) -> list[list[object]]:
    return [
        [
            record.get("pier_id") or "",
            record.get("section") or "",
            _number(record.get("h1_m")),
            _number(record.get("h2_m")),
            _number(record.get("h_cm"), 1),
            _number(record.get("excavation_depth_m")),
            record.get("excavation_type") or "",
            _number(record.get("p_value"), 2),
            record.get("likelihood_level") or "",
        ]
        for record in records
    ]


def _scoring_values(values: tuple[object, ...]) -> tuple[list[float], dict | None, str]:
    if len(values) != 20:
        raise ValueError("The assessment form is incomplete.")
    indicator_scores = validate_indicator_scores(values[:7])
    advanced = values[7:]
    if any(value is None or value == "" for value in advanced):
        return indicator_scores, None, (
            "P was not calculated because the advanced weights, lambda, or depth Ri scores were left blank."
        )
    scoring = {
        "indicator_scores": indicator_scores,
        "weights": list(advanced[:8]),
        "lambda_factor": advanced[8],
        "depth_scores": list(advanced[9:]),
    }
    return indicator_scores, validate_scoring(scoring), ""


def _safe_error(exc: Exception, api_key: str) -> str:
    message = str(exc)
    return message.replace(api_key, "[redacted]") if api_key else message


def _error_outputs(message: str) -> tuple[str, list, None, None, None, None, None]:
    return (f"Error: {message}", [], None, None, None, None, None)


def run_assessment(
    selected_sample: str,
    uploaded_pdf: str | None,
    icl_image: str | None,
    api_key: str,
    model_id: str,
    device: str,
    *assessment_values: object,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, list[list[object]], str | None, list[str] | None, str | None, str | None, str | None]:
    """Run all stages and return the current run's results for the interface."""
    pdf_path = Path(uploaded_pdf) if uploaded_pdf else sample_pdf_paths().get(selected_sample)
    if pdf_path is None or not pdf_path.is_file():
        return _error_outputs("Select a drawing PDF with a profile page and section pages.")
    if not icl_image or not Path(icl_image).is_file():
        return _error_outputs("Select a valid ICL reference image.")
    api_key = (api_key or os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        return _error_outputs("Enter a Gemini API key or set GEMINI_API_KEY.")
    if not STAGE1.is_file() or not STAGE2.is_file():
        return _error_outputs("Required detection files are missing from the release package.")

    try:
        indicator_scores, scoring, score_notice = _scoring_values(assessment_values)
    except ValueError as exc:
        return _error_outputs(str(exc))
    run_dir = OUTPUT_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    progress(0.05, desc="Rendering drawing pages")
    try:
        render_started = time.perf_counter()
        profile, sections = render_drawing_pdf(pdf_path, run_dir / "input")
        render_seconds = time.perf_counter() - render_started
        progress(0.13, desc=f"PDF rendering complete ({render_seconds:.1f}s); starting CV/OCR")
        cv_started = time.perf_counter()
        cv_result = run_cv_pipeline(
            profile_image=profile,
            output_dir=run_dir / "cv",
            stage1_weights=STAGE1,
            stage2_weights=STAGE2,
            device=device,
            progress=lambda message: progress(CV_PROGRESS.get(message, 0.64), desc=f"CV/OCR: {message}"),
        )
        cv_seconds = time.perf_counter() - cv_started
        progress(0.75, desc=f"CV/OCR complete ({cv_seconds:.1f}s); preparing Gemini request")
        warnings = list(cv_result.get("warnings") or [])
        gemini_started = time.perf_counter()
        try:
            heights = _run_with_gemini_timer(
                lambda: extract_section_heights(
                    sections,
                    Path(icl_image),
                    api_key,
                    model_id,
                    status=lambda message: progress(
                        0.78 if message.startswith("Preparing") else 0.83 if message.startswith("Waiting") else 0.88,
                        desc=f"CV/OCR complete; Gemini: {message}",
                    ),
                ),
                progress,
                gemini_started,
            )
        except Exception as exc:
            heights = {}
            warnings.append(f"Section height extraction failed: {_safe_error(exc, api_key)}")
        gemini_seconds = time.perf_counter() - gemini_started
        progress(0.91, desc=f"Gemini finished ({gemini_seconds:.1f}s); calculating results")
        candidate_records = assess_records(cv_result.get("records", []), heights, scoring)
        records = [row for row in candidate_records if row.get("excavation_type") != "excluded"]
        summary = {
            "source_pdf": pdf_path.name,
            "profile_image": profile.name,
            "section_images": [path.name for path in sections],
            "records": records,
            "excluded_candidate_count": len(candidate_records) - len(records),
            "section_heights_cm": heights,
            "stage_timings_s": {
                "pdf_render": round(render_seconds, 2),
                "cv_ocr": round(cv_seconds, 2),
                "gemini": round(gemini_seconds, 2),
            },
            "indicator_scores": dict(zip(INDICATORS[:7], indicator_scores)),
            "weights": scoring["weights"] if scoring else None,
            "weight_note": (
                "Equal weights of 0.125 are release defaults, not specified by the source interface or guideline."
                if scoring and all(abs(weight - 0.125) < 1e-9 for weight in scoring["weights"])
                else "User-selected weights." if scoring else "Weights were not fully provided."
            ),
            "lambda_value": scoring["lambda_factor"] if scoring else assessment_values[15],
            "depth_scores": scoring["depth_scores"] if scoring else list(assessment_values[16:20]),
            "scoring_note": score_notice,
            "warnings": warnings,
        }
        summary_path = run_dir / "assessment.json"
        with summary_path.open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2)
        visualization = draw_assessment_visualization(
            profile,
            records,
            cv_result.get("artifacts", {}).get("ocr_summary"),
            run_dir / "final_visualization.png",
            cv_result.get("artifacts", {}).get("ground_summary"),
        )
        progress(0.96, desc="Results calculated; generating assessment report")
        report_path = generate_report(
            summary,
            run_dir / "assessment_report.pdf",
            profile_image=profile,
            section_images=sections,
            result_image=visualization,
        )
        progress(1.0, desc="Complete")
        status_parts = [
            f"Completed: {pdf_path.name}",
            f"Records: {len(records)}",
            f"Time: PDF {render_seconds:.1f}s | CV/OCR {cv_seconds:.1f}s | Gemini {gemini_seconds:.1f}s",
        ]
        if score_notice:
            status_parts.append(score_notice)
        if warnings:
            status_parts.append("Warnings: " + " | ".join(warnings))
        return (
            "\n".join(status_parts),
            _display_rows(records),
            str(profile),
            [str(path) for path in sections],
            str(visualization),
            str(summary_path),
            str(report_path),
        )
    except Exception as exc:
        return _error_outputs(
            f"Run failed. Partial results were kept in {run_dir.name}. {_safe_error(exc, api_key)}"
        )


def build_interface() -> gr.Blocks:
    sample_names = list(sample_pdf_paths())
    with gr.Blocks(title="Bridge Component Construction Collapse Risk Assessment Tool") as demo:
        gr.Markdown(
            "# Bridge Component Construction Collapse Risk Assessment Tool\n\n"
            "Upload a bridge drawing PDF with a profile page followed by section views, select a VLM, set indicator scores Ri, "
            "and run the safety risk assessment.",
            elem_id="title",
        )
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 1. Input files")
                with gr.Row():
                    sample = gr.Dropdown(
                        choices=sample_names,
                        value=sample_names[0] if sample_names else None,
                        label="Sample drawing PDF",
                        scale=4,
                    )
                    refresh_samples = gr.Button("Refresh samples", scale=1)
                custom_pdf = gr.UploadButton(
                    "Upload a bridge drawing PDF", file_types=[".pdf"], type="filepath"
                )
                uploaded_name = gr.Textbox(label="Uploaded PDF", interactive=False)
                exemplar = gr.State(str(EXAMPLE))
                exemplar_view = gr.Image(value=str(EXAMPLE), label="ICL reference image", interactive=False)
                exemplar_upload = gr.UploadButton(
                    "Replace ICL reference image", file_types=[".png", ".jpg", ".jpeg", ".webp"], type="filepath"
                )
                gr.Markdown("## 2. VLM setting")
                model = gr.Dropdown(choices=MODEL_CHOICES, value="gemini-3-flash-preview", label="Select VLM")
                key = gr.Textbox(label="API key", type="password", placeholder="Enter your Gemini API key")
                with gr.Accordion("Compute device", open=False):
                    device = gr.Dropdown(["cpu", "cuda:0"], value="cpu", label="Device")
            with gr.Column(scale=1):
                gr.Markdown("## 3. Indicator Score Ri setting")
                indicator_scores = [
                    gr.Number(label=name, value=score, precision=2)
                    for name, score in zip(INDICATORS[:7], DEFAULT_RI)
                ]
                gr.Markdown(
                    "Default P uses eight equal example weights (0.125 each). "
                    "Open the settings below to edit the weights, λ, or depth Ri values."
                )
                with gr.Accordion("Advanced P calculation", open=False):
                    gr.Markdown(
                        "Enter all eight weights γ (each 0–1; total 1), λ, and four depth Ri scores "
                        "to calculate P. Choose one Ri within each depth band's allowed range. "
                        "The calculated d selects the corresponding band for each beam. "
                        "The initial weights are equal example values (0.125 each), not guideline coefficients; "
                        "edit them for your project."
                    )
                    weights = [
                        gr.Number(label=f"Weight γ: {name}", value=weight, precision=4)
                        for name, weight in zip(INDICATORS, DEFAULT_WEIGHTS)
                    ]
                    factor = gr.Dropdown(
                        choices=list(LAMBDA_CHOICES), value=1.0, label="Safety management factor λ"
                    )
                    depth_scores = [
                        gr.Number(label=f"Depth score Ri: {band}", value=score, precision=2)
                        for band, score in zip(DEPTH_BANDS, DEFAULT_DEPTH_RI)
                    ]
                with gr.Row():
                    run_button = gr.Button("Run Safety Assessment", variant="primary", elem_id="assessment-run")
                    clear_button = gr.Button("Clear Outputs")
        gr.Markdown("---")
        status = gr.Textbox(label="Status", lines=4)
        results = gr.Dataframe(headers=list(TABLE_HEADERS), label="Assessment results", interactive=False)
        with gr.Row():
            profile_view = gr.Image(label="Profile drawing")
            section_view = gr.Gallery(label="Section drawings", columns=2, height=300)
            result_view = gr.Image(label="Final ground tie beam visualization")
        with gr.Row():
            json_file = gr.File(label="Structured results")
            pdf_file = gr.File(label="Assessment report")
        run_button.click(
            run_assessment,
            inputs=[sample, custom_pdf, exemplar, key, model, device, *indicator_scores, *weights, factor, *depth_scores],
            outputs=[status, results, profile_view, section_view, result_view, json_file, pdf_file],
        )
        custom_pdf.upload(
            lambda path: Path(path).name if path else "",
            inputs=custom_pdf,
            outputs=uploaded_name,
        )
        sample.input(
            lambda _sample: (None, ""),
            inputs=sample,
            outputs=[custom_pdf, uploaded_name],
        )
        refresh_samples.click(refresh_sample_choices, inputs=sample, outputs=sample)
        exemplar_upload.upload(
            lambda path: (path, path),
            inputs=exemplar_upload,
            outputs=[exemplar, exemplar_view],
        )
        clear_button.click(
            lambda: ("", [], None, None, None, None, None),
            outputs=[status, results, profile_view, section_view, result_view, json_file, pdf_file],
        )
    return demo


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    build_interface().queue().launch(
        server_name="127.0.0.1", inbrowser=True, allowed_paths=[str(OUTPUT_DIR)],
        css=CUSTOM_CSS, footer_links=[], run_history=False,
    )
