"""Run the drawing recognition stages on one profile drawing."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any, Callable

from . import cv_ground, cv_match, cv_section_ocr, cv_stage1, cv_stage2


def _names(model: Any) -> dict[int, str]:
    raw = getattr(model, "names", {})
    if isinstance(raw, dict):
        return {int(key): str(value).strip().lower() for key, value in raw.items()}
    if isinstance(raw, (list, tuple)):
        return {index: str(value).strip().lower() for index, value in enumerate(raw)}
    return {}


def _record(item: dict[str, Any]) -> dict[str, Any]:
    section_data = item.get("section_symbol") or {}
    section = section_data.get("text") or section_data.get("label")
    if section in (None, "", "NO_SEC"):
        section = None
    pier_id = item.get("source_pier_id")
    if pier_id is None:
        pier_id = item.get("pier_group_id")
    h1 = (item.get("beam_level") or {}).get("value")
    h2 = (item.get("ground_level") or {}).get("value")
    return {
        "pier_id": int(pier_id) if pier_id is not None else None,
        "section": str(section) if section is not None else None,
        "h1_m": float(h1) if h1 is not None else None,
        "h1_ocr_correction": (item.get("beam_level") or {}).get("correction"),
        "h2_m": float(h2) if h2 is not None else None,
        "candidate_bbox": item.get("candidate_bbox"),
        "candidate_id": int(item.get("candidate_id", 0)),
        "status": str(item.get("status") or "unknown"),
    }


def run_cv_pipeline(
    profile_image: Path,
    output_dir: Path,
    stage1_weights: Path,
    stage2_weights: Path,
    device: str = "cpu",
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Detect piers, sections, beam candidates, elevations, and their links."""
    profile_image = Path(profile_image).resolve()
    output_dir = Path(output_dir).resolve()
    stage1_weights = Path(stage1_weights).resolve()
    stage2_weights = Path(stage2_weights).resolve()
    if not profile_image.is_file():
        raise FileNotFoundError(f"Drawing image not found: {profile_image}")
    if not stage1_weights.is_file() or not stage2_weights.is_file():
        raise FileNotFoundError("A required detection file is missing.")
    output_dir.mkdir(parents=True, exist_ok=True)
    emit = progress if progress is not None else lambda _message: None
    resolved_device = cv_stage1.resolve_device(device)
    stem = profile_image.stem
    warnings: list[str] = []

    emit("Detecting piers and section symbols")
    stage1_model = cv_stage1.build_model(stage1_weights)
    if _names(stage1_model) != {0: "pier", 1: "section_symbol"}:
        raise ValueError("The first detection file has unexpected classes.")
    stage1_args = Namespace(
        conf=0.2,
        imgsz=1280,
        iou=0.70,
        max_det=300,
        dedup_iou=0.50,
        weights=str(stage1_weights),
        weights_resolved=str(stage1_weights),
    )
    stage1_root = output_dir / "stage1"
    cv_stage1.process_one_image(profile_image, stage1_root, stage1_model, stage1_args, resolved_device)
    del stage1_model
    stage1_summary = stage1_root / stem / "json" / f"{stem}_pier_section_summary.json"

    emit("Reading section symbols")
    ocr_root = output_dir / "ocr"
    section_ocr = cv_section_ocr.OCRRunner("auto", cv_section_ocr.DEFAULT_OCR_MODEL, cv_section_ocr.resolve_ocr_device("auto", resolved_device))
    cv_section_ocr.process_one(stage1_summary, ocr_root, profile_image.parent, section_ocr, threshold=125)
    ocr_summary = ocr_root / stem / "json" / f"{stem}_ocr_summary.json"

    emit("Detecting beam candidates and level boxes")
    stage2_model = cv_stage2.build_model(stage2_weights)
    class_names = cv_stage2.model_class_names(stage2_model)
    if set(class_names.values()) != {"candidate", "beam_level"} or len(class_names) != 2:
        raise ValueError("The second detection file has unexpected classes.")
    stage2_args = Namespace(
        conf=0.2,
        imgsz=1280,
        iou=0.70,
        max_det=300,
        weights=str(stage2_weights),
        weights_resolved=str(stage2_weights),
    )
    stage2_root = output_dir / "stage2"
    stage2_summary_data = cv_stage2.process_one_project(
        stage1_root / stem / "pier_crops",
        stage2_root,
        stage2_model,
        class_names,
        stage2_args,
        resolved_device,
    )
    del stage2_model
    stage2_summary = Path(stage2_summary_data["outputs"]["summary_json"])

    emit("Matching level boxes to beam candidates")
    match_root = output_dir / "match"
    match_ocr = cv_match.OCRRunner("auto", cv_match.DEFAULT_OCR_MODEL, cv_match.resolve_ocr_device("auto", resolved_device))
    cv_match.process_one(stage1_summary, ocr_root, stage2_root, match_root, profile_image.parent, match_ocr, threshold=160)
    match_summary = match_root / stem / "json" / f"{stem}_matched_results.json"

    emit("Reading ground elevation values")
    ground_root = output_dir / "ground"
    ground_ocr = cv_ground.OCRRunner("auto", cv_ground.DEFAULT_OCR_MODEL, cv_ground.resolve_ocr_device("auto", resolved_device))
    ground_data = cv_ground.process_one(match_summary, ground_root, profile_image.parent, ground_ocr, threshold=165)
    if ground_data["status"] != "completed":
        warnings.append("The ground elevation row could not be read automatically; H2 values are unavailable.")
    records = [_record(item) for item in ground_data["candidate_beams"]]
    if not records:
        warnings.append("No beam candidates were detected in the drawing.")
    if any(item["h1_m"] is None for item in records):
        warnings.append("One or more beam elevations could not be read.")
    if any(item["h1_ocr_correction"] is not None for item in records):
        warnings.append("One or more beam elevations were corrected using OCR alternatives and drawing geometry.")
    if ground_data["status"] == "completed" and any(item["h2_m"] is None for item in records):
        warnings.append("One or more ground elevations could not be read.")
    if any(item["section"] is None for item in records):
        warnings.append("One or more section symbols could not be assigned.")
    return {
        "records": records,
        "artifacts": {
            "stage1_summary": stage1_summary,
            "ocr_summary": ocr_summary,
            "stage2_summary": stage2_summary,
            "match_summary": match_summary,
            "ground_summary": Path(ground_data["outputs"]["summary_json"]),
            "visualization": Path(ground_data["outputs"]["visualization"]),
        },
        "warnings": warnings,
    }
