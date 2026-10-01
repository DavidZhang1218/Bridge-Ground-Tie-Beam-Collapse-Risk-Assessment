"""Apply deterministic excavation and possibility scoring rules."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .vlm import normalize_section_label


DEMO_INDICATOR_SCORES = (55.0, 45.0, 80.0, 40.0, 25.0, 35.0, 20.0)
INDICATOR_NAMES = (
    "Groundwater condition",
    "Foundation pit surroundings",
    "Construction-period rainfall condition",
    "Geological ground condition",
    "Excavation method",
    "Monitoring condition",
    "Technical applicability",
    "Foundation pit depth",
)
DEPTH_BANDS = ("<3 m", "3–<5 m", "5–10 m", ">10 m")
DEPTH_SCORE_RANGES = ((0.0, 25.0), (25.0, 50.0), (50.0, 75.0), (75.0, 100.0))
ALLOWED_LAMBDA = (0.9, 0.95, 1.0, 1.05, 1.1)


def _number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def validate_indicator_scores(values: Sequence) -> list[float]:
    """Validate the seven user-selected site and construction Ri scores."""
    if isinstance(values, (str, bytes)) or len(values) != 7:
        raise ValueError("Seven indicator Ri scores are required.")
    scores = [_number(value) for value in values]
    if any(score is None or score < 0 or score > 100 for score in scores):
        raise ValueError("Each indicator Ri score must be between 0 and 100.")
    return scores


def validate_scoring(scoring: Mapping | None) -> dict | None:
    """Return validated scoring inputs, or None when required inputs are absent."""
    if scoring is None:
        return None
    if not isinstance(scoring, Mapping):
        raise ValueError("Scoring settings must be an object.")
    indicator_scores = validate_indicator_scores(
        scoring.get("indicator_scores", DEMO_INDICATOR_SCORES)
    )
    if any(scoring.get(key) is None for key in ("weights", "lambda_factor", "depth_scores")):
        return None
    weights = scoring["weights"]
    depth_scores = scoring["depth_scores"]
    if not isinstance(weights, Sequence) or isinstance(weights, (str, bytes)) or len(weights) != 8:
        raise ValueError("Eight indicator weights are required.")
    if not isinstance(depth_scores, Sequence) or isinstance(depth_scores, (str, bytes)) or len(depth_scores) != 4:
        raise ValueError("Four depth scores are required.")
    if any(value is None or (isinstance(value, str) and not value.strip()) for value in (*weights, *depth_scores, scoring["lambda_factor"])):
        return None
    weights = [_number(value) for value in weights]
    depth_scores = [_number(value) for value in depth_scores]
    factor = _number(scoring["lambda_factor"])
    if any(value is None for value in weights + depth_scores) or factor is None:
        raise ValueError("Indicator weights, depth scores, and the safety management factor must be numeric.")
    if any(value < 0 or value > 1 for value in weights):
        raise ValueError("Each indicator weight must be between 0 and 1.")
    if not math.isclose(sum(weights), 1.0, abs_tol=1e-6):
        raise ValueError("The eight indicator weights must sum to 1.")
    if not any(math.isclose(factor, allowed, abs_tol=1e-9) for allowed in ALLOWED_LAMBDA):
        raise ValueError("The safety management factor must be 0.9, 0.95, 1.0, 1.05, or 1.1.")
    for index, score in enumerate(depth_scores):
        low, high = DEPTH_SCORE_RANGES[index]
        if not low <= score <= high:
            raise ValueError(f"The depth score for {DEPTH_BANDS[index]} must be between {low:g} and {high:g}.")
    return {
        "indicator_scores": indicator_scores,
        "weights": weights,
        "lambda_factor": factor,
        "depth_scores": depth_scores,
    }


def depth_band_index(depth_m: float) -> int:
    """Map a positive excavation depth to the guideline's four bands."""
    depth = _number(depth_m)
    if depth is None or depth <= 0:
        raise ValueError("Excavation depth must be positive.")
    if depth < 3:
        return 0
    if depth < 5:
        return 1
    if depth <= 10:
        return 2
    return 3


def likelihood_level(p_value: float) -> int:
    """Classify P using Table 7 of JT/T 1375.2-2025."""
    score = _number(p_value)
    if score is None or score < 0:
        raise ValueError("P must be nonnegative.")
    if score > 60:
        return 5
    if score > 45:
        return 4
    if score > 30:
        return 3
    if score > 15:
        return 2
    return 1


def score_excavation(depth_m: float, scoring: Mapping) -> tuple[float, int, float, str]:
    """Return P, likelihood level, selected depth Ri, and depth band."""
    settings = validate_scoring(scoring)
    if settings is None:
        raise ValueError("All scoring inputs are required to compute P.")
    index = depth_band_index(depth_m)
    depth_score = settings["depth_scores"][index]
    scores = (*settings["indicator_scores"], depth_score)
    p_value = round(settings["lambda_factor"] * sum(
        score * weight for score, weight in zip(scores, settings["weights"])
    ), 8)
    return p_value, likelihood_level(p_value), depth_score, DEPTH_BANDS[index]


def _record_section(record: Mapping) -> str | None:
    label = record.get("section")
    if label is None:
        symbol = record.get("section_symbol")
        if isinstance(symbol, Mapping):
            label = symbol.get("label") or symbol.get("text") or symbol.get("value")
    return normalize_section_label(label)


def _record_elevation(record: Mapping, flat_key: str, nested_key: str) -> float | None:
    value = record.get(flat_key)
    if value is None:
        nested = record.get(nested_key)
        if isinstance(nested, Mapping):
            value = nested.get("value")
    return _number(value)


def assess_records(records: list[dict], heights: dict, scoring: dict | None) -> list[dict]:
    """Join section heights to candidates and assess each valid excavation."""
    settings = validate_scoring(scoring)
    normalized_heights = {}
    for label, value in heights.items():
        section = normalize_section_label(label)
        if section is not None:
            normalized_heights[section] = _number(value)
    assessed = []
    for record in records:
        result = dict(record)
        section = _record_section(record)
        h1 = _record_elevation(record, "h1_m", "beam_level")
        h2 = _record_elevation(record, "h2_m", "ground_level")
        h_cm = normalized_heights.get(section)
        result.update(
            section=section,
            h_cm=h_cm,
            h1_m=h1,
            h2_m=h2,
            excavation_depth_m=None,
            excavation_type=None,
            depth_score=None,
            depth_band=None,
            p_value=None,
            likelihood_level=None,
            scoring_status=None,
        )
        if h_cm is None or h_cm <= 0:
            result["excavation_type"] = "height_unavailable"
        elif h1 is None or h2 is None:
            result["excavation_type"] = "elevation_unavailable"
        else:
            height_m = round(h_cm / 100.0, 8)
            difference = round(h1 - h2, 8)
            if difference >= height_m:
                result["excavation_type"] = "excluded"
            else:
                result["excavation_type"] = "full" if difference <= 0 else "partial"
                depth = round(height_m - difference, 8)
                result["excavation_depth_m"] = depth
                if settings is not None:
                    p_value, level, depth_score, band = score_excavation(depth, settings)
                    result.update(
                        p_value=p_value,
                        likelihood_level=level,
                        depth_score=depth_score,
                        depth_band=band,
                        scoring_status="computed",
                    )
        if result["scoring_status"] is None:
            result["scoring_status"] = (
                "not_configured" if result["excavation_type"] in ("full", "partial") else result["excavation_type"]
            )
        assessed.append(result)
    return assessed
