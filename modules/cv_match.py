from __future__ import annotations


import math


import json


import os


import re


import subprocess


import tempfile
from statistics import median


from pathlib import Path


from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


import numpy as np


from PIL import Image, ImageDraw, ImageFont


os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


try:
    import torch
except Exception:
    torch = None


THIS_DIR = Path(__file__).resolve().parent


DEFAULT_OCR_MODEL = "PP-OCRv5_server_rec"


MIN_OCR_SCORE = 0.50


MIN_ELEVATION_VALUE = -1000.0


MAX_ELEVATION_VALUE = 10000.0


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
]


def resolve_path(path_text: str | Path, *, base: Path = THIS_DIR) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_font(size: int = 16) -> ImageFont.ImageFont:
    for font_path in FONT_CANDIDATES:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def resolve_device(requested: str) -> str:
    req = (requested or "cpu").strip().lower()
    if req.startswith("gpu"):
        suffix = req.split(":", 1)[1] if ":" in req else "0"
        req = f"cuda:{suffix}"
    if req.startswith("cuda") and (torch is None or not torch.cuda.is_available()):
        print(f"[WARN] Requested device '{requested}' but CUDA is unavailable, fallback to cpu.")
        return "cpu"
    return req


def resolve_ocr_device(raw_device: str, yolo_device: str) -> str:
    raw = (raw_device or "auto").strip().lower()
    if raw == "auto":
        raw = "gpu:0" if str(yolo_device).startswith("cuda") else "cpu"
    if raw.startswith("cuda"):
        suffix = raw.split(":", 1)[1] if ":" in raw else "0"
        raw = f"gpu:{suffix}"
    if raw.startswith("gpu") and (torch is None or not torch.cuda.is_available()):
        return "cpu"
    return raw


def list_detector_summaries(detector_results: Path) -> List[Path]:
    direct = list(detector_results.glob("json/*_pier_section_summary.json"))
    nested = list(detector_results.glob("*/json/*_pier_section_summary.json"))
    return sorted(direct + nested)


def find_project_json(root: Path, stem: str, suffix: str) -> Path:
    preferred = root / stem / "json" / f"{stem}{suffix}"
    if preferred.exists():
        return preferred
    matches = sorted(root.glob(f"*/json/*{suffix}"))
    for path in matches:
        if path.name == f"{stem}{suffix}" or path.name.startswith(stem):
            return path
    raise FileNotFoundError(f"Cannot find {suffix} for {stem} under {root}")


def locate_original_image(summary: Dict[str, Any], image_root: Optional[Path]) -> Path:
    image_path = Path(str(summary.get("image_path") or ""))
    if image_path.exists():
        return image_path
    image_name = str(summary.get("image_name") or "")
    if image_root is not None and image_name:
        direct = image_root / image_name
        if direct.exists():
            return direct.resolve()
        stem = Path(image_name).stem
        for ext in SUPPORTED_IMAGE_EXTS:
            candidate = image_root / f"{stem}{ext}"
            if candidate.exists():
                return candidate.resolve()
    raise FileNotFoundError(f"Original image not found for {summary.get('image_name')}")


def normalize_class_name(name: Any) -> str:
    text = str(name or "").strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    return "_".join(part for part in text.split("_") if part)


def class_bucket(name: Any) -> str:
    normalized = normalize_class_name(name)
    if normalized in {"candidate", "beam_candidate", "ground_beam_candidate", "ground_tie_beam_candidate"}:
        return "candidate"
    if normalized in {"beam_level", "beamlevel"}:
        return "beam_level"
    return "other"


def box_center(box: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def box_size(box: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(1.0, x2 - x1), max(1.0, y2 - y1)


def clip_box(box: Sequence[float], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0, min(int(round(x1)), width))
    y1 = max(0, min(int(round(y1)), height))
    x2 = max(0, min(int(round(x2)), width))
    y2 = max(0, min(int(round(y2)), height))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return [x1, y1, x2, y2]


def padded_box(box: Sequence[float], width: int, height: int, ratio: float = 0.15, min_pad: int = 6) -> List[int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = box_size(box)
    pad_x = max(float(min_pad), bw * ratio)
    pad_y = max(float(min_pad), bh * ratio)
    return clip_box([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], width, height)


def threshold_image(img: Image.Image, threshold: int) -> Image.Image:
    threshold = max(0, min(255, int(threshold)))
    gray = img.convert("L")
    bw = gray.point(lambda p: 255 if p >= threshold else 0)
    return bw.convert("RGB")


def upscale_image(img: Image.Image, scale: float = 2.4, max_side: int = 960) -> Image.Image:
    side = max(img.size)
    if side <= 0:
        return img
    scale = min(float(scale), float(max_side) / float(side))
    scale = max(1.0, scale)
    size = (max(1, int(round(img.width * scale))), max(1, int(round(img.height * scale))))
    if size == img.size:
        return img
    return img.resize(size, Image.Resampling.LANCZOS)


def parse_paddle_output(item: Any) -> Dict[str, Any]:
    data = getattr(item, "json", item)
    if callable(data):
        data = data()
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            data = {}
    if isinstance(data, dict) and "res" in data:
        data = data["res"]
    if not isinstance(data, dict):
        data = {}
    return {
        "text": str(data.get("rec_text") or data.get("text") or "").strip(),
        "score": float(data.get("rec_score") or data.get("score") or 0.0),
        "raw": data,
    }


class OCRRunner:
    def __init__(self, backend: str, model_name: str, device: str) -> None:
        self.backend = backend
        self.model_name = model_name
        self.device = device
        self._paddle_model: Any = None

    def _build_paddle(self) -> Any:
        if self._paddle_model is not None:
            return self._paddle_model
        try:
            from paddleocr import TextRecognition
        except Exception as exc:
            raise RuntimeError(f"PaddleOCR TextRecognition is unavailable: {exc}") from exc
        self._paddle_model = TextRecognition(model_name=self.model_name, device=self.device)
        return self._paddle_model

    def predict_one(self, img: Image.Image) -> Dict[str, Any]:
        if self.backend == "none":
            return {"text": "", "score": 0.0, "backend": "none"}
        if self.backend in {"auto", "paddle"}:
            try:
                model = self._build_paddle()
                arr = np.array(img.convert("RGB"))
                try:
                    outputs = model.predict(input=arr, batch_size=1)
                except TypeError:
                    outputs = model.predict([arr])
                parsed = [parse_paddle_output(item) for item in (outputs or [])]
                best = max(parsed, key=lambda item: item.get("score", 0.0), default={"text": "", "score": 0.0})
                best["backend"] = "paddle"
                return best
            except Exception as exc:
                if self.backend == "paddle":
                    raise
                print(f"[WARN] PaddleOCR unavailable, fallback to tesseract: {exc}")
                self.backend = "tesseract"
        return self._predict_tesseract(img)

    def _predict_tesseract(self, img: Image.Image) -> Dict[str, Any]:
        whitelist = "0123456789.+-OolI"
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            img.convert("RGB").save(tmp_path)
            cmd = [
                "tesseract",
                str(tmp_path),
                "stdout",
                "-l",
                "eng",
                "--psm",
                "7",
                "-c",
                f"tessedit_char_whitelist={whitelist}",
            ]
            completed = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
            text = str(completed.stdout or "").strip()
            return {"text": text, "score": 0.60 if text else 0.0, "backend": "tesseract"}
        except FileNotFoundError:
            return {"text": "", "score": 0.0, "backend": "tesseract_missing"}
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


def parse_numeric_text(text: Any) -> Tuple[Optional[float], str]:
    raw = str(text or "").strip()
    if not re.search(r"\d", raw):
        return None, raw
    cleaned = raw.replace("O", "0").replace("o", "0").replace("I", "1").replace("l", "1")
    cleaned = cleaned.replace("\uff0c", ".").replace(",", ".").replace("\u3002", ".")
    cleaned = cleaned.replace("\u2212", "-").replace("\u2014", "-").replace("\u2013", "-")
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", cleaned)
    if not matches:
        return None, raw
    for token in sorted(matches, key=len, reverse=True):
        try:
            value = float(token)
        except Exception:
            continue
        if MIN_ELEVATION_VALUE <= value <= MAX_ELEVATION_VALUE:
            return value, token
    return None, raw


def recognize_level_value(
    full_image: Image.Image,
    bbox: Sequence[float],
    ocr: OCRRunner,
    threshold: int,
) -> Dict[str, Any]:
    crop_box = padded_box(bbox, full_image.width, full_image.height, ratio=0.18, min_pad=8)
    crop = full_image.crop(tuple(crop_box))
    variants = [
        ("orig_up", upscale_image(crop)),
        ("binary_up", upscale_image(threshold_image(crop, threshold))),
        ("dark_binary_up", upscale_image(threshold_image(crop, max(0, int(threshold) - 50)))),
    ]
    candidates: List[Dict[str, Any]] = []
    for name, img in variants:
        pred = ocr.predict_one(img)
        value, numeric_text = parse_numeric_text(pred.get("text", ""))
        raw_score = float(pred.get("score", 0.0) or 0.0)
        if raw_score < MIN_OCR_SCORE:
            value = None
        score = raw_score + (0.18 if value is not None else -0.10)
        candidates.append(
            {
                "variant": name,
                "backend": pred.get("backend"),
                "raw_text": pred.get("text", ""),
                "raw_score": raw_score,
                "numeric_text": numeric_text,
                "numeric_value": value,
                "final_score": round(score, 6),
            }
        )
    candidates.sort(key=lambda item: item["final_score"], reverse=True)
    best = candidates[0] if candidates else {}
    return {
        "crop_rect_in_full": crop_box,
        "recognized_value": best.get("numeric_value"),
        "ocr_text": best.get("raw_text", ""),
        "ocr_score": best.get("raw_score", 0.0),
        "ocr_variant": best.get("variant", ""),
        "candidates": candidates,
    }


def correct_level_decimal_omissions(levels: List[Dict[str, Any]]) -> None:
    """Resolve a missing decimal only when nearby level geometry supports it."""
    references = []
    box_heights = []
    for level in levels:
        box = level.get("bbox_in_full") or []
        if len(box) == 4:
            box_heights.append(max(1.0, float(box[3]) - float(box[1])))
        value = level.get("recognized_value")
        text = str(level.get("ocr_text") or "")
        center = level.get("center_in_full") or []
        if value is not None and "." in text and len(center) == 2 and float(level.get("ocr_score", 0.0)) >= MIN_OCR_SCORE:
            references.append((float(center[1]), float(value)))
    if len(references) < 4 or not box_heights:
        return

    typical_box_height = float(median(box_heights))
    nearby_distance = 8.0 * typical_box_height
    same_row_distance = 0.5 * typical_box_height
    for level in levels:
        original_text = str(level.get("ocr_text") or "").strip()
        digit_match = re.fullmatch(r"([+-]?)(\d{4,6})", original_text)
        original_value = level.get("recognized_value")
        center = level.get("center_in_full") or []
        if digit_match is None or original_value is None or len(center) != 2:
            continue
        y = float(center[1])
        above = [(ref_y, value) for ref_y, value in references if 0 < y - ref_y <= nearby_distance]
        below = [(ref_y, value) for ref_y, value in references if 0 < ref_y - y <= nearby_distance]
        if not above or not below:
            continue
        upper_y = max(ref_y for ref_y, _value in above)
        lower_y = min(ref_y for ref_y, _value in below)
        upper_value = float(median(value for ref_y, value in above if upper_y - ref_y <= same_row_distance))
        lower_value = float(median(value for ref_y, value in below if ref_y - lower_y <= same_row_distance))
        if upper_value <= lower_value:
            continue
        span = upper_value - lower_value
        tolerance = max(0.75, 0.2 * span)
        if lower_value - tolerance <= float(original_value) <= upper_value + tolerance:
            continue
        expected = upper_value + (lower_value - upper_value) * (y - upper_y) / (lower_y - upper_y)
        digits = digit_match.group(2)
        sign = -1.0 if digit_match.group(1) == "-" else 1.0
        alternatives = []
        for candidate in (level.get("ocr_debug") or {}).get("candidates", []):
            text = str(candidate.get("numeric_text") or "")
            value = candidate.get("numeric_value")
            if value is None or "." not in text or re.sub(r"\D", "", text) != digits:
                continue
            if float(candidate.get("raw_score", 0.0)) < MIN_OCR_SCORE:
                continue
            alternatives.append((float(value), "ocr_variant", candidate))
        if (not alternatives and len(above) >= 2 and len(below) >= 2
                and y - upper_y <= 4.0 * typical_box_height
                and lower_y - y <= 4.0 * typical_box_height):
            for position in range(1, len(digits)):
                value = sign * float(f"{digits[:position]}.{digits[position:]}")
                alternatives.append((value, "decimal_insertion", None))
        plausible = [
            item for item in alternatives
            if lower_value - tolerance <= item[0] <= upper_value + tolerance
            and abs(item[0] - expected) <= max(1.5, 0.35 * span)
        ]
        if not plausible:
            continue
        if plausible[0][1] == "decimal_insertion" and len({item[0] for item in plausible}) != 1:
            continue
        corrected_value, source, candidate = min(plausible, key=lambda item: abs(item[0] - expected))
        correction = {
            "source": source,
            "original_value": original_value,
            "corrected_value": corrected_value,
            "original_text": original_text,
            "upper_level_value": round(upper_value, 3),
            "lower_level_value": round(lower_value, 3),
            "expected_value": round(expected, 3),
        }
        level["recognized_value"] = corrected_value
        level["recognized_text"] = str(corrected_value)
        level["context_correction"] = correction
        if candidate is not None:
            level["ocr_text"] = str(candidate.get("raw_text") or candidate.get("numeric_text") or "")
            level["ocr_score"] = float(candidate.get("raw_score", 0.0))
            level["ocr_variant"] = str(candidate.get("variant") or "")
        level["ocr_debug"]["context_correction"] = correction


def crop_metadata_by_name(detector_summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in detector_summary.get("pier_crops", []) or []:
        file_name = str(item.get("file_name") or Path(str(item.get("path") or "")).name)
        if file_name:
            out[file_name] = item
    return out


def collect_detector2_items(detector2_summary: Dict[str, Any], detector_summary: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    crop_meta = crop_metadata_by_name(detector_summary)
    candidates: List[Dict[str, Any]] = []
    levels: List[Dict[str, Any]] = []
    for crop in detector2_summary.get("crops", []) or []:
        meta = crop_meta.get(str(crop.get("image_name") or ""))
        rect = meta.get("rect_in_full") if isinstance(meta, dict) else None
        for det in crop.get("detections", []) or []:
            bucket = class_bucket(det.get("class_name"))
            if bucket not in {"candidate", "beam_level"}:
                continue
            full_box = det.get("bbox_in_full")
            if not isinstance(full_box, list) or len(full_box) != 4:
                crop_box = det.get("bbox_in_crop")
                if isinstance(rect, list) and len(rect) == 4 and isinstance(crop_box, list) and len(crop_box) == 4:
                    full_box = [
                        float(crop_box[0]) + float(rect[0]),
                        float(crop_box[1]) + float(rect[1]),
                        float(crop_box[2]) + float(rect[0]),
                        float(crop_box[3]) + float(rect[1]),
                    ]
                else:
                    continue
            item = {
                "class_name": bucket,
                "score": float(det.get("score", 0.0) or 0.0),
                "bbox_in_full": [float(v) for v in full_box],
                "center_in_full": box_center(full_box),
                "bbox_in_crop": det.get("bbox_in_crop"),
                "source_crop": crop.get("image_name"),
                "source_pier_id": det.get("source_pier_id") or (meta.get("pier_id") if isinstance(meta, dict) else None),
            }
            if bucket == "candidate":
                candidates.append(item)
            else:
                levels.append(item)
    candidates.sort(key=lambda item: (item["center_in_full"][0], item["center_in_full"][1]))
    levels.sort(key=lambda item: (item["center_in_full"][0], item["center_in_full"][1]))
    for idx, item in enumerate(candidates, start=1):
        item["candidate_id"] = idx
    for idx, item in enumerate(levels, start=1):
        item["level_id"] = idx
    return candidates, levels


def build_match_score(candidate_box: Sequence[float], level_box: Sequence[float]) -> float:
    candidate_top = ((float(candidate_box[0]) + float(candidate_box[2])) / 2.0, float(candidate_box[1]))
    level_bottom = ((float(level_box[0]) + float(level_box[2])) / 2.0, float(level_box[3]))
    return math.hypot(candidate_top[0] - level_bottom[0], candidate_top[1] - level_bottom[1])


def greedy_match(candidates: List[Dict[str, Any]], levels: List[Dict[str, Any]]) -> Tuple[Dict[int, int], List[Dict[str, Any]]]:
    pairs: List[Dict[str, Any]] = []
    for candidate in candidates:
        for level in levels:
            candidate_crop = str(candidate.get("source_crop") or "")
            level_crop = str(level.get("source_crop") or "")
            candidate_pier = candidate.get("source_pier_id")
            level_pier = level.get("source_pier_id")
            same_crop = bool(candidate_crop and level_crop and candidate_crop == level_crop)
            same_pier = bool(
                candidate_pier is not None
                and level_pier is not None
                and str(candidate_pier) == str(level_pier)
            )
            if not (same_crop or same_pier):
                continue
            score = build_match_score(candidate["bbox_in_full"], level["bbox_in_full"])
            pairs.append(
                {
                    "candidate_id": int(candidate["candidate_id"]),
                    "level_id": int(level["level_id"]),
                    "score": round(float(score), 6),
                }
            )
    pairs.sort(key=lambda item: (item["score"], item["candidate_id"], item["level_id"]))
    used_candidates: set[int] = set()
    used_levels: set[int] = set()
    matched: Dict[int, int] = {}
    for pair in pairs:
        candidate_id = int(pair["candidate_id"])
        level_id = int(pair["level_id"])
        if candidate_id in used_candidates or level_id in used_levels:
            continue
        used_candidates.add(candidate_id)
        used_levels.add(level_id)
        matched[candidate_id] = level_id
    return matched, pairs


def int_to_roman(value: int) -> str:
    parts: List[str] = []
    rest = int(value)
    for number, glyph in ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while rest >= number:
            parts.append(glyph)
            rest -= number
    return "".join(parts)


def symbol_label(symbol: Dict[str, Any]) -> str:
    text = str(symbol.get("symbol_text") or "").strip()
    if text:
        return text
    value = symbol.get("symbol_value")
    if value is None:
        return ""
    try:
        return int_to_roman(int(value))
    except Exception:
        return str(value)


def assign_section_symbol_by_x(candidate_center_x: float, section_symbols: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = sorted(
        [
            symbol
            for symbol in section_symbols
            if symbol.get("center_in_full") and symbol.get("symbol_value") is not None
        ],
        key=lambda item: float(item["center_in_full"][0]),
    )
    if not valid:
        return {
            "status": "no_valid_section_symbols",
            "selected_symbol_id": None,
            "selected_symbol_value": None,
            "selected_symbol_text": None,
            "selected_symbol_bbox": None,
            "selected_symbol_orientation": None,
            "candidate_center_x": float(candidate_center_x),
            "assignment_rule": "directed_section_symbol_orientation",
        }

    left_symbol = None
    right_symbol = None
    for symbol in valid:
        sx = float(symbol["center_in_full"][0])
        if sx <= candidate_center_x:
            left_symbol = symbol
        elif right_symbol is None:
            right_symbol = symbol
            break

    def orientation(symbol: Optional[Dict[str, Any]]) -> Optional[str]:
        if symbol is None:
            return None
        value = str(symbol.get("orientation") or "").strip().lower()
        return value if value in {"left", "right"} else None

    def points_to_candidate(symbol: Optional[Dict[str, Any]]) -> bool:
        if symbol is None:
            return False
        sx = float(symbol["center_in_full"][0])
        orient = orientation(symbol)
        return bool((orient == "right" and candidate_center_x >= sx) or (orient == "left" and candidate_center_x <= sx))

    left_points = points_to_candidate(left_symbol)
    right_points = points_to_candidate(right_symbol)
    selected = None
    status = "no_symbol_points_to_candidate"
    if left_points and not right_points:
        selected = left_symbol
        status = "left_symbol_points_right"
    elif right_points and not left_points:
        selected = right_symbol
        status = "right_symbol_points_left"
    elif left_points and right_points:
        selected = left_symbol if left_symbol is not None else right_symbol
        status = "both_symbols_point_to_candidate"
    else:
        choices = [item for item in (left_symbol, right_symbol) if item is not None]
        selected = min(choices, key=lambda item: abs(float(item["center_in_full"][0]) - candidate_center_x)) if choices else None
        status = "nearest_symbol_fallback" if selected is not None else status

    return {
        "status": status,
        "selected_symbol_id": int(selected.get("section_symbol_id") or selected.get("id")) if selected is not None else None,
        "selected_symbol_value": selected.get("symbol_value") if selected is not None else None,
        "selected_symbol_text": symbol_label(selected) if selected is not None else None,
        "selected_symbol_bbox": selected.get("bbox_in_full") if selected is not None else None,
        "selected_symbol_orientation": orientation(selected),
        "left_symbol_id": int(left_symbol.get("section_symbol_id") or left_symbol.get("id")) if left_symbol is not None else None,
        "right_symbol_id": int(right_symbol.get("section_symbol_id") or right_symbol.get("id")) if right_symbol is not None else None,
        "left_symbol_orientation": orientation(left_symbol),
        "right_symbol_orientation": orientation(right_symbol),
        "left_symbol_points_to_candidate": left_points,
        "right_symbol_points_to_candidate": right_points,
        "candidate_center_x": float(candidate_center_x),
        "assignment_rule": "directed_section_symbol_orientation",
    }


def draw_text_box(draw: ImageDraw.ImageDraw, xy: Tuple[float, float], text: str, font: ImageFont.ImageFont) -> None:
    if not text:
        return
    x, y = xy
    try:
        bbox = draw.textbbox((x, y), text, font=font)
        draw.rectangle(bbox, fill=(255, 255, 180))
        draw.text((x, y), text, fill=(0, 0, 0), font=font)
    except Exception:
        draw.text((x, y), text, fill=(0, 0, 0), font=font)


def draw_match_visualization(
    image: Image.Image,
    candidates: List[Dict[str, Any]],
    levels: List[Dict[str, Any]],
    matched_candidate_to_level: Dict[int, int],
    output_path: Path,
) -> None:
    ensure_dir(output_path.parent)
    vis = image.convert("RGB").copy()
    draw = ImageDraw.Draw(vis)
    font = load_font(15)
    level_by_id = {int(item["level_id"]): item for item in levels}
    for level in levels:
        x1, y1, x2, y2 = [float(v) for v in level["bbox_in_full"]]
        text = str(level.get("recognized_text") or level.get("ocr_text") or f"L{level['level_id']}")
        draw.rectangle([x1, y1, x2, y2], outline=(59, 130, 246), width=2)
        draw_text_box(draw, (x1, max(0, y1 - 18)), f"L{level['level_id']} {text}", font)
    for candidate in candidates:
        candidate_id = int(candidate["candidate_id"])
        x1, y1, x2, y2 = [float(v) for v in candidate["bbox_in_full"]]
        draw.rectangle([x1, y1, x2, y2], outline=(34, 197, 94), width=3)
        assignment = candidate.get("section_symbol_assignment") if isinstance(candidate.get("section_symbol_assignment"), dict) else {}
        section_text = assignment.get("selected_symbol_text") or "NO_SEC"
        label = f"C{candidate_id} S={section_text}"
        if candidate_id in matched_candidate_to_level:
            level = level_by_id[matched_candidate_to_level[candidate_id]]
            cx, cy = candidate["center_in_full"]
            lx, ly = level["center_in_full"]
            draw.line([cx, cy, lx, ly], fill=(255, 210, 0), width=2)
            label = f"{label} -> L{level['level_id']}"
        draw_text_box(draw, (x1, max(0, y1 - 20)), label, font)
    vis.save(output_path)


def process_one(
    detector_summary_path: Path,
    ocr_root: Path,
    detector2_root: Path,
    output_root: Path,
    image_root: Optional[Path],
    ocr_runner: OCRRunner,
    threshold: int,
) -> Dict[str, Any]:
    detector_summary = load_json(detector_summary_path)
    image_path = locate_original_image(detector_summary, image_root)
    image = Image.open(image_path).convert("RGB")
    stem = Path(str(detector_summary.get("image_name") or image_path.name)).stem
    ocr_summary_path = find_project_json(ocr_root, stem, "_ocr_summary.json")
    detector2_summary_path = find_project_json(detector2_root, stem, "_detector2_summary.json")
    ocr_summary = load_json(ocr_summary_path)
    detector2_summary = load_json(detector2_summary_path)

    candidates, levels = collect_detector2_items(detector2_summary, detector_summary)
    for level in levels:
        ocr_result = recognize_level_value(image, level["bbox_in_full"], ocr_runner, threshold)
        level["level_id"] = int(level["level_id"])
        level["recognized_value"] = ocr_result.get("recognized_value")
        recognized_value = ocr_result.get("recognized_value")
        level["recognized_text"] = str(recognized_value) if recognized_value is not None else ""
        level["ocr_text"] = ocr_result.get("ocr_text", "")
        level["ocr_score"] = float(ocr_result.get("ocr_score", 0.0) or 0.0)
        level["ocr_variant"] = ocr_result.get("ocr_variant", "")
        level["ocr_debug"] = ocr_result

    correct_level_decimal_omissions(levels)

    section_symbols = list(ocr_summary.get("section_symbols", []) or [])
    for candidate in candidates:
        assignment = assign_section_symbol_by_x(float(candidate["center_in_full"][0]), section_symbols)
        candidate["section_symbol_assignment"] = assignment

    matched_candidate_to_level, candidate_pairs = greedy_match(candidates, levels)
    level_by_id = {int(item["level_id"]): item for item in levels}
    used_level_ids = set(matched_candidate_to_level.values())

    out_dir = output_root / stem
    json_dir = out_dir / "json"
    vis_dir = out_dir / "visualizations"
    ensure_dir(json_dir)
    ensure_dir(vis_dir)

    level_ocr_results = [
        {
            "level_id": int(level["level_id"]),
            "bbox_in_full": [round(float(v), 2) for v in level["bbox_in_full"]],
            "center_in_full": [round(float(v), 2) for v in level["center_in_full"]],
            "score": round(float(level.get("score", 0.0)), 6),
            "recognized_value": level.get("recognized_value"),
            "ocr_text": level.get("ocr_text", ""),
            "ocr_score": round(float(level.get("ocr_score", 0.0)), 6),
            "ocr_variant": level.get("ocr_variant", ""),
            "ocr_debug": level.get("ocr_debug", {}),
            "context_correction": level.get("context_correction"),
        }
        for level in levels
    ]

    results: List[Dict[str, Any]] = []
    candidate_values: List[Dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = int(candidate["candidate_id"])
        item = {
            "candidate_id": candidate_id,
            "candidate_bbox": [round(float(v), 2) for v in candidate["bbox_in_full"]],
            "candidate_center": [round(float(v), 2) for v in candidate["center_in_full"]],
            "candidate_score": round(float(candidate.get("score", 0.0)), 6),
            "match_status": "level_missed",
            "matched_level_id": None,
            "matched_level_bbox": None,
            "matched_level_center": None,
            "matched_level_text": "level_missed",
            "matched_level_value": None,
            "matched_level_score": None,
            "matched_level_ocr_text": "",
            "matched_level_ocr_score": 0.0,
            "matched_level_ocr_variant": "",
            "matched_level_correction": None,
            "section_symbol_assignment": candidate.get("section_symbol_assignment"),
            "source_pier_id": candidate.get("source_pier_id"),
            "source_crop": candidate.get("source_crop"),
        }
        if candidate_id in matched_candidate_to_level:
            level = level_by_id[matched_candidate_to_level[candidate_id]]
            value = level.get("recognized_value")
            item.update(
                {
                    "match_status": "matched",
                    "matched_level_id": int(level["level_id"]),
                    "matched_level_bbox": [round(float(v), 2) for v in level["bbox_in_full"]],
                    "matched_level_center": [round(float(v), 2) for v in level["center_in_full"]],
                    "matched_level_text": str(value) if value is not None else str(level.get("ocr_text") or ""),
                    "matched_level_value": value,
                    "matched_level_score": round(float(level.get("score", 0.0)), 6),
                    "matched_level_ocr_text": level.get("ocr_text", ""),
                    "matched_level_ocr_score": round(float(level.get("ocr_score", 0.0)), 6),
                    "matched_level_ocr_variant": level.get("ocr_variant", ""),
                    "matched_level_correction": level.get("context_correction"),
                }
            )
        results.append(item)
        candidate_values.append(
            {
                "candidate_id": item["candidate_id"],
                "candidate_bbox": item["candidate_bbox"],
                "candidate_score": item["candidate_score"],
                "match_status": item["match_status"],
                "level_value": item["matched_level_text"],
                "level_numeric_value": item["matched_level_value"],
                "level_id": item["matched_level_id"],
                "level_ocr_text": item["matched_level_ocr_text"],
                "level_ocr_score": item["matched_level_ocr_score"],
                "level_correction": item["matched_level_correction"],
                "section_symbol_assignment": item.get("section_symbol_assignment"),
            }
        )

    unused_levels = [
        {
            "level_id": int(level["level_id"]),
            "level_bbox": [round(float(v), 2) for v in level["bbox_in_full"]],
            "level_center": [round(float(v), 2) for v in level["center_in_full"]],
            "level_text": str(level.get("recognized_value") or level.get("ocr_text") or ""),
            "level_score": round(float(level.get("score", 0.0)), 6),
            "ocr_text": level.get("ocr_text", ""),
            "ocr_score": round(float(level.get("ocr_score", 0.0)), 6),
        }
        for level in levels
        if int(level["level_id"]) not in used_level_ids
    ]

    vis_path = vis_dir / f"{stem}_matched_visualization.png"
    draw_match_visualization(image, candidates, levels, matched_candidate_to_level, vis_path)

    level_ocr_path = json_dir / f"{stem}_level_ocr_results.json"
    matched_path = json_dir / f"{stem}_matched_results.json"
    candidate_values_path = json_dir / f"{stem}_candidate_level_values.json"
    save_json(level_ocr_results, level_ocr_path)
    matched_json = {
        "image_name": detector_summary.get("image_name"),
        "full_image_path": str(image_path),
        "detector_summary_json": str(detector_summary_path),
        "ocr_summary_json": str(ocr_summary_path),
        "detector2_summary_json": str(detector2_summary_path),
        "candidate_count": len(candidates),
        "beam_level_count": len(levels),
        "matched_count": len(matched_candidate_to_level),
        "section_symbols": section_symbols,
        "results": results,
        "unused_levels": unused_levels,
        "candidate_pairs_debug": candidate_pairs,
        "outputs": {
            "level_ocr_results_json": str(level_ocr_path),
            "matched_results_json": str(matched_path),
            "candidate_level_values_json": str(candidate_values_path),
            "matched_visualization": str(vis_path),
        },
        "match_rule": {
            "assignment": "same_pier_level_bottom_to_beam_top_euclidean_one_to_one",
            "section_symbol_assignment": "directed_by_section_symbol_orientation",
        },
    }
    save_json(matched_json, matched_path)
    save_json({"candidates": candidate_values}, candidate_values_path)
    print(
        f"[OK] {stem}: candidate={len(candidates)} level={len(levels)} "
        f"matched={len(matched_candidate_to_level)} -> {out_dir}"
    )
    return matched_json
