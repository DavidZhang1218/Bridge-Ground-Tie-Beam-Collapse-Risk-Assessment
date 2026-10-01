from __future__ import annotations


import json


import os


import re


import subprocess


import tempfile


from pathlib import Path


from typing import Any, Dict, List, Optional, Sequence, Tuple


import numpy as np


from PIL import Image, ImageDraw, ImageFont


os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


try:
    import cv2
except Exception:
    cv2 = None


try:
    import torch
except Exception:
    torch = None


THIS_DIR = Path(__file__).resolve().parent


DEFAULT_OCR_MODEL = "PP-OCRv5_server_rec"


MIN_OCR_SCORE = 0.50


MIN_ELEVATION_VALUE = -1000.0


MAX_ELEVATION_VALUE = 10000.0


PARTIAL_GROUND_ROW_SCORE = 0.65


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
]


GROUND_ROW_KEYWORDS = ("\u5730\u9762\u9ad8\u7a0b", "\u5730\u9762\u6807\u9ad8", "\u539f\u5730\u9762\u9ad8\u7a0b", "\u81ea\u7136\u5730\u9762\u9ad8\u7a0b", "\u73b0\u72b6\u5730\u9762\u9ad8\u7a0b")


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


def list_match_summaries(match_results: Path) -> List[Path]:
    direct = list(match_results.glob("json/*_matched_results.json"))
    nested = list(match_results.glob("*/json/*_matched_results.json"))
    return sorted(direct + nested)


def locate_original_image(matched: Dict[str, Any], image_root: Optional[Path]) -> Path:
    image_path = Path(str(matched.get("full_image_path") or ""))
    if image_path.exists():
        return image_path
    image_name = str(matched.get("image_name") or "")
    if image_root is not None and image_name:
        candidate = image_root / image_name
        if candidate.exists():
            return candidate.resolve()
        stem = Path(image_name).stem
        for ext in SUPPORTED_IMAGE_EXTS:
            candidate = image_root / f"{stem}{ext}"
            if candidate.exists():
                return candidate.resolve()
    raise FileNotFoundError(f"Original image not found for {matched.get('image_name')}")


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

    def predict_one(self, img: Image.Image, numeric: bool = False) -> Dict[str, Any]:
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
        return self._predict_tesseract(img, numeric=numeric)

    def _predict_tesseract(self, img: Image.Image, numeric: bool = False) -> Dict[str, Any]:
        whitelist = "0123456789.+-OolI" if numeric else ""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            img.convert("RGB").save(tmp_path)
            cmd = ["tesseract", str(tmp_path), "stdout", "-l", "eng", "--psm", "7"]
            if whitelist:
                cmd.extend(["-c", f"tessedit_char_whitelist={whitelist}"])
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


def recognize_text_variants(
    crop: Image.Image,
    ocr: OCRRunner,
    threshold: int,
    numeric: bool,
    rotations: Sequence[int] = (0,),
) -> Dict[str, Any]:
    variants: List[Tuple[str, Image.Image]] = []
    for rotation in rotations:
        normalized_rotation = int(rotation) % 360
        rotated = crop if normalized_rotation == 0 else crop.rotate(normalized_rotation, expand=True)
        prefix = "" if normalized_rotation == 0 else f"rot{normalized_rotation}_"
        variants.extend(
            [
                (f"{prefix}orig_up", upscale_image(rotated)),
                (f"{prefix}binary_up", upscale_image(threshold_image(rotated, threshold))),
            ]
        )
    candidates: List[Dict[str, Any]] = []
    for name, img in variants:
        pred = ocr.predict_one(img, numeric=numeric)
        value, numeric_text = parse_numeric_text(pred.get("text", "")) if numeric else (None, "")
        raw_score = float(pred.get("score", 0.0) or 0.0)
        if numeric and raw_score < MIN_OCR_SCORE:
            value = None
        score = raw_score
        if numeric:
            score += 0.18 if value is not None else -0.10
        candidates.append(
            {
                "variant": name,
                "backend": pred.get("backend"),
                "raw_text": pred.get("text", ""),
                "raw_score": raw_score,
                "numeric_value": value,
                "numeric_text": numeric_text,
                "final_score": round(score, 6),
            }
        )
    candidates.sort(
        key=lambda item: (item.get("numeric_value") is not None, item["final_score"])
        if numeric
        else (True, item["final_score"]),
        reverse=True,
    )
    best = candidates[0] if candidates else {}
    return {"best": best, "candidates": candidates}


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


def box_center(box: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def merge_positions(values: List[int], tolerance: int = 8) -> List[int]:
    if not values:
        return []
    values = sorted(values)
    groups: List[List[int]] = [[values[0]]]
    for value in values[1:]:
        if abs(value - groups[-1][-1]) <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [int(round(sum(group) / len(group))) for group in groups]


def detect_table_grid(image: Image.Image, threshold: int) -> Dict[str, Any]:
    width, height = image.size
    table_bbox = [0, int(height * 0.58), width, int(height * 0.98)]
    if cv2 is None:
        return {
            "table_bbox": table_bbox,
            "rows": [],
            "columns": [],
            "vertical_line_segments": [],
            "status": "cv2_unavailable",
        }

    crop = image.crop(tuple(table_bbox)).convert("L")
    gray = np.array(crop)
    _, mask = cv2.threshold(gray, max(0, min(255, int(threshold))), 255, cv2.THRESH_BINARY_INV)
    h, w = mask.shape[:2]
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 18), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 8)))
    h_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, h_kernel)
    v_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, v_kernel)

    y_positions: List[int] = []
    table_lines: List[Tuple[int, int]] = []
    contours_info = cv2.findContours(h_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw >= w * 0.18:
            y_positions.append(table_bbox[1] + y + bh // 2)
        full_y = table_bbox[1] + y
        if w * 0.30 <= bw <= w * 0.80 and height * 0.69 <= full_y <= height * 0.92 and bh <= 12:
            table_lines.append((x, bw))

    if len(table_lines) >= 3:
        table_left_x = int(round(float(np.median([line[0] for line in table_lines]))))
        label_width = max(80, int(round(float(np.median([line[1] for line in table_lines])) * 0.15)))
    else:
        table_left_x = table_bbox[0]
        label_width = max(80, int(width * 0.25))

    x_positions: List[int] = []
    vertical_line_segments: List[Dict[str, int]] = []
    contours_info = cv2.findContours(v_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bh >= h * 0.12:
            line_x = table_bbox[0] + x + bw // 2
            x_positions.append(line_x)
            vertical_line_segments.append({
                "x": line_x,
                "y1": table_bbox[1] + y,
                "y2": table_bbox[1] + y + bh,
            })

    y_positions = merge_positions([table_bbox[1], table_bbox[3]] + y_positions)
    x_positions = merge_positions([table_bbox[0], table_bbox[2]] + x_positions)
    rows = []
    for idx, (y1, y2) in enumerate(zip(y_positions, y_positions[1:]), start=1):
        if y2 - y1 >= 14:
            rows.append({
                "row_id": idx,
                "bbox": [table_bbox[0], y1, table_bbox[2], y2],
                "center_y": (y1 + y2) / 2.0,
                "label_start_x": table_left_x,
                "label_end_x": min(width, table_left_x + label_width),
            })
    columns = []
    for idx, (x1, x2) in enumerate(zip(x_positions, x_positions[1:]), start=1):
        if x2 - x1 >= 12:
            columns.append({"column_id": idx, "bbox": [x1, table_bbox[1], x2, table_bbox[3]], "center_x": (x1 + x2) / 2.0})
    return {
        "table_bbox": table_bbox,
        "rows": rows,
        "columns": columns,
        "status": "ok" if rows else "no_rows_detected",
        "horizontal_line_y": y_positions,
        "vertical_line_x": x_positions,
        "vertical_line_segments": vertical_line_segments,
        "table_left_x": table_left_x,
    }


def row_left_label_crop(image: Image.Image, row: Dict[str, Any]) -> Image.Image:
    x1, y1, x2, y2 = row["bbox"]
    label_x1 = int(row.get("label_start_x", x1))
    label_x2 = int(row.get("label_end_x", x1 + max(80, int((x2 - x1) * 0.25))))
    return image.crop((label_x1, y1, min(x2, label_x2), y2))


def detect_ground_row(
    image: Image.Image,
    grid: Dict[str, Any],
    ocr: OCRRunner,
    threshold: int,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    rows = list(grid.get("rows", []) or [])
    if not rows:
        return None, {"mode": "manual_required", "reason": "no_rows_detected", "row_labels": []}

    label_debug: List[Dict[str, Any]] = []
    exact_matches: List[Tuple[float, Dict[str, Any]]] = []
    partial_matches: List[Tuple[float, Dict[str, Any]]] = []
    for row in rows:
        crop = row_left_label_crop(image, row)
        rec = recognize_text_variants(crop, ocr, threshold, numeric=False)
        best = rec.get("best") or {}
        text = str(best.get("raw_text") or "")
        normalized_text = re.sub(r"\s+", "", text)
        score = float(best.get("raw_score", 0.0) or 0.0)
        row["label_text"] = text
        row["label_score"] = score
        match_type = "none"
        if score >= MIN_OCR_SCORE and any(keyword in normalized_text for keyword in GROUND_ROW_KEYWORDS):
            exact_matches.append((score, row))
            match_type = "exact_keyword"
        elif score >= PARTIAL_GROUND_ROW_SCORE and "\u5730\u9762" in normalized_text:
            partial_matches.append((score, row))
            match_type = "partial_keyword"
        label_debug.append(
            {
                "row_id": int(row["row_id"]),
                "text": text,
                "score": round(score, 6),
                "variant": best.get("variant"),
                "match_type": match_type,
            }
        )

    if exact_matches:
        _score, selected = max(exact_matches, key=lambda item: item[0])
        return selected, {"mode": "automatic_exact", "row_labels": label_debug}
    if partial_matches:
        _score, selected = max(partial_matches, key=lambda item: item[0])
        return selected, {"mode": "automatic_partial", "row_labels": label_debug}
    return None, {"mode": "manual_required", "reason": "ground_row_not_recognized", "row_labels": label_debug}


def find_row_by_id(rows: List[Dict[str, Any]], row_id: int) -> Optional[Dict[str, Any]]:
    for row in rows:
        try:
            if int(row.get("row_id")) == int(row_id):
                return row
        except Exception:
            continue
    return None


def render_row_selection_image(
    image: Image.Image,
    rows: List[Dict[str, Any]],
    selected_row_id: Optional[int] = None,
) -> Image.Image:
    rendered = image.convert("RGB").copy()
    draw = ImageDraw.Draw(rendered)
    font = load_font(18)
    for row in rows:
        bbox = row.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        row_id = int(row.get("row_id", 0) or 0)
        color = (255, 170, 0) if row_id == selected_row_id else (20, 120, 255)
        x1, y1, x2, y2 = [float(v) for v in bbox]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        draw_text_box(draw, (x1 + 6, y1 + 4), f"R{row_id}", font)
    return rendered


def save_row_selection_preview(
    image: Image.Image,
    rows: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    ensure_dir(output_path.parent)
    render_row_selection_image(image, rows).save(output_path)
    return output_path


def ground_value_bbox_for_x(
    row: Dict[str, Any],
    x: float,
    image_size: Tuple[int, int],
) -> List[int]:
    width, height = image_size
    _x1, y1, _x2, y2 = [float(v) for v in row["bbox"]]
    row_height = max(1.0, y2 - y1)
    return clip_box(
        [float(x) - 0.38 * row_height, y1, float(x) - 0.07 * row_height, y2],
        width,
        height,
    )


def ground_lookup_x(row: Dict[str, Any], grid: Dict[str, Any], anchor_x: float) -> float:
    """Use the nearby table divider as the ground-value reference."""
    row_y1, row_y2 = float(row["bbox"][1]), float(row["bbox"][3])
    row_height = max(1.0, row_y2 - row_y1)
    segments = grid.get("vertical_line_segments")
    if segments is None:
        lines = grid.get("vertical_line_x") or []
    else:
        lines = [
            segment["x"] for segment in segments
            if max(0.0, min(row_y2, float(segment["y2"])) - max(row_y1, float(segment["y1"])))
            >= 0.45 * row_height
        ]
    if not lines:
        return anchor_x
    nearest = min(lines, key=lambda line: abs(float(line) - anchor_x))
    return float(nearest) if abs(float(nearest) - anchor_x) <= 0.45 * row_height else anchor_x


def parse_candidate_value(item: Dict[str, Any]) -> Tuple[Optional[float], str]:
    value = item.get("matched_level_value")
    if value is not None:
        try:
            numeric_value = float(value)
            if MIN_ELEVATION_VALUE <= numeric_value <= MAX_ELEVATION_VALUE:
                return numeric_value, str(value)
        except Exception:
            pass
    return parse_numeric_text(item.get("matched_level_text") or item.get("matched_level_ocr_text"))


def group_candidates(candidates: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    by_pier: Dict[Any, List[Dict[str, Any]]] = {}
    for item in candidates:
        pier_id = item.get("source_pier_id")
        if pier_id is not None:
            by_pier.setdefault(pier_id, []).append(item)
    if by_pier:
        return [sorted(items, key=lambda x: x["candidate_center"][1]) for _key, items in sorted(by_pier.items(), key=lambda pair: str(pair[0]))]

    sorted_items = sorted(candidates, key=lambda item: float(item["candidate_center"][0]))
    if not sorted_items:
        return []
    groups: List[List[Dict[str, Any]]] = [[sorted_items[0]]]
    for item in sorted_items[1:]:
        prev = groups[-1][-1]
        if float(item["candidate_center"][0]) - float(prev["candidate_center"][0]) <= 180.0:
            groups[-1].append(item)
        else:
            groups.append([item])
    return groups


def section_symbol_summary(assignment: Any) -> Dict[str, Any]:
    data = assignment if isinstance(assignment, dict) else {}
    return {
        "id": data.get("selected_symbol_id"),
        "value": data.get("selected_symbol_value"),
        "text": data.get("selected_symbol_text"),
        "label": data.get("selected_symbol_text") or data.get("selected_symbol_value") or "NO_SEC",
        "orientation": data.get("selected_symbol_orientation"),
        "bbox": data.get("selected_symbol_bbox"),
        "assignment_status": data.get("status"),
        "assignment_rule": data.get("assignment_rule"),
        "left_symbol_id": data.get("left_symbol_id"),
        "left_symbol_orientation": data.get("left_symbol_orientation"),
        "right_symbol_id": data.get("right_symbol_id"),
        "right_symbol_orientation": data.get("right_symbol_orientation"),
    }


def draw_text_box(draw: ImageDraw.ImageDraw, xy: Tuple[float, float], text: str, font: ImageFont.ImageFont) -> None:
    if not text:
        return
    x, y = xy
    try:
        if "\n" in text:
            bbox = draw.multiline_textbbox((x, y), text, font=font, spacing=2)
        else:
            bbox = draw.textbbox((x, y), text, font=font)
        draw.rectangle(bbox, fill=(255, 255, 180))
        if "\n" in text:
            draw.multiline_text((x, y), text, fill=(0, 0, 0), font=font, spacing=2)
        else:
            draw.text((x, y), text, fill=(0, 0, 0), font=font)
    except Exception:
        draw.text((x, y), text, fill=(0, 0, 0), font=font)


def draw_final_visualization(
    image: Image.Image,
    ground_tie_beams: List[Dict[str, Any]],
    grid: Dict[str, Any],
    ground_row: Optional[Dict[str, Any]],
    output_path: Path,
) -> None:
    ensure_dir(output_path.parent)
    vis = image.convert("RGB").copy()
    draw = ImageDraw.Draw(vis)
    font = load_font(28)
    font_small = load_font(18)
    table_bbox = grid.get("table_bbox")
    if isinstance(table_bbox, list) and len(table_bbox) == 4:
        draw.rectangle([float(v) for v in table_bbox], outline=(160, 160, 160), width=2)
    if ground_row is not None:
        draw.rectangle([float(v) for v in ground_row["bbox"]], outline=(59, 130, 246), width=3)
        draw_text_box(draw, (ground_row["bbox"][0] + 4, max(0, ground_row["bbox"][1] - 22)), "ground row", font_small)
    for item in ground_tie_beams:
        box = item.get("candidate_bbox")
        if not isinstance(box, list) or len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=(34, 197, 94), width=4)
        section = item.get("section_symbol", {}) or {}
        beam_level = item.get("beam_level", {}) or {}
        ground_level = item.get("ground_level", {}) or {}
        section_label = section.get("label") or section.get("text") or section.get("value") or "NO_SEC"
        beam_value = beam_level.get("value")
        ground_value = ground_level.get("value")
        label = (
            f"GTB{item['ground_tie_beam_id']} S={section_label}\n"
            f"B={beam_value} G={ground_value}"
        )
        text_bbox = draw.multiline_textbbox((0, 0), label, font=font, spacing=2)
        text_width = max(0, text_bbox[2] - text_bbox[0])
        text_height = max(0, text_bbox[3] - text_bbox[1])
        label_x = min(max(0.0, x1), max(0.0, float(vis.width - text_width - 4)))
        label_y = max(0.0, y1 - text_height - 8)
        draw_text_box(draw, (label_x, label_y), label, font)
        cell_bbox = ground_level.get("cell_bbox")
        if isinstance(cell_bbox, list) and len(cell_bbox) == 4:
            gx1, gy1, gx2, gy2 = [float(v) for v in cell_bbox]
            draw.rectangle([gx1, gy1, gx2, gy2], outline=(255, 170, 0), width=3)
            ground_text_bbox = draw.textbbox((0, 0), f"G={ground_value}", font=font_small)
            ground_text_height = max(0, ground_text_bbox[3] - ground_text_bbox[1])
            draw_text_box(draw, (gx1 + 2, max(0.0, gy1 - ground_text_height - 4)), f"G={ground_value}", font_small)
    vis.save(output_path)


def build_ground_tie_beams(
    image: Image.Image,
    matched: Dict[str, Any],
    ground_row: Optional[Dict[str, Any]],
    grid: Dict[str, Any],
    ocr: OCRRunner,
    threshold: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    candidates = [item for item in matched.get("results", []) or [] if isinstance(item, dict)]
    groups = group_candidates(candidates)
    candidate_beams: List[Dict[str, Any]] = []
    group_results: List[Dict[str, Any]] = []

    for group_idx, group in enumerate(groups, start=1):
        center_values = [
            float(item["candidate_center"][0])
            for item in group
            if isinstance(item.get("candidate_center"), list) and len(item["candidate_center"]) == 2
        ]
        anchor_x = sum(center_values) / len(center_values) if center_values else 0.0
        ground_cell_bbox: Optional[List[int]] = None
        ground_best: Dict[str, Any] = {}
        if ground_row is not None:
            lookup_x = ground_lookup_x(ground_row, grid, anchor_x)
            ground_cell_bbox = ground_value_bbox_for_x(ground_row, lookup_x, image.size)
            ground_recognition = recognize_text_variants(
                image.crop(tuple(ground_cell_bbox)),
                ocr,
                threshold,
                numeric=True,
                rotations=(0, 270),
            )
            ground_best = ground_recognition.get("best") or {}
            if ground_best.get("numeric_value") is None and lookup_x != anchor_x:
                anchor_bbox = ground_value_bbox_for_x(ground_row, anchor_x, image.size)
                anchor_recognition = recognize_text_variants(
                    image.crop(tuple(anchor_bbox)),
                    ocr,
                    threshold,
                    numeric=True,
                    rotations=(0, 270),
                )
                anchor_best = anchor_recognition.get("best") or {}
                if anchor_best.get("numeric_value") is not None:
                    ground_cell_bbox = anchor_bbox
                    ground_best = anchor_best
        ground_value = ground_best.get("numeric_value")
        ground_text = str(ground_best.get("numeric_text") or ground_best.get("raw_text") or "")

        for candidate in sorted(group, key=lambda item: int(item.get("candidate_id", 0) or 0)):
            beam_value, beam_text = parse_candidate_value(candidate)
            if candidate.get("match_status") != "matched":
                beam_value = None
            delta = round(float(beam_value) - float(ground_value), 4) if beam_value is not None and ground_value is not None else None
            status = "ready" if delta is not None else ("ground_level_missing" if ground_value is None else "beam_level_missing")
            candidate_beams.append(
                {
                    "ground_tie_beam_id": len(candidate_beams) + 1,
                    "pier_group_id": group_idx,
                    "source_pier_id": candidate.get("source_pier_id"),
                    "candidate_id": int(candidate.get("candidate_id", 0) or 0),
                    "section_symbol": section_symbol_summary(candidate.get("section_symbol_assignment")),
                    "beam_level": {
                        "value": beam_value,
                        "text": beam_text,
                        "correction": candidate.get("matched_level_correction"),
                    },
                    "ground_level": {
                        "value": ground_value,
                        "text": ground_text,
                        "cell_bbox": ground_cell_bbox,
                        "ocr_score": round(float(ground_best.get("raw_score", 0.0) or 0.0), 6),
                        "ocr_variant": ground_best.get("variant"),
                    },
                    "beam_minus_ground": delta,
                    "candidate_bbox": candidate.get("candidate_bbox"),
                    "candidate_center": candidate.get("candidate_center"),
                    "candidate_score": candidate.get("candidate_score"),
                    "match_status": candidate.get("match_status"),
                    "status": status,
                }
            )

        group_results.append(
            {
                "pier_group_id": group_idx,
                "candidate_ids": [int(item.get("candidate_id", 0) or 0) for item in group],
                "anchor_x": round(anchor_x, 2),
                "ground_level": {
                    "value": ground_value,
                    "text": ground_text,
                    "cell_bbox": ground_cell_bbox,
                    "ocr_score": round(float(ground_best.get("raw_score", 0.0) or 0.0), 6),
                    "ocr_variant": ground_best.get("variant"),
                },
                "status": "ready" if ground_value is not None else "ground_level_missing",
            }
        )
    return candidate_beams, group_results


def process_one(
    matched_path: Path,
    output_root: Path,
    image_root: Optional[Path],
    ocr: OCRRunner,
    threshold: int = 165,
) -> Dict[str, Any]:
    matched = load_json(matched_path)
    image_path = locate_original_image(matched, image_root)
    image = Image.open(image_path).convert("RGB")
    stem = Path(str(matched.get("image_name") or image_path.name)).stem
    out_dir = output_root / stem
    ensure_dir(out_dir / "json")
    ensure_dir(out_dir / "visualizations")

    grid = detect_table_grid(image, threshold)
    ground_row, row_debug = detect_ground_row(image, grid, ocr, threshold)
    rows = list(grid.get("rows", []) or [])
    selection_preview_path = out_dir / "visualizations" / f"{stem}_ground_row_selection.png"
    if ground_row is None and rows:
        save_row_selection_preview(image, rows, selection_preview_path)

    candidate_beams, pier_group_results = build_ground_tie_beams(
        image=image,
        matched=matched,
        ground_row=ground_row,
        grid=grid,
        ocr=ocr,
        threshold=threshold,
    )
    vis_path = out_dir / "visualizations" / f"{stem}_candidate_beams_visualization.png"
    draw_final_visualization(image, candidate_beams, grid, ground_row, vis_path)
    summary_path = out_dir / "json" / f"{stem}_candidate_beams_summary.json"
    summary = {
        "status": "completed" if ground_row is not None else "partial",
        "reason": None if ground_row is not None else "ground_row_not_recognized",
        "image_name": matched.get("image_name"),
        "full_image_path": str(image_path),
        "matched_results_json": str(matched_path),
        "candidate_count": len(candidate_beams),
        "candidate_beams": candidate_beams,
        "pier_group_results": pier_group_results,
        "detected_ground_row": {
            "row_id": int(ground_row["row_id"]) if ground_row is not None else None,
            "bbox": ground_row.get("bbox") if ground_row is not None else None,
            "label_text": ground_row.get("label_text") if ground_row is not None else None,
        },
        "table_debug": {"grid": grid, "row_detection": row_debug},
        "outputs": {
            "summary_json": str(summary_path),
            "visualization": str(vis_path),
            "ground_row_selection_preview": str(selection_preview_path) if selection_preview_path.exists() else None,
        },
    }
    save_json(summary, summary_path)
    return summary
