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
    import torch
except Exception:
    torch = None


try:
    import cv2
except Exception:
    cv2 = None


THIS_DIR = Path(__file__).resolve().parent


DEFAULT_OCR_MODEL = "PP-OCRv5_server_rec"


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


def save_json(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def list_summary_files(detector_results: Path) -> List[Path]:
    direct = list(detector_results.glob("json/*_pier_section_summary.json"))
    nested = list(detector_results.glob("*/json/*_pier_section_summary.json"))
    return sorted(direct + nested)


def find_summary_for_image(image_path: Path, detector_results: Path) -> Optional[Path]:
    image_stem = image_path.stem
    summaries = list_summary_files(detector_results)
    for summary_path in summaries:
        try:
            summary = load_json(summary_path)
        except Exception:
            continue
        summary_image_name = str(summary.get("image_name") or "")
        summary_image_path = str(summary.get("image_path") or "")
        if Path(summary_image_name).stem == image_stem or Path(summary_image_path).stem == image_stem:
            return summary_path
    for summary_path in summaries:
        if summary_path.name.startswith(image_stem):
            return summary_path
    return None


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


def padded_box(
    box: Sequence[float],
    width: int,
    height: int,
    pad_x_ratio: float = 0.12,
    pad_y_ratio: float = 0.18,
    min_pad: int = 4,
) -> List[int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    px = max(float(min_pad), bw * pad_x_ratio)
    py = max(float(min_pad), bh * pad_y_ratio)
    return clip_box([x1 - px, y1 - py, x2 + px, y2 + py], width, height)


def threshold_image(img: Image.Image, threshold: int) -> Image.Image:
    threshold = max(0, min(255, int(threshold)))
    gray = img.convert("L")
    bw = gray.point(lambda p: 255 if p >= threshold else 0)
    return bw.convert("RGB")


def upscale_image(img: Image.Image, scale: float = 2.5, max_side: int = 960) -> Image.Image:
    side = max(img.size)
    if side <= 0:
        return img
    scale = min(float(scale), float(max_side) / float(side))
    scale = max(1.0, scale)
    size = (max(1, int(round(img.width * scale))), max(1, int(round(img.height * scale))))
    if size == img.size:
        return img
    return img.resize(size, Image.Resampling.LANCZOS)


def ocr_variants(img: Image.Image, threshold: int) -> List[Tuple[str, Image.Image]]:
    bw = threshold_image(img, threshold)
    return [
        ("orig_up", upscale_image(img)),
        ("binary_up", upscale_image(bw)),
    ]


def roman_to_int(text: str) -> Optional[int]:
    values = {"I": 1, "V": 5, "X": 10}
    cleaned = re.sub(r"[^IVX]", "", str(text or "").upper())
    if not cleaned:
        return None
    total = 0
    prev = 0
    for ch in reversed(cleaned):
        value = values.get(ch, 0)
        if value < prev:
            total -= value
        else:
            total += value
            prev = value
    if total <= 0 or int_to_roman(total) != cleaned:
        return None
    return total


def int_to_roman(value: int) -> str:
    if value <= 0:
        return ""
    parts: List[str] = []
    rest = int(value)
    for number, glyph in ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while rest >= number:
            parts.append(glyph)
            rest -= number
    return "".join(parts)


def normalize_roman_text(text: str) -> str:
    roman_glyphs = {
        "\u2160": "I",
        "\u2161": "II",
        "\u2162": "III",
        "\u2163": "IV",
        "\u2164": "V",
        "\u2165": "VI",
        "\u2166": "VII",
        "\u2167": "VIII",
        "\u2168": "IX",
        "\u2169": "X",
    }
    replacements = {
        "1": "I",
        "I": "I",
        "i": "I",
        "l": "I",
        "|": "I",
        "!": "I",
        "V": "V",
        "v": "V",
        "X": "X",
        "x": "X",
    }
    parts: List[str] = []
    for ch in str(text or "").strip():
        if ch in roman_glyphs:
            parts.append(roman_glyphs[ch])
        elif ch in replacements:
            parts.append(replacements[ch])
    return "".join(parts)


def parse_roman_value(text: str) -> Tuple[Optional[int], Optional[str]]:
    normalized = normalize_roman_text(text)
    match = re.search(r"[IVX]+", normalized)
    if not match:
        return None, None
    roman = match.group(0)
    value = roman_to_int(roman)
    if value is None:
        return None, roman
    return value, int_to_roman(value)


def guess_roman_by_components(img: Image.Image) -> Optional[Dict[str, Any]]:
    if cv2 is None:
        return None
    gray = np.array(img.convert("L"))
    mask = (gray < 180).astype("uint8")
    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    height, width = mask.shape
    components: List[Tuple[int, int, int, int, int]] = []
    min_area = max(20, int(height * width * 0.001))
    for idx in range(1, num_labels):
        x, y, w, h, area = [int(v) for v in stats[idx]]
        if area < min_area:
            continue
        if h < height * 0.30:
            continue
        if w > width * 0.30 and h > height * 0.45:
            continue
        if w > width * 0.60:
            continue
        components.append((x, y, w, h, area))
    if not components:
        return None

    chars: List[str] = []
    for _x, _y, w, h, _area in sorted(components):
        aspect = w / max(1, h)
        if aspect < 0.42:
            chars.append("I")
        elif aspect < 1.0:
            chars.append("V")
        else:
            chars.append("X")
    text = "".join(chars)
    value = roman_to_int(text)
    if value is None:
        return None
    return {
        "variant": "cv_components",
        "backend": "cv2",
        "raw_text": text,
        "normalized_text": text,
        "parsed_text": int_to_roman(value),
        "parsed_value": value,
        "raw_score": 0.92,
        "final_score": 1.02,
        "components": [
            {"x": x, "y": y, "w": w, "h": h, "area": area}
            for x, y, w, h, area in sorted(components)
        ],
    }


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
        self.active_backend = "none"

    def _build_paddle(self) -> Any:
        if self._paddle_model is not None:
            return self._paddle_model
        try:
            from paddleocr import TextRecognition
        except Exception as exc:
            raise RuntimeError(f"PaddleOCR TextRecognition is unavailable: {exc}") from exc
        self._paddle_model = TextRecognition(model_name=self.model_name, device=self.device)
        self.active_backend = "paddle"
        return self._paddle_model

    def predict_one(self, img: Image.Image, charset: str) -> Dict[str, Any]:
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
        return self._predict_tesseract(img, charset)

    def _predict_tesseract(self, img: Image.Image, charset: str) -> Dict[str, Any]:
        whitelist = "IVXivxl1|!"
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
            score = 0.65 if text else 0.0
            return {"text": text, "score": score, "backend": "tesseract"}
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


def recognize_with_variants(
    crop: Image.Image,
    ocr: OCRRunner,
    charset: str,
    threshold: int,
) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    for variant_name, variant in ocr_variants(crop, threshold):
        pred = ocr.predict_one(variant, charset)
        raw_text = str(pred.get("text", "")).strip()
        raw_score = float(pred.get("score", 0.0) or 0.0)
        value, parsed_text = parse_roman_value(raw_text)
        final_score = raw_score + (0.18 if value is not None else -0.15)
        final_score += 0.02 * min(len(parsed_text or ""), 4)
        normalized = normalize_roman_text(raw_text)
        parsed_value = value
        candidates.append(
            {
                "variant": variant_name,
                "backend": pred.get("backend"),
                "raw_text": raw_text,
                "normalized_text": normalized,
                "parsed_text": parsed_text,
                "parsed_value": parsed_value,
                "raw_score": raw_score,
                "final_score": round(float(final_score), 6),
            }
        )
    if charset == "roman":
        cv_guess = guess_roman_by_components(crop)
        if cv_guess is not None:
            candidates.append(cv_guess)
    candidates.sort(key=lambda item: item["final_score"], reverse=True)
    best = candidates[0] if candidates else None
    return {
        "best": best,
        "candidates": candidates,
    }


def _line_components(mask: np.ndarray, axis: str) -> List[Dict[str, Any]]:
    if cv2 is None or mask.size == 0:
        return []
    h, w = mask.shape[:2]
    if axis == "horizontal":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, int(w * 0.12)), 1))
        min_len = max(5, int(w * 0.12))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, int(h * 0.16))))
        min_len = max(5, int(h * 0.16))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours_info = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    lines: List[Dict[str, Any]] = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if axis == "horizontal":
            if bw < min_len or bw < max(5, bh * 2.5):
                continue
            length = bw
        else:
            if bh < min_len or bh < max(5, bw * 2.2):
                continue
            length = bh
        lines.append(
            {
                "bbox": [int(x), int(y), int(x + bw), int(y + bh)],
                "center": [float(x + bw / 2.0), float(y + bh / 2.0)],
                "length": float(length),
            }
        )
    lines.sort(key=lambda item: item["length"], reverse=True)
    return lines


def detect_section_symbol_orientation(
    full_image: Image.Image,
    bbox_in_full: Sequence[float],
    threshold: int,
) -> Dict[str, Any]:
    if cv2 is None:
        return {
            "orientation": "unknown",
            "confidence": 0.0,
            "method": "cv_axis_lines_basic",
            "status": "cv2_unavailable",
        }
    crop_box = padded_box(bbox_in_full, full_image.width, full_image.height, 0.04, 0.04, 3)
    crop = full_image.crop(tuple(crop_box)).convert("L")
    gray = np.array(crop)
    if gray.size == 0:
        return {
            "orientation": "unknown",
            "confidence": 0.0,
            "method": "cv_axis_lines_basic",
            "status": "empty_crop",
            "crop_rect_in_full": crop_box,
        }
    _, mask = cv2.threshold(gray, max(0, min(255, int(threshold))), 255, cv2.THRESH_BINARY_INV)
    if min(mask.shape[:2]) >= 3:
        mask = cv2.medianBlur(mask, 3)

    horizontal_lines = _line_components(mask, "horizontal")
    vertical_lines = _line_components(mask, "vertical")
    best: Optional[Dict[str, Any]] = None
    best_score = -1.0
    h, w = mask.shape[:2]
    endpoint_tol = max(8.0, w * 0.28)
    y_tol = max(5.0, h * 0.18)
    for hline in horizontal_lines[:6]:
        hx1, hy1, hx2, hy2 = hline["bbox"]
        hcx, hcy = hline["center"]
        for vline in vertical_lines[:6]:
            vx1, vy1, vx2, vy2 = vline["bbox"]
            vcx, _vcy = vline["center"]
            if hcy < vy1 - y_tol or hcy > vy2 + y_tol:
                continue
            endpoint_dist = min(abs(vcx - hx1), abs(vcx - hx2))
            if endpoint_dist > endpoint_tol:
                continue
            delta = hcx - vcx
            if abs(delta) < max(2.0, w * 0.03):
                continue
            score = float(hline["length"]) * 1.5 + float(vline["length"]) - endpoint_dist
            if score > best_score:
                best_score = score
                confidence = min(
                    0.98,
                    0.50
                    + min(0.25, float(hline["length"]) / max(1.0, w) * 0.5)
                    + min(0.20, float(vline["length"]) / max(1.0, h) * 0.4),
                )
                best = {
                    "orientation": "right" if delta > 0 else "left",
                    "confidence": round(float(confidence), 6),
                    "horizontal_line_bbox_in_crop": hline["bbox"],
                    "vertical_line_bbox_in_crop": vline["bbox"],
                    "endpoint_distance": round(float(endpoint_dist), 3),
                    "score": round(float(score), 6),
                }

    if best is None:
        return {
            "orientation": "unknown",
            "confidence": 0.0,
            "method": "cv_axis_lines_basic",
            "status": "line_pair_not_found",
            "crop_rect_in_full": crop_box,
            "num_horizontal_lines": len(horizontal_lines),
            "num_vertical_lines": len(vertical_lines),
        }

    cx1, cy1, _cx2, _cy2 = crop_box
    hbox = best["horizontal_line_bbox_in_crop"]
    vbox = best["vertical_line_bbox_in_crop"]
    best.update(
        {
            "method": "cv_axis_lines_basic",
            "status": "ok",
            "crop_rect_in_full": crop_box,
            "horizontal_line_bbox_full": [hbox[0] + cx1, hbox[1] + cy1, hbox[2] + cx1, hbox[3] + cy1],
            "vertical_line_bbox_full": [vbox[0] + cx1, vbox[1] + cy1, vbox[2] + cx1, vbox[3] + cy1],
            "num_horizontal_lines": len(horizontal_lines),
            "num_vertical_lines": len(vertical_lines),
        }
    )
    return best


def infer_roman_side(crop: Image.Image, best: Dict[str, Any]) -> str:
    components = best.get("components") if isinstance(best, dict) else None
    if isinstance(components, list) and components:
        xs: List[float] = []
        for item in components:
            if not isinstance(item, dict):
                continue
            x = item.get("x")
            w = item.get("w")
            if x is not None and w is not None:
                xs.extend([float(x), float(x) + float(w)])
        if xs:
            center_x = (min(xs) + max(xs)) / 2.0
            return "left" if center_x < crop.width / 2.0 else "right"

    gray = np.array(crop.convert("L"))
    ys, xs_np = np.where(gray < 180)
    if len(xs_np) == 0 or len(ys) == 0:
        return "unknown"
    center_x = (float(xs_np.min()) + float(xs_np.max())) / 2.0
    return "left" if center_x < crop.width / 2.0 else "right"


def orientation_mark(orientation: Any) -> str:
    value = str(orientation or "").strip().lower()
    if value == "left":
        return "L"
    if value == "right":
        return "R"
    return "?"


def locate_original_image(summary: Dict[str, Any], image_root: Optional[Path]) -> Path:
    image_path = Path(str(summary.get("image_path") or ""))
    if image_path.exists():
        return image_path
    image_name = str(summary.get("image_name") or "")
    if image_root is not None and image_name:
        candidate = image_root / image_name
        if candidate.exists():
            return candidate.resolve()
        stem = Path(image_name).stem
        for ext in SUPPORTED_IMAGE_EXTS:
            candidate = image_root / f"{stem}{ext}"
            if candidate.exists():
                return candidate.resolve()
    raise FileNotFoundError(f"Original image not found for {summary.get('image_name')}")


def load_pier_crop_image(pier_item: Dict[str, Any], full_image: Image.Image) -> Optional[Image.Image]:
    src = Path(str(pier_item.get("path") or ""))
    if src.exists():
        return Image.open(src).convert("RGB")
    rect = pier_item.get("rect_in_full")
    if isinstance(rect, list) and len(rect) == 4:
        crop_box = clip_box(rect, full_image.width, full_image.height)
        return full_image.crop(tuple(crop_box)).convert("RGB")
    return None


def copy_detector_pier_crops(summary: Dict[str, Any], out_dir: Path, full_image: Image.Image) -> Dict[int, Path]:
    copied: Dict[int, Path] = {}
    pier_copy_dir = out_dir / "crops" / "piers"
    ensure_dir(pier_copy_dir)
    stem = Path(str(summary.get("image_name") or "image")).stem
    for item in summary.get("pier_crops", []) or []:
        pier_id = int(item.get("pier_id") or len(copied) + 1)
        src = Path(str(item.get("path") or ""))
        pier_img = load_pier_crop_image(item, full_image)
        if pier_img is None:
            continue
        dst_name = src.name if src.name else f"{stem}_pier_{pier_id:03d}.png"
        dst = pier_copy_dir / dst_name
        pier_img.save(dst)
        copied[pier_id] = dst
    return copied


def crop_section_symbols(
    image: Image.Image,
    summary: Dict[str, Any],
    out_dir: Path,
    ocr: OCRRunner,
    threshold: int,
) -> List[Dict[str, Any]]:
    crop_dir = out_dir / "crops" / "section_symbols"
    ensure_dir(crop_dir)
    width, height = image.size
    results: List[Dict[str, Any]] = []
    symbols = sorted(summary.get("section_symbols", []) or [], key=lambda item: (item.get("center_in_full") or [0, 0])[0])
    for idx, det in enumerate(symbols, start=1):
        bbox = det.get("bbox_in_full")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        crop_box = padded_box(bbox, width, height, pad_x_ratio=0.08, pad_y_ratio=0.12, min_pad=6)
        crop = image.crop(tuple(crop_box))
        crop_name = f"{Path(summary['image_name']).stem}_section_{idx:03d}.png"
        crop_path = crop_dir / crop_name
        crop.save(crop_path)
        rec = recognize_with_variants(crop, ocr, "roman", threshold)
        best = rec.get("best") or {}
        orientation_debug = detect_section_symbol_orientation(image, bbox, threshold)
        orientation = str(orientation_debug.get("orientation") or "unknown")
        results.append(
            {
                "id": int(det.get("section_symbol_id") or idx),
                "section_symbol_id": int(det.get("section_symbol_id") or idx),
                "crop_path": str(crop_path),
                "crop_rect_in_full": crop_box,
                "bbox_in_full": bbox,
                "center_in_full": det.get("center_in_full"),
                "det_score": det.get("score"),
                "symbol_text": best.get("parsed_text"),
                "symbol_value": best.get("parsed_value"),
                "roman_side": infer_roman_side(crop, best),
                "orientation": orientation,
                "orientation_confidence": float(orientation_debug.get("confidence", 0.0) or 0.0),
                "orientation_debug": orientation_debug,
                "ocr_text": best.get("raw_text"),
                "ocr_score": best.get("raw_score"),
                "ocr_variant": best.get("variant"),
                "ocr_candidates": rec.get("candidates", []),
            }
        )
    return results


def draw_full_visualization(
    image: Image.Image,
    section_results: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    ensure_dir(output_path.parent)
    vis = image.convert("RGB").copy()
    draw = ImageDraw.Draw(vis)
    font = load_font(18)
    for item in section_results:
        box = item.get("bbox_in_full")
        if not box:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        label = str(item.get("symbol_text") or item.get("ocr_text") or "")
        orient = orientation_mark(item.get("orientation"))
        draw.rectangle([x1, y1, x2, y2], outline=(0, 92, 255), width=3)
        if label:
            draw.text((x1, max(0, y1 - 22)), f"S={label} {orient}", fill=(0, 92, 255), font=font)
    vis.save(output_path)


def process_one(
    summary_path: Path,
    output_root: Path,
    image_root: Optional[Path],
    ocr: OCRRunner,
    threshold: int = 125,
) -> Dict[str, Any]:
    summary = load_json(summary_path)
    image_path = locate_original_image(summary, image_root)
    image = Image.open(image_path).convert("RGB")
    stem = image_path.stem
    out_dir = output_root / stem
    ensure_dir(out_dir)
    copy_detector_pier_crops(summary, out_dir, image)
    section_results = crop_section_symbols(image, summary, out_dir, ocr, threshold)
    vis_path = out_dir / "visualizations" / f"{stem}_ocr_on_full.png"
    draw_full_visualization(image, section_results, vis_path)
    output = {
        "image_name": summary.get("image_name"),
        "image_path": str(image_path),
        "detector_summary": str(summary_path),
        "section_symbol_count": len(section_results),
        "section_symbols": section_results,
        "outputs": {
            "root": str(out_dir),
            "section_symbol_crops": str(out_dir / "crops" / "section_symbols"),
            "pier_crops": str(out_dir / "crops" / "piers"),
            "visualization": str(vis_path),
        },
    }
    save_json(output, out_dir / "json" / f"{stem}_ocr_summary.json")
    return output
