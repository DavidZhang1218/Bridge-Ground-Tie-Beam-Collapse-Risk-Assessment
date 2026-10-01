from __future__ import annotations


import json


import os


from pathlib import Path


from typing import Any, Dict, Iterable, List, Sequence


import numpy as np


from PIL import Image, ImageDraw, ImageFont


try:
    import torch
except Exception:
    torch = None


try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    YOLO = None
    YOLO_IMPORT_ERROR = exc
else:
    YOLO_IMPORT_ERROR = None


THIS_DIR = Path(__file__).resolve().parent


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


CLASS_NAMES = {0: "pier", 1: "section_symbol"}


CLASS_COLORS = {"pier": (255, 214, 10), "section_symbol": (180, 90, 255)}


FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
]


def resolve_path(path_text: str, *, must_exist: bool = False) -> Path:
    path = Path(str(path_text)).expanduser()
    if path.is_absolute():
        return path.resolve()

    script_relative = THIS_DIR / path
    if script_relative.exists() or not must_exist:
        return script_relative.resolve()

    cwd_relative = Path.cwd() / path
    return cwd_relative.resolve()


def safe_path_part(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(text).strip())
    cleaned = cleaned.strip("._")
    return cleaned or "weights"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_font(size: int = 14) -> ImageFont.ImageFont:
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


def list_images(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    return [
        path
        for path in sorted(input_path.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS
    ]


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


def box_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    x11, y11, x12, y12 = [float(v) for v in box1]
    x21, y21, x22, y22 = [float(v) for v in box2]
    ix1 = max(x11, x21)
    iy1 = max(y11, y21)
    ix2 = min(x12, x22)
    iy2 = min(y12, y22)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(box1) + box_area(box2) - inter
    return inter / union if union > 1e-9 else 0.0


def dedup_detections(detections: List[Dict[str, Any]], iou_thr: float) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for class_name in ("pier", "section_symbol"):
        items = [item for item in detections if item.get("class_name") == class_name]
        if class_name == "pier":
            items.sort(
                key=lambda item: (
                    box_area(item["bbox_in_full"]),
                    float(item.get("score", 0.0)),
                ),
                reverse=True,
            )
        else:
            items.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        class_kept: List[Dict[str, Any]] = []
        for item in items:
            if all(box_iou(item["bbox_in_full"], old["bbox_in_full"]) <= iou_thr for old in class_kept):
                class_kept.append(item)
        kept.extend(class_kept)
    return kept


def build_model(weights: Path) -> Any:
    if YOLO is None:
        raise RuntimeError(f"ultralytics is unavailable: {YOLO_IMPORT_ERROR}")
    if not weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights}")
    return YOLO(str(weights))


def run_yolo(
    model: Any,
    image: Image.Image,
    device: str,
    conf: float,
    imgsz: int,
    iou: float,
    max_det: int,
) -> List[Dict[str, Any]]:
    rgb = image.convert("RGB")
    result = model.predict(
        source=np.array(rgb),
        conf=max(0.001, float(conf)),
        iou=float(iou),
        imgsz=int(imgsz),
        device=device,
        max_det=int(max_det),
        rect=True,
        verbose=False,
    )[0]
    detections: List[Dict[str, Any]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return detections

    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    clss = boxes.cls.detach().cpu().numpy()
    for raw_box, score, cls_id in zip(xyxy, confs, clss):
        cls_id = int(cls_id)
        if cls_id not in CLASS_NAMES:
            continue
        box = [float(v) for v in raw_box.tolist()]
        detections.append(
            {
                "class_id": cls_id,
                "class_name": CLASS_NAMES[cls_id],
                "score": float(score),
                "bbox_in_full": box,
                "center_in_full": box_center(box),
            }
        )
    return detections


def draw_detections(
    image: Image.Image,
    detections: Iterable[Dict[str, Any]],
    output_path: Path,
    bbox_key: str,
    title_prefix: str = "",
) -> None:
    ensure_dir(output_path.parent)
    rendered = image.convert("RGB").copy()
    draw = ImageDraw.Draw(rendered)
    font = load_font(14)
    for det in detections:
        box = det.get(bbox_key)
        if not box:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        class_name = str(det.get("class_name", "obj"))
        color = CLASS_COLORS.get(class_name, (255, 0, 0))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{title_prefix}{class_name} {float(det.get('score', 0.0)):.3f}"
        try:
            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except Exception:
            tw, th = font.getsize(label)
        ty = max(0, y1 - th - 2)
        draw.rectangle([x1, ty, x1 + tw, ty + th], fill="yellow")
        draw.text((x1, ty), label, fill="blue", font=font)
    rendered.save(output_path)


def process_one_image(
    image_path: Path,
    output_root: Path,
    model: Any,
    args: argparse.Namespace,
    device: str,
) -> Dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    stem = image_path.stem
    out_dir = output_root / stem
    dirs = {
        "detections": out_dir / "detections",
        "pier_crops": out_dir / "pier_crops",
        "section_symbols": out_dir / "section_symbols",
        "json": out_dir / "json",
    }
    for path in dirs.values():
        ensure_dir(path)

    detections = run_yolo(
        model=model,
        image=image,
        device=device,
        conf=float(args.conf),
        imgsz=int(args.imgsz),
        iou=float(args.iou),
        max_det=int(args.max_det),
    )
    detections = dedup_detections(detections, iou_thr=float(args.dedup_iou))
    for idx, det in enumerate(detections, start=1):
        det["id"] = idx

    pier_dets = [det for det in detections if det["class_name"] == "pier"]
    section_dets = [det for det in detections if det["class_name"] == "section_symbol"]
    pier_dets.sort(key=lambda item: (item["center_in_full"][0], item["center_in_full"][1]))
    section_dets.sort(key=lambda item: (item["center_in_full"][0], item["center_in_full"][1]))
    for pier_id, det in enumerate(pier_dets, start=1):
        det["pier_id"] = pier_id
    for section_symbol_id, det in enumerate(section_dets, start=1):
        det["section_symbol_id"] = section_symbol_id

    pier_crops: List[Dict[str, Any]] = []
    for pier_id, det in enumerate(pier_dets, start=1):
        crop_box = clip_box(det["bbox_in_full"], width, height)
        crop = image.crop(tuple(crop_box))
        crop_name = f"{stem}_pier_{pier_id:03d}.png"
        crop_path = dirs["pier_crops"] / crop_name
        crop.save(crop_path)
        pier_crops.append(
            {
                "pier_id": pier_id,
                "file_name": crop_name,
                "path": str(crop_path),
                "rect_in_full": crop_box,
                "size": [crop.width, crop.height],
                "source_detection": det,
            }
        )

    detections_full_path = dirs["detections"] / f"{stem}_detections_on_full.png"
    section_image_path = dirs["section_symbols"] / f"{stem}_section_symbols_on_full.png"
    draw_detections(image, detections, detections_full_path, bbox_key="bbox_in_full")
    draw_detections(image, section_dets, section_image_path, bbox_key="bbox_in_full")

    section_json = {
        "image_name": image_path.name,
        "section_symbol_count": len(section_dets),
        "section_symbols": section_dets,
        "section_symbol_image": str(section_image_path),
    }
    pier_crop_json = {
        "image_name": image_path.name,
        "pier_count": len(pier_dets),
        "pier_crops": pier_crops,
    }
    summary = {
        "image_name": image_path.name,
        "image_path": str(image_path.resolve()),
        "image_size": [width, height],
        "weights": getattr(args, "weights_resolved", str(Path(args.weights).resolve())),
        "device": device,
        "conf": float(args.conf),
        "imgsz": int(args.imgsz),
        "detections_on_full": str(detections_full_path),
        "num_detections": len(detections),
        "pier_count": len(pier_dets),
        "section_symbol_count": len(section_dets),
        "detections": detections,
        "pier_detections": pier_dets,
        "pier_crops": pier_crops,
        "section_symbols": section_dets,
        "section_symbol_image": str(section_image_path),
    }

    save_json(section_json, dirs["section_symbols"] / f"{stem}_section_symbols.json")
    save_json(pier_crop_json, dirs["pier_crops"] / f"{stem}_pier_crops.json")
    summary_path = dirs["json"] / f"{stem}_pier_section_summary.json"
    save_json(summary, summary_path)
    print(f"[OK] {image_path.name}: pier={len(pier_dets)} section_symbol={len(section_dets)} -> {out_dir}")
    return summary
