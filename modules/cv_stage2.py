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


CLASS_COLORS = {
    "candidate": (34, 197, 94),
    "beam_level": (59, 130, 246),
    "other": (245, 158, 11),
}


FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
]


def resolve_path(path_text: str | Path, *, must_exist: bool = False) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path.resolve()

    script_relative = THIS_DIR / path
    if script_relative.exists() or not must_exist:
        return script_relative.resolve()

    cwd_relative = Path.cwd() / path
    return cwd_relative.resolve()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def list_pier_crop_dirs(input_path: Path) -> List[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path must be a directory: {input_path}")
    if input_path.name == "pier_crops":
        return [input_path]

    project_pier_crops = input_path / "pier_crops"
    if project_pier_crops.is_dir():
        return [project_pier_crops]

    return sorted(
        path
        for path in input_path.glob("*/pier_crops")
        if path.is_dir()
    )


def list_images(input_path: Path) -> List[Path]:
    return [
        path
        for path in sorted(input_path.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS
    ]


def project_name_from_crop_dir(crop_dir: Path) -> str:
    if crop_dir.name == "pier_crops" and crop_dir.parent.name:
        return crop_dir.parent.name
    return crop_dir.name


def normalize_class_name(name: Any) -> str:
    text = str(name or "").strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    return "_".join(part for part in text.split("_") if part)


def class_bucket(class_name: Any) -> str:
    normalized = normalize_class_name(class_name)
    if normalized in {"candidate", "beam_candidate", "ground_beam_candidate", "ground_tie_beam_candidate"}:
        return "candidate"
    if normalized in {"beam_level", "beamlevel"}:
        return "beam_level"
    return "other"


def public_class_name(class_name: Any) -> str:
    bucket = class_bucket(class_name)
    if bucket == "candidate":
        return "candidate"
    if bucket == "beam_level":
        return "beam_level"
    return normalize_class_name(class_name) or str(class_name)


def class_counts(detections: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"candidate_count": 0, "beam_level_count": 0}
    for det in detections:
        bucket = class_bucket(det.get("class_name"))
        if bucket == "candidate":
            counts["candidate_count"] += 1
        elif bucket == "beam_level":
            counts["beam_level_count"] += 1
    return counts


def box_center(box: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def clip_box(box: Sequence[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(x1, float(width)))
    y1 = max(0.0, min(y1, float(height)))
    x2 = max(0.0, min(x2, float(width)))
    y2 = max(0.0, min(y2, float(height)))
    if x2 <= x1:
        x2 = min(float(width), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(height), y1 + 1.0)
    return [x1, y1, x2, y2]


def offset_box(box: Sequence[float], offset_x: float, offset_y: float) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y]


def load_pier_crop_metadata(crop_dir: Path) -> Dict[str, Dict[str, Any]]:
    metadata: Dict[str, Dict[str, Any]] = {}
    for path in sorted(crop_dir.glob("*_pier_crops.json")):
        try:
            data = load_json(path)
        except Exception:
            continue
        for item in data.get("pier_crops", []) or []:
            file_name = str(item.get("file_name") or Path(str(item.get("path") or "")).name)
            if file_name:
                metadata[file_name] = item
    return metadata


def build_model(weights: Path) -> Any:
    if YOLO is None:
        raise RuntimeError(f"ultralytics is unavailable: {YOLO_IMPORT_ERROR}")
    if not weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights}")
    return YOLO(str(weights))


def model_class_names(model: Any) -> Dict[int, str]:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return {int(key): public_class_name(value) for key, value in names.items()}
    if isinstance(names, (list, tuple)):
        return {idx: public_class_name(value) for idx, value in enumerate(names)}
    return {}


def run_yolo(
    model: Any,
    class_names: Dict[int, str],
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

    result_names = getattr(result, "names", None)
    if isinstance(result_names, dict):
        class_names = {int(key): public_class_name(value) for key, value in result_names.items()}

    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    clss = boxes.cls.detach().cpu().numpy()
    for raw_box, score, cls_id in zip(xyxy, confs, clss):
        cls_id = int(cls_id)
        box = clip_box(raw_box.tolist(), image.width, image.height)
        class_name = public_class_name(class_names.get(cls_id, str(cls_id)))
        detections.append(
            {
                "class_id": cls_id,
                "class_name": class_name,
                "score": float(score),
                "bbox_in_crop": box,
                "center_in_crop": box_center(box),
            }
        )
    return detections


def draw_detections(
    image: Image.Image,
    detections: Iterable[Dict[str, Any]],
    output_path: Path,
) -> None:
    ensure_dir(output_path.parent)
    rendered = image.convert("RGB").copy()
    draw = ImageDraw.Draw(rendered)
    font = load_font(14)
    for det in detections:
        box = det.get("bbox_in_crop")
        if not box:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        class_name = str(det.get("class_name", "obj"))
        color = CLASS_COLORS.get(class_bucket(class_name), CLASS_COLORS["other"])
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{class_name} {float(det.get('score', 0.0)):.3f}"
        try:
            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except Exception:
            tw, th = font.getsize(label)
        ty = max(0, y1 - th - 2)
        draw.rectangle([x1, ty, x1 + tw + 2, ty + th + 2], fill=color)
        draw.text((x1 + 1, ty + 1), label, fill="white", font=font)
    rendered.save(output_path)


def process_one_crop(
    crop_path: Path,
    output_dir: Path,
    model: Any,
    class_names: Dict[int, str],
    args: argparse.Namespace,
    device: str,
    crop_meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    image = Image.open(crop_path).convert("RGB")
    detections = run_yolo(
        model=model,
        class_names=class_names,
        image=image,
        device=device,
        conf=float(args.conf),
        imgsz=int(args.imgsz),
        iou=float(args.iou),
        max_det=int(args.max_det),
    )
    detections.sort(key=lambda item: (item["center_in_crop"][1], item["center_in_crop"][0]))
    rect = crop_meta.get("rect_in_full") if isinstance(crop_meta, dict) else None
    if isinstance(rect, list) and len(rect) == 4:
        offset_x, offset_y = float(rect[0]), float(rect[1])
        for det in detections:
            full_box = offset_box(det["bbox_in_crop"], offset_x, offset_y)
            det["bbox_in_full"] = full_box
            det["center_in_full"] = box_center(full_box)
            det["crop_rect_in_full"] = rect
            if crop_meta.get("pier_id") is not None:
                det["source_pier_id"] = crop_meta.get("pier_id")
    for idx, det in enumerate(detections, start=1):
        det["id"] = idx

    vis_path = output_dir / "detections" / f"{crop_path.stem}_detector2.png"
    draw_detections(image, detections, vis_path)
    counts = class_counts(detections)
    return {
        "image_name": crop_path.name,
        "image_path": str(crop_path.resolve()),
        "image_size": [image.width, image.height],
        "detections_on_crop": str(vis_path),
        "num_detections": len(detections),
        "candidate_count": counts["candidate_count"],
        "beam_level_count": counts["beam_level_count"],
        "detections": detections,
    }


def process_one_project(
    crop_dir: Path,
    output_root: Path,
    model: Any,
    class_names: Dict[int, str],
    args: argparse.Namespace,
    device: str,
) -> Dict[str, Any]:
    project_name = project_name_from_crop_dir(crop_dir)
    out_dir = output_root / project_name
    ensure_dir(out_dir / "detections")
    ensure_dir(out_dir / "json")

    crop_paths = list_images(crop_dir)
    crop_metadata = load_pier_crop_metadata(crop_dir)
    crop_outputs = [
        process_one_crop(
            crop_path=crop_path,
            output_dir=out_dir,
            model=model,
            class_names=class_names,
            args=args,
            device=device,
            crop_meta=crop_metadata.get(crop_path.name),
        )
        for crop_path in crop_paths
    ]
    candidate_count = sum(item["candidate_count"] for item in crop_outputs)
    beam_level_count = sum(item["beam_level_count"] for item in crop_outputs)
    num_detections = sum(item["num_detections"] for item in crop_outputs)
    summary_path = out_dir / "json" / f"{project_name}_detector2_summary.json"
    summary = {
        "project_name": project_name,
        "input_pier_crops": str(crop_dir.resolve()),
        "weights": getattr(args, "weights_resolved", str(Path(args.weights).resolve())),
        "model_names": class_names,
        "device": device,
        "conf": float(args.conf),
        "imgsz": int(args.imgsz),
        "iou": float(args.iou),
        "max_det": int(args.max_det),
        "crop_count": len(crop_outputs),
        "num_detections": num_detections,
        "candidate_count": candidate_count,
        "beam_level_count": beam_level_count,
        "crops": crop_outputs,
        "outputs": {
            "root": str(out_dir),
            "detections": str(out_dir / "detections"),
            "json": str(out_dir / "json"),
            "summary_json": str(summary_path),
        },
    }
    save_json(summary, summary_path)
    print(
        f"[OK] {project_name}: crops={len(crop_outputs)} "
        f"candidate={candidate_count} beam_level={beam_level_count} -> {out_dir}"
    )
    return summary
