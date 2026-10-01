"""Extract ground tie beam heights from a bridge section drawing."""

from __future__ import annotations

import io
import json
import math
import re
import time
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Callable


REFERENCE_PROMPT = """The first image is an annotated reference for ground tie beam height extraction.

The blue dashed boxes identify local section titles. Each Roman-numeral title identifies an independent section. A ground tie beam is a horizontal rectangular member near or between the pier columns, piles, or pile cap; the red dashed box shows an example. Its height is the vertical dimension inside the rectangular member; the green dashed box marks a height of 150 cm in Section I-I. If a section contains repeated beams or repeated height labels with the same value, report that value once. Report every Roman-numeral section, including sections without a ground tie beam, whose height must be null.

The reference example corresponds to:
[{"roman_numeral":"I-I","grounding_beam_thickness":150},{"roman_numeral":"II-II","grounding_beam_thickness":null}]

Use this example to interpret the following section-view pages. Report measurements from those pages only."""

TARGET_PROMPT = """Every following image is a section-view page from the same PDF, in page order. Extract the ground tie beam height for every Roman-numeral section on all these pages using the reference rules. Return only a JSON array. Each item must contain roman_numeral (for example, "I-I") and grounding_beam_thickness (a number in centimeters, or null if the section has no ground tie beam). Include every Roman-numeral section shown; report a repeated section label only once."""

class SectionHeightError(RuntimeError):
    """The section drawing could not be interpreted reliably."""


def normalize_section_label(label: object) -> str | None:
    """Convert a section label such as ``II-II`` or ``Ⅱ–Ⅱ`` to ``II``."""
    if label is None or isinstance(label, bool):
        return None
    if isinstance(label, int) or (isinstance(label, str) and label.strip().isdigit()):
        number = int(label)
        if not 1 <= number <= 20:
            return None
        tens, units = divmod(number, 10)
        return "X" * tens + ("", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX")[units]

    text = unicodedata.normalize("NFKC", str(label)).upper().strip()
    text = re.sub(r"\s+", "", text)
    for dash in ("–", "—", "−", "‑", "‐", "－"):
        text = text.replace(dash, "-")
    match = re.fullmatch(r"([IVXLCDM]+)(?:-([IVXLCDM]+))?", text)
    if match is None or (match.group(2) and match.group(1) != match.group(2)):
        return None
    return match.group(1)


def _response_text(response: object) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    parts = []
    for candidate in getattr(response, "candidates", None) or []:
        for part in getattr(getattr(candidate, "content", None), "parts", None) or []:
            value = getattr(part, "text", None)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
    return "\n".join(parts)


def _load_json_response(text: str) -> object:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for opening, closing in (("[", "]"), ("{", "}")):
        start, end = cleaned.find(opening), cleaned.rfind(closing)
        if 0 <= start < end:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                pass
    raise SectionHeightError("The section response was not valid JSON.")


def parse_height_response(text: str) -> dict[str, float | None]:
    """Validate the JSON response and return heights in centimeters by section."""
    payload = _load_json_response(text)
    if isinstance(payload, dict):
        payload = payload.get("results", payload)
    if isinstance(payload, dict) and "roman_numeral" in payload:
        payload = [payload]
    if not isinstance(payload, list) or not payload:
        raise SectionHeightError("No section height records were returned.")

    heights: dict[str, float | None] = {}
    for item in payload:
        if not isinstance(item, dict):
            raise SectionHeightError("A section height record is not an object.")
        section = normalize_section_label(item.get("roman_numeral"))
        if section is None or "grounding_beam_thickness" not in item:
            raise SectionHeightError("A section height record has an invalid label or missing height.")
        raw_height = item["grounding_beam_thickness"]
        if raw_height is None or (isinstance(raw_height, str) and raw_height.strip().lower() == "null"):
            height = None
        elif isinstance(raw_height, bool):
            raise SectionHeightError("A section height is not numeric.")
        else:
            value = str(raw_height).strip()
            if re.fullmatch(r"\d+(?:\.\d+)?(?:\s*cm)?", value, flags=re.IGNORECASE) is None:
                raise SectionHeightError("A section height is not a number in centimeters.")
            height = float(re.match(r"\d+(?:\.\d+)?", value).group())
            if not math.isfinite(height) or height <= 0:
                raise SectionHeightError("A section height must be positive.")
        if section in heights and heights[section] != height:
            raise SectionHeightError("Conflicting heights were returned for one section.")
        heights[section] = height
    return heights


def _image_part(path: Path, *, white_background: bool):
    from google.genai import types
    from PIL import Image, ImageOps

    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    if white_background:
        with Image.open(path) as source:
            rgba = ImageOps.exif_transpose(source).convert("RGBA")
            composed = Image.new("RGBA", rgba.size, "white")
            composed.alpha_composite(rgba)
            buffer = io.BytesIO()
            composed.convert("RGB").save(buffer, format="PNG")
        return types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/png")
    suffix = path.suffix.lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(suffix)
    if mime is None:
        raise ValueError("Section image must be JPEG, PNG, or WebP.")
    return types.Part.from_bytes(data=path.read_bytes(), mime_type=mime)


def extract_section_heights(
    section_images: Sequence[Path],
    icl_image: Path,
    api_key: str,
    model_id: str = "gemini-3-flash-preview",
    status: Callable[[str], None] | None = None,
) -> dict[str, float | None]:
    """Extract heights from all section pages with one multimodal ICL request."""
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("A Gemini API key is required.")
    if not model_id or not model_id.strip():
        raise ValueError("A Gemini model ID is required.")
    section_images = [Path(path) for path in section_images]
    icl_image = Path(icl_image)
    if not section_images or not icl_image.is_file() or any(not path.is_file() for path in section_images):
        raise FileNotFoundError("A section page or ICL reference image is missing.")
    try:
        from google import genai
        from google.genai import types
        from httpx import TransportError
    except ImportError:
        raise SectionHeightError("The google-genai package is required for section extraction.") from None

    if status:
        status("Preparing section images for Gemini")
    try:
        reference_part = _image_part(icl_image, white_background=True)
        contents = [REFERENCE_PROMPT, reference_part, TARGET_PROMPT]
        for page_number, section_image in enumerate(section_images, start=2):
            contents.extend((f"PDF page {page_number}: section-view drawing.", _image_part(section_image, white_background=False)))
    except Exception:
        raise SectionHeightError("The reference or section image could not be prepared for Gemini.") from None

    for attempt in range(2):
        try:
            client = genai.Client(api_key=api_key.strip())
            try:
                if status and attempt == 0:
                    status("Waiting for Gemini API response")
                response = client.models.generate_content(
                    model=model_id.strip(),
                    contents=contents,
                    config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json"),
                )
            finally:
                client.close()
            break
        except Exception as error:
            code = getattr(error, "code", None) or getattr(error, "status_code", None)
            retryable = isinstance(error, TransportError) or code in (429, 500, 502, 503, 504)
            if attempt == 0 and retryable:
                if status:
                    status("Gemini API connection or service interrupted; retrying once")
                time.sleep(2)
                continue
            raise SectionHeightError(
                "Gemini section extraction failed. Check the API key, model ID, connection, and quota."
            ) from None

    if status:
        status("Checking Gemini section heights")
    try:
        response_text = _response_text(response)
    except Exception:
        raise SectionHeightError("Gemini returned no readable section response.") from None
    return parse_height_response(response_text)
