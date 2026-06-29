"""PIL-based overlays to combine classification, segmentation and pose outputs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("bcs_pipeline")


# Class -> overlay RGB color. Class 1 (background) is intentionally absent
# so it remains transparent; class 2 (border) shares the foreground green
# nuance for a coherent silhouette.
DEFAULT_OVERLAY_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 200, 0),       # foreground (animal) - green
    2: (255, 200, 0),     # border - yellow
}

DEFAULT_TRANSPARENT_CLASSES: Set[int] = {1}  # background class


def _load_font(size: int = 18) -> ImageFont.ImageFont:
    """Try DejaVuSans, fall back to PIL default. Default font ignores size."""
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def overlay_segmentation(
    image: Image.Image,
    mask: np.ndarray,
    alpha: float = 0.4,
    colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    transparent_classes: Optional[Set[int]] = None,
) -> Image.Image:
    """Alpha-blend a colored mask on top of a PIL image.

    ``mask`` must already match ``image.size`` (W, H) i.e. shape (H, W).
    Classes in ``transparent_classes`` are not painted.
    """
    if colors is None:
        colors = DEFAULT_OVERLAY_COLORS
    if transparent_classes is None:
        transparent_classes = DEFAULT_TRANSPARENT_CLASSES

    base = image.convert("RGB")
    h, w = mask.shape
    if (w, h) != base.size:
        # Defensive: if caller forgot to resize, do it here (NEAREST).
        mask_img = Image.fromarray(mask.astype(np.uint8)).resize(base.size, Image.NEAREST)
        mask = np.array(mask_img, dtype=np.int64)
        h, w = mask.shape

    rgba_overlay = np.zeros((h, w, 4), dtype=np.uint8)
    for cls_idx, color in colors.items():
        if cls_idx in transparent_classes:
            continue
        sel = mask == cls_idx
        if not sel.any():
            continue
        rgba_overlay[sel, 0] = color[0]
        rgba_overlay[sel, 1] = color[1]
        rgba_overlay[sel, 2] = color[2]
        rgba_overlay[sel, 3] = int(255 * alpha)

    overlay_img = Image.fromarray(rgba_overlay, mode="RGBA")
    composed = Image.alpha_composite(base.convert("RGBA"), overlay_img)
    return composed.convert("RGB")


def draw_pose(
    image: Image.Image,
    boxes: np.ndarray,
    keypoints: np.ndarray,
    kpt_confs: np.ndarray,
    box_color: Tuple[int, int, int] = (50, 255, 50),
    kpt_color: Tuple[int, int, int] = (255, 0, 255),
    kpt_radius: int = 5,
    kpt_threshold: float = 0.3,
    box_width: int = 3,
    show_boxes: bool = True,
    show_keypoints: bool = True,
) -> Image.Image:
    """Draw bounding boxes and keypoint dots.

    Each visual element can be independently toggled via ``show_boxes``
    and ``show_keypoints``.
    """
    if boxes.shape[0] == 0:
        return image

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    if show_boxes:
        for x1, y1, x2, y2 in boxes:
            draw.rectangle([(float(x1), float(y1)), (float(x2), float(y2))],
                           outline=box_color, width=box_width)

    if show_keypoints:
        for n in range(keypoints.shape[0]):
            for k in range(keypoints.shape[1]):
                if kpt_confs[n, k] < kpt_threshold:
                    continue
                x, y = keypoints[n, k]
                r = kpt_radius
                draw.ellipse([(float(x) - r, float(y) - r),
                              (float(x) + r, float(y) + r)],
                             fill=kpt_color, outline=(255, 255, 255))

    return canvas


def draw_label_banner(
    image: Image.Image,
    text: str,
    bg_alpha: int = 180,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    height: int = 36,
    font_size: int = 18,
) -> Image.Image:
    """Draw a translucent black bar across the top of the image with text."""
    base = image.convert("RGBA")
    w, _ = base.size
    bar = Image.new("RGBA", (w, height), (0, 0, 0, bg_alpha))
    base.paste(bar, (0, 0), bar)

    draw = ImageDraw.Draw(base)
    font = _load_font(font_size)
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:
        text_w, text_h = draw.textsize(text, font=font)

    x = max(8, (w - text_w) // 2)
    y = max(0, (height - text_h) // 2)
    draw.text((x, y), text, fill=text_color, font=font)
    return base.convert("RGB")


def _format_classification_label(classification: Dict) -> str:
    name = classification.get("class_name") or f"Class {classification.get('class_id', '?')}"
    conf = classification.get("confidence", 0.0)
    return f"{name} ({conf * 100:.1f}%)"


def _format_banner_label(
    classification: Optional[Dict] = None, bcs: Optional[Dict] = None
) -> str:
    """Build the top-banner text from the available branches."""
    parts: List[str] = []
    if classification is not None:
        parts.append(_format_classification_label(classification))
    if bcs is not None and bcs.get("bcs") is not None:
        parts.append(f"BCS {bcs['bcs']:.1f}/9 — {bcs.get('category', '')}".rstrip(" —"))
    return "  |  ".join(parts)


def render_combined(
    image: Image.Image,
    *,
    classification: Optional[Dict] = None,
    segmentation: Optional[Dict] = None,
    pose: Optional[Dict] = None,
    bcs: Optional[Dict] = None,
    seg_alpha: float = 0.4,
    show_segmentation: bool = True,
    show_boxes: bool = True,
    show_keypoints: bool = True,
    show_label: bool = True,
) -> Image.Image:
    """Compose available overlays onto a single PIL image.

    Each layer can be independently toggled:
    ``show_segmentation``, ``show_boxes``, ``show_keypoints``,
    ``show_label``.  When a toggle is ``False`` the
    corresponding visual is skipped even if the data is provided.
    """
    canvas = image.convert("RGB")

    if segmentation is not None and "mask" in segmentation and show_segmentation:
        canvas = overlay_segmentation(canvas, segmentation["mask"], alpha=seg_alpha)

    if pose is not None and pose.get("num_detections", 0) > 0:
        canvas = draw_pose(
            canvas,
            boxes=pose["boxes"],
            keypoints=pose["keypoints"],
            kpt_confs=pose["kpt_confs"],
            show_boxes=show_boxes,
            show_keypoints=show_keypoints,
        )

    if show_label and (classification is not None or bcs is not None):
        label = _format_banner_label(classification, bcs)
        if label:
            canvas = draw_label_banner(canvas, label)

    return canvas


def render_segmentation_layer(
    image: Image.Image,
    mask: np.ndarray,
    alpha: float = 0.4,
    colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    transparent_classes: Optional[Set[int]] = None,
) -> Image.Image:
    """Return an RGBA image with *only* the segmentation colour overlay on a
    fully-transparent background, ready for CSS stacking."""
    if colors is None:
        colors = DEFAULT_OVERLAY_COLORS
    if transparent_classes is None:
        transparent_classes = DEFAULT_TRANSPARENT_CLASSES

    base = image.convert("RGB")
    h, w = mask.shape
    if (w, h) != base.size:
        mask_img = Image.fromarray(mask.astype(np.uint8)).resize(base.size, Image.NEAREST)
        mask = np.array(mask_img, dtype=np.int64)
        h, w = mask.shape

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for cls_idx, color in colors.items():
        if cls_idx in transparent_classes:
            continue
        sel = mask == cls_idx
        if not sel.any():
            continue
        rgba[sel, 0] = color[0]
        rgba[sel, 1] = color[1]
        rgba[sel, 2] = color[2]
        rgba[sel, 3] = int(255 * alpha)

    return Image.fromarray(rgba, mode="RGBA")


def render_pose_layer(
    image_size: Tuple[int, int],
    boxes: np.ndarray,
    keypoints: np.ndarray,
    kpt_confs: np.ndarray,
    box_color: Tuple[int, int, int] = (50, 255, 50),
    kpt_color: Tuple[int, int, int] = (255, 0, 255),
    kpt_radius: int = 5,
    kpt_threshold: float = 0.3,
    box_width: int = 3,
    show_boxes: bool = True,
    show_keypoints: bool = True,
) -> Image.Image:
    """Return an RGBA image with *only* the pose elements (boxes and/or
    keypoints) on a fully-transparent background."""
    canvas = Image.new("RGBA", image_size, (0, 0, 0, 0))
    if boxes.shape[0] == 0:
        return canvas
    draw = ImageDraw.Draw(canvas)

    if show_boxes:
        for x1, y1, x2, y2 in boxes:
            draw.rectangle(
                [(float(x1), float(y1)), (float(x2), float(y2))],
                outline=box_color + (255,),
                width=box_width,
            )

    if show_keypoints:
        for n in range(keypoints.shape[0]):
            for k in range(keypoints.shape[1]):
                if kpt_confs[n, k] < kpt_threshold:
                    continue
                x, y = keypoints[n, k]
                r = kpt_radius
                draw.ellipse(
                    [(float(x) - r, float(y) - r), (float(x) + r, float(y) + r)],
                    fill=kpt_color + (255,),
                    outline=(255, 255, 255, 255),
                )

    return canvas


def save_visualization(image: Image.Image, output_path: Union[str, Path]) -> Path:
    """Save a PIL image as PNG, creating parent directories if needed."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    logger.info("Saved visualization to %s", path)
    return path
