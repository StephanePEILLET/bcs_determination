"""Vectorize inference outputs into a COCO-format dict.

This module converts the raster outputs of the inference pipeline (segmentation
mask in particular) into vector polygons, and assembles all predictions
(classification, pose, segmentation) into a single COCO-compatible JSON dict.

The COCO format is the de-facto standard for bbox + keypoints + polygon
segmentation, and is directly readable by annotation tools such as CVAT,
Label Studio, FiftyOne and ``pycocotools``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("bcs_pipeline")


# Category ids used in the COCO output.
CAT_DOG = 1
CAT_FOREGROUND = 2

# Default keypoint visibility threshold (matches the visualization default).
DEFAULT_KPT_VISIBILITY_THRESHOLD = 0.3


def mask_to_polygons(
    mask: np.ndarray,
    class_id: int = 0,
    simplify_tolerance: float = 2.0,
    min_area: float = 20.0,
) -> List[List[float]]:
    """Extract simplified polygons of a given class from an integer mask.

    Uses ``cv2.findContours`` (RETR_EXTERNAL) + ``cv2.approxPolyDP``
    (Douglas-Peucker) for a good size/fidelity tradeoff. Polygons whose area
    is below ``min_area`` (in pixels²) are dropped.

    Returns a list of flat polygons in COCO format: ``[[x1, y1, x2, y2, ...], ...]``.
    """
    binary = (mask == class_id).astype(np.uint8)
    if binary.sum() == 0:
        return []

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: List[List[float]] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        # Need at least 3 points for a polygon.
        approx = cv2.approxPolyDP(cnt, simplify_tolerance, closed=True)
        if approx.shape[0] < 3:
            continue
        flat = approx.reshape(-1).astype(float).tolist()
        polygons.append(flat)
    return polygons


def _polygon_bbox(polygon: List[float]) -> List[float]:
    """COCO bbox from a flat polygon: [x, y, w, h]."""
    xs = polygon[0::2]
    ys = polygon[1::2]
    x_min, x_max = float(min(xs)), float(max(xs))
    y_min, y_max = float(min(ys)), float(max(ys))
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def _polygon_area(polygon: List[float]) -> float:
    """Shoelace formula on a flat polygon."""
    pts = np.array(polygon, dtype=float).reshape(-1, 2)
    if pts.shape[0] < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _xyxy_to_coco_bbox(xyxy: np.ndarray) -> List[float]:
    """Convert YOLO ``[x1, y1, x2, y2]`` to COCO ``[x, y, w, h]``."""
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    return [x1, y1, x2 - x1, y2 - y1]


def _resolve_keypoint_names(num_kpts: int) -> List[str]:
    """Generate generic keypoint names ``kpt_0..kpt_{N-1}``.

    Custom-trained YOLO pose models do not embed semantic names, so we use a
    stable generic naming scheme. Callers can rename them downstream.
    """
    return [f"kpt_{i}" for i in range(num_kpts)]


def _pose_annotations(
    pose: Dict[str, Any],
    image_id: int,
    next_id: int,
    kpt_visibility_threshold: float = DEFAULT_KPT_VISIBILITY_THRESHOLD,
) -> List[Dict[str, Any]]:
    """One COCO annotation per pose detection (bbox + keypoints)."""
    if pose is None or pose.get("num_detections", 0) == 0:
        return []

    boxes = pose["boxes"]            # (N, 4) xyxy
    box_confs = pose["box_confs"]    # (N,)
    keypoints = pose["keypoints"]    # (N, K, 2)
    kpt_confs = pose["kpt_confs"]    # (N, K)

    annotations: List[Dict[str, Any]] = []
    for n in range(boxes.shape[0]):
        bbox = _xyxy_to_coco_bbox(boxes[n])
        area = bbox[2] * bbox[3]

        kpts_flat: List[float] = []
        n_visible = 0
        for k in range(keypoints.shape[1]):
            x, y = float(keypoints[n, k, 0]), float(keypoints[n, k, 1])
            if kpt_confs[n, k] > kpt_visibility_threshold:
                v = 2
                n_visible += 1
            else:
                v = 0
                # COCO convention: when v=0 the (x, y) are ignored.
                x, y = 0.0, 0.0
            kpts_flat.extend([x, y, v])

        annotations.append({
            "id": next_id,
            "image_id": image_id,
            "category_id": CAT_DOG,
            "bbox": [round(v, 2) for v in bbox],
            "area": round(area, 2),
            "iscrowd": 0,
            "keypoints": [round(v, 2) for v in kpts_flat],
            "num_keypoints": n_visible,
            "score": float(box_confs[n]),
            "source": "yolo_pose",
        })
        next_id += 1
    return annotations


def _segmentation_annotations(
    segmentation: Dict[str, Any],
    image_id: int,
    next_id: int,
    polygon_simplify: float,
    min_area: float = 20.0,
) -> List[Dict[str, Any]]:
    """One COCO annotation per foreground polygon extracted from the mask."""
    if segmentation is None or "mask" not in segmentation:
        return []

    polygons = mask_to_polygons(
        segmentation["mask"],
        class_id=0,  # foreground in the trimap convention
        simplify_tolerance=polygon_simplify,
        min_area=min_area,
    )
    if not polygons:
        return []

    backend = segmentation.get("backend", "unknown")
    annotations: List[Dict[str, Any]] = []
    for poly in polygons:
        bbox = _polygon_bbox(poly)
        annotations.append({
            "id": next_id,
            "image_id": image_id,
            "category_id": CAT_FOREGROUND,
            "bbox": [round(v, 2) for v in bbox],
            "area": round(_polygon_area(poly), 2),
            "iscrowd": 0,
            "segmentation": [[round(v, 2) for v in poly]],
            "source": backend,
        })
        next_id += 1
    return annotations


def build_coco(
    image: Image.Image,
    *,
    image_path: Optional[str] = None,
    classification: Optional[Dict[str, Any]] = None,
    segmentation: Optional[Dict[str, Any]] = None,
    pose: Optional[Dict[str, Any]] = None,
    polygon_simplify: float = 2.0,
    polygon_min_area: float = 20.0,
    kpt_visibility_threshold: float = DEFAULT_KPT_VISIBILITY_THRESHOLD,
) -> Dict[str, Any]:
    """Build a COCO-format dict from the inference pipeline outputs.

    All inputs are optional: branches without data simply produce no
    annotations. The whole-image breed classification is stored as an
    ``predicted_breed`` extension on the image record (out-of-spec but
    convenient and ignored by strict COCO loaders).
    """
    width, height = image.size
    image_id = 1

    file_name = (
        Path(image_path).name if image_path else "inference_input"
    )

    image_record: Dict[str, Any] = {
        "id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height,
    }
    if image_path is not None:
        image_record["path"] = str(image_path)

    if classification is not None:
        image_record["predicted_breed"] = {
            "class_id": int(classification.get("class_id", -1)),
            "class_name": classification.get("class_name"),
            "confidence": float(classification.get("confidence", 0.0)),
            "top_k": classification.get("top_k", []),
        }

    # Categories.
    num_kpts = 0
    if pose is not None and pose.get("num_detections", 0) > 0:
        num_kpts = int(pose["keypoints"].shape[1])

    categories: List[Dict[str, Any]] = [
        {
            "id": CAT_DOG,
            "name": "dog",
            "supercategory": "animal",
            "keypoints": _resolve_keypoint_names(num_kpts),
            "skeleton": [],
        },
        {
            "id": CAT_FOREGROUND,
            "name": "foreground",
            "supercategory": "segmentation",
        },
    ]

    # Annotations: pose first (one per detection), then seg polygons.
    annotations: List[Dict[str, Any]] = []
    next_id = 1
    pose_anns = _pose_annotations(
        pose, image_id, next_id,
        kpt_visibility_threshold=kpt_visibility_threshold,
    )
    annotations.extend(pose_anns)
    next_id += len(pose_anns)
    annotations.extend(_segmentation_annotations(
        segmentation, image_id, next_id,
        polygon_simplify=polygon_simplify,
        min_area=polygon_min_area,
    ))

    return {
        "info": {
            "description": "bcs_pipeline inference output",
            "version": "1.0",
            "date_created": datetime.now(timezone.utc).isoformat(),
        },
        "images": [image_record],
        "categories": categories,
        "annotations": annotations,
    }


def save_coco(coco: Dict[str, Any], output_path: Union[str, Path]) -> Path:
    """Write a COCO dict to disk as pretty-printed JSON."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)
    logger.info("Saved COCO JSON to %s", path)
    return path
