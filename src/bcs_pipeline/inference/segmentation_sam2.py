"""SAM 2 segmentation backend.

SAM 2 is a zero-shot foundation segmentation model: no fine-tuning was performed
on Oxford-IIIT Pet, so this backend trades the supervised quality of DeepLabV3
for a much stronger generalization (works on cats, exotic breeds, OOD photos).

Three modes are exposed:

* ``"prompted"`` — places a single positive point prompt at the image center
  and selects the highest-score mask. Fastest, most reliable when the subject
  is roughly centered.
* ``"automatic"`` — runs the dense ``SAM2AutomaticMaskGenerator`` and selects
  the mask whose centroid is closest to the image center. Slower but works on
  off-center subjects.
* ``"pose_prompted"`` — uses the YOLO pose detector outputs (bounding box +
  keypoints) as SAM 2 prompts. The bbox constrains the mask spatially and the
  keypoints act as positive points inside the subject. Most accurate when the
  pose model has high-confidence detections.

The binary SAM 2 mask is exposed directly as a 2-class map
(0=foreground, 1=background). Unlike DeepLabV3, no border class is derived
since the SAM 2 silhouette is already pixel-accurate.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

from bcs_pipeline.utils.device import get_best_device

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("bcs_pipeline")


# Default config name shipped with the ``sam2`` PyPI package. Hiera Large.
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def load_sam2_model(
    checkpoint_path: str,
    config_path: str = DEFAULT_SAM2_CONFIG,
    device: Optional[torch.device] = None,
):
    """Load a SAM 2 image predictor + automatic mask generator.

    Returns a dict with the underlying ``sam2_model``, ``predictor`` and
    ``auto_gen`` so both modes can be used without reloading.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint_path}")

    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if device is None:
        device = get_best_device()

    logger.info("Loading SAM2 model from %s (device=%s)…", checkpoint_path, device)
    sam2_model = build_sam2(config_path, checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    auto_gen = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=32,
        points_per_batch=64,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.92,
    )
    return {
        "sam2_model": sam2_model,
        "predictor": predictor,
        "auto_gen": auto_gen,
        "device": device,
    }


def _sam2_prompted(handle: Dict[str, Any], image: Image.Image) -> np.ndarray:
    """Single positive point prompt at the image center."""
    img_np = np.array(image.convert("RGB"))
    w, h = image.size
    predictor = handle["predictor"]
    predictor.set_image(img_np)
    masks, scores, _ = predictor.predict(
        point_coords=np.array([[w // 2, h // 2]]),
        point_labels=np.array([1]),
        multimask_output=True,
    )
    return masks[int(np.argmax(scores))].astype(bool)


def _sam2_pose_prompted(
    handle: Dict[str, Any],
    image: Image.Image,
    pose_result: Dict[str, Any],
    kpt_threshold: float = 0.3,
) -> np.ndarray:
    """Use YOLO pose outputs (highest-confidence detection) as SAM 2 prompts.

    The bbox is passed as a hard spatial constraint, and keypoints with
    confidence ``> kpt_threshold`` are passed as positive points inside the
    subject. Falls back to the ``prompted`` (image-center) strategy when the
    pose detector returned no detection.
    """
    if pose_result is None or pose_result.get("num_detections", 0) == 0:
        logger.warning("pose_prompted SAM2 requested but no pose detection — "
                       "falling back to center-point prompt.")
        return _sam2_prompted(handle, image)

    img_np = np.array(image.convert("RGB"))
    predictor = handle["predictor"]
    predictor.set_image(img_np)

    # Pick the highest-confidence detection.
    best = int(np.argmax(pose_result["box_confs"]))
    box = pose_result["boxes"][best].astype(np.float32)            # (4,) xyxy
    kpts = pose_result["keypoints"][best]                          # (K, 2)
    kpt_confs = pose_result["kpt_confs"][best]                     # (K,)

    valid = kpt_confs > kpt_threshold
    if valid.any():
        point_coords = kpts[valid].astype(np.float32)
        point_labels = np.ones(int(valid.sum()), dtype=np.int32)
    else:
        point_coords = None
        point_labels = None

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=True,
    )
    return masks[int(np.argmax(scores))].astype(bool)


def _sam2_automatic(handle: Dict[str, Any], image: Image.Image) -> np.ndarray:
    """Pick the auto-generated mask whose centroid is closest to the center."""
    img_np = np.array(image.convert("RGB"))
    w, h = image.size
    centre = np.array([w / 2, h / 2])
    all_masks = handle["auto_gen"].generate(img_np)
    if not all_masks:
        return np.zeros((h, w), dtype=bool)

    best_m, best_d = None, float("inf")
    for m in all_masks:
        ys, xs = np.where(m["segmentation"])
        if len(xs) == 0:
            continue
        d = float(np.hypot(xs.mean() - centre[0], ys.mean() - centre[1]))
        if d < best_d:
            best_d, best_m = d, m["segmentation"]
    if best_m is None:
        best_m = all_masks[0]["segmentation"]
    return np.asarray(best_m, dtype=bool)


def sam2_mask_to_trimap(binary_mask: np.ndarray) -> np.ndarray:
    """Convert a boolean SAM2 mask into a 2-class map.

    ``0 = foreground``, ``1 = background``. The DeepLabV3 border class is
    intentionally not produced for SAM 2 — the silhouette is already crisp.
    """
    binary_mask = np.asarray(binary_mask, dtype=bool)
    out = np.ones(binary_mask.shape, dtype=np.int64)
    out[binary_mask] = 0
    return out


def predict_segmentation_sam2(
    handle: Dict[str, Any],
    image: Image.Image,
    mode: str = "prompted",
    pose_result: Optional[Dict[str, Any]] = None,
    kpt_threshold: float = 0.3,
) -> Dict[str, Any]:
    """Run SAM2 segmentation and return the same dict shape as DeepLabV3.

    ``mode`` is ``"prompted"`` (default, fastest), ``"automatic"`` (slower) or
    ``"pose_prompted"`` (requires ``pose_result``). The returned mask matches
    the input image size.
    """
    if mode == "prompted":
        binary = _sam2_prompted(handle, image)
    elif mode == "automatic":
        binary = _sam2_automatic(handle, image)
    elif mode == "pose_prompted":
        binary = _sam2_pose_prompted(
            handle, image, pose_result, kpt_threshold=kpt_threshold,
        )
    else:
        raise ValueError(
            f"Unknown SAM2 mode: {mode!r}. "
            "Use 'prompted', 'automatic' or 'pose_prompted'."
        )

    mask = sam2_mask_to_trimap(binary)
    return {
        "mask": mask,
        "mask_small": mask,  # SAM2 already produces full-resolution masks
        "num_classes": 2,
        "binary_mask": binary,
        "backend": f"sam2_{mode}",
    }
