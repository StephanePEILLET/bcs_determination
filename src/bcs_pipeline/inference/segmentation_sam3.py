"""SAM 3 segmentation backend.

SAM 3 is the next iteration of Meta's Segment Anything family. In addition to
the SAM-1 / SAM-2 visual prompts (points, boxes, masks), SAM 3 natively
supports a ``concept-grounded`` mode where the prompt is a free-text phrase
describing the target object. This backend exposes both, and combines them so
the BCS pipeline can prompt SAM 3 with everything it already computes:

* the **bounding box** + **keypoints** of the YOLO pose detection (visual
  prompts), and
* the **predicted breed** of the classifier (text prompt).

The notebook this implementation follows is ``examples/sam3_for_sam1_task_example.ipynb``
in https://github.com/facebookresearch/sam3 (visual prompts via
``model.predict_inst``) plus the README's text-prompt API
(``processor.set_text_prompt``).

Four modes are exposed, all returning the same dict shape as the SAM 2 backend
(2-class trimap: ``0=foreground, 1=background``):

* ``"prompted"`` — single positive point at the image center (cheapest,
  matches SAM 2's default behaviour).
* ``"pose_prompted"`` — YOLO bbox + every visible keypoint (``kpt_conf > 0``).
* ``"concept_prompted"`` — predicted breed name as a text/concept prompt.
* ``"pose_concept_prompted"`` — runs both visual and concept paths, picks the
  candidate with the highest score. Falls back gracefully when one signal
  is missing (e.g. no pose detection or no class name).

The ``sam3`` package is imported lazily inside ``load_sam3_model`` so the rest
of the application keeps working when SAM 3 is not installed.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from bcs_pipeline.utils.device import get_best_device

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("bcs_pipeline")


def load_sam3_model(
    checkpoint_path: str,
    bpe_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Build a SAM 3 image model + processor.

    Returns a handle dict ``{"model", "processor", "device", "checkpoint_path"}``
    so both the visual-prompt path (``model.predict_inst``) and the
    text-prompt path (``processor.set_text_prompt``) can be reused.

    The ``sam3`` package is imported lazily so the application starts even
    when it is not installed (Python ≥ 3.12 / CUDA ≥ 12.6 requirement).
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint_path}")

    try:
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as exc:
        # Distinguish "package missing" from "package present but a transitive
        # import failed" — the latter happens e.g. with setuptools ≥ 81 where
        # `pkg_resources` is no longer in the default install (sam3 imports it
        # unconditionally at module load).
        missing = exc.name or ""
        if missing == "sam3":
            hint = (
                "Re-run `uv sync --extra dev` (or `pip install -e \".[dev]\"`) — "
                "SAM 3 is now in the default dependencies "
                "(requires Python ≥ 3.12, PyTorch ≥ 2.7, CUDA ≥ 12.6)."
            )
        elif missing == "pkg_resources":
            hint = (
                "The upstream `sam3` package imports `pkg_resources`, which "
                "setuptools ≥ 81 no longer ships. Run `uv pip install "
                "'setuptools<81'` (this is pinned in the default dependencies)."
            )
        else:
            hint = (
                f"`sam3` is installed but a transitive import failed "
                f"({missing or 'unknown'}). The original error is chained "
                f"below — install the missing dependency and retry."
            )
        raise ImportError(f"SAM 3 backend unavailable: {exc}. {hint}") from exc

    if device is None:
        device = get_best_device()

    logger.info("Loading SAM3 model from %s (device=%s)…", checkpoint_path, device)
    device_str = "cuda" if device.type == "cuda" else "cpu"
    build_kwargs: Dict[str, Any] = {
        "enable_inst_interactivity": True,
        "checkpoint_path": checkpoint_path,
        "load_from_HF": False,
        "device": device_str,
    }
    if bpe_path and os.path.isfile(bpe_path):
        build_kwargs["bpe_path"] = bpe_path

    model = build_sam3_image_model(**build_kwargs)

    processor = Sam3Processor(model)

    return {
        "model": model,
        "processor": processor,
        "device": device,
        "checkpoint_path": checkpoint_path,
    }


def _binary_mask_from_predict_inst(masks: Any, scores: Any) -> np.ndarray:
    """Pick the highest-scoring mask returned by ``model.predict_inst``.

    Handles both ``np.ndarray`` and ``torch.Tensor`` outputs. The notebook
    documents shape ``(num_masks, H, W)`` in the single-image case.
    """
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().to(torch.float32).cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().to(torch.float32).cpu().numpy()

    masks = np.asarray(masks)
    scores = np.asarray(scores).reshape(-1)

    if masks.ndim == 4:  # (B, N, H, W) — squeeze batch
        masks = masks[0]
    if masks.ndim == 2:  # already a single (H, W) mask
        return masks.astype(bool)

    if scores.size == 0:
        return masks[0].astype(bool)
    best = int(np.argmax(scores))
    return masks[best].astype(bool)


def _binary_mask_from_text_output(output: Dict[str, Any]) -> np.ndarray:
    """Pick the highest-scoring mask from ``processor.set_text_prompt`` output.

    The README documents the dict ``{"masks", "boxes", "scores"}`` but does
    not pin shapes; we accept both ``(N, H, W)`` and ``(H, W)``.
    """
    masks = output.get("masks")
    scores = output.get("scores")

    if isinstance(masks, torch.Tensor):
        masks = masks.detach().to(torch.float32).cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().to(torch.float32).cpu().numpy()

    masks = np.asarray(masks)
    if masks.ndim == 2:
        return masks.astype(bool)

    if masks.ndim == 4:
        masks = masks[0]
    scores = np.asarray(scores).reshape(-1) if scores is not None else np.array([])
    if scores.size == 0:
        return masks[0].astype(bool)
    best = int(np.argmax(scores))
    return masks[best].astype(bool)


def _score_for_mask(masks: Any, scores: Any) -> float:
    """Return the best score across the SAM3 outputs (for cross-mode arbitration)."""
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().to(torch.float32).cpu().numpy()
    if scores is None:
        return 0.0
    arr = np.asarray(scores).reshape(-1)
    return float(arr.max()) if arr.size else 0.0


def _sam3_prompted(handle: Dict[str, Any], image: Image.Image) -> np.ndarray:
    """Single positive point prompt at the image center (cheapest fallback)."""
    image = image.convert("RGB")
    w, h = image.size
    state = handle["processor"].set_image(image)
    point_coords = np.array([[w // 2, h // 2]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)
    masks, scores, _ = handle["model"].predict_inst(
        state,
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    return _binary_mask_from_predict_inst(masks, scores)


def _sam3_pose_prompted(
    handle: Dict[str, Any],
    image: Image.Image,
    pose_result: Optional[Dict[str, Any]],
    kpt_threshold: float = 0.0,
) -> np.ndarray:
    """Use YOLO pose outputs (highest-confidence detection) as SAM 3 prompts.

    All keypoints with confidence > ``kpt_threshold`` are passed as positive
    points; the bbox is passed as a hard spatial constraint. Falls back to the
    center-point strategy when the pose detector returned no detection.
    """
    if pose_result is None or pose_result.get("num_detections", 0) == 0:
        logger.warning(
            "pose_prompted SAM3 requested but no pose detection — "
            "falling back to center-point prompt."
        )
        return _sam3_prompted(handle, image)

    state = handle["processor"].set_image(image.convert("RGB"))

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

    masks, scores, _ = handle["model"].predict_inst(
        state,
        point_coords=point_coords,
        point_labels=point_labels,
        box=box[None, :],
        multimask_output=False,
    )
    return _binary_mask_from_predict_inst(masks, scores)


def _classification_to_text_prompt(classification_result: Optional[Dict[str, Any]]) -> Optional[str]:
    """Turn a classifier output dict into a SAM 3 text prompt phrase.

    Class names are stored as e.g. ``"Golden_retriever"`` or
    ``"Egyptian_Mau"``. We normalise underscores to spaces and lowercase the
    phrase since SAM 3 was trained on natural-language descriptions.
    """
    if not classification_result:
        return None
    name = classification_result.get("class_name")
    if not name:
        return None
    return name.replace("_", " ").strip().lower()


def _sam3_concept_prompted(
    handle: Dict[str, Any],
    image: Image.Image,
    classification_result: Optional[Dict[str, Any]],
) -> np.ndarray:
    """Use the predicted breed as a text/concept prompt to SAM 3."""
    prompt = _classification_to_text_prompt(classification_result)
    if not prompt:
        logger.warning(
            "concept_prompted SAM3 requested but no class_name available — "
            "falling back to center-point prompt."
        )
        return _sam3_prompted(handle, image)

    state = handle["processor"].set_image(image.convert("RGB"))
    output = handle["processor"].set_text_prompt(state=state, prompt=prompt)
    return _binary_mask_from_text_output(output)


def _sam3_pose_concept_prompted(
    handle: Dict[str, Any],
    image: Image.Image,
    pose_result: Optional[Dict[str, Any]],
    classification_result: Optional[Dict[str, Any]],
    kpt_threshold: float = 0.0,
) -> np.ndarray:
    """Combine visual prompts (pose) and a text prompt (breed).

    Strategy: run both branches when both signals are available and pick the
    candidate with the highest reported score. Gracefully fall back to a
    single branch when one signal is missing.
    """
    has_pose = pose_result is not None and pose_result.get("num_detections", 0) > 0
    text_prompt = _classification_to_text_prompt(classification_result)
    has_text = text_prompt is not None

    if not has_pose and not has_text:
        return _sam3_prompted(handle, image)
    if has_pose and not has_text:
        return _sam3_pose_prompted(handle, image, pose_result, kpt_threshold=kpt_threshold)
    if has_text and not has_pose:
        return _sam3_concept_prompted(handle, image, classification_result)

    state = handle["processor"].set_image(image.convert("RGB"))

    # Branch (A): visual prompts via predict_inst.
    best = int(np.argmax(pose_result["box_confs"]))
    box = pose_result["boxes"][best].astype(np.float32)
    kpts = pose_result["keypoints"][best]
    kpt_confs = pose_result["kpt_confs"][best]
    valid = kpt_confs > kpt_threshold
    point_coords = kpts[valid].astype(np.float32) if valid.any() else None
    point_labels = np.ones(int(valid.sum()), dtype=np.int32) if valid.any() else None

    visual_masks, visual_scores, _ = handle["model"].predict_inst(
        state,
        point_coords=point_coords,
        point_labels=point_labels,
        box=box[None, :],
        multimask_output=False,
    )

    # Branch (B): concept prompt via set_text_prompt.
    text_output = handle["processor"].set_text_prompt(state=state, prompt=text_prompt)

    visual_score = _score_for_mask(visual_masks, visual_scores)
    text_score = _score_for_mask(text_output.get("masks"), text_output.get("scores"))

    if visual_score >= text_score:
        return _binary_mask_from_predict_inst(visual_masks, visual_scores)
    return _binary_mask_from_text_output(text_output)


def sam3_mask_to_trimap(binary_mask: np.ndarray) -> np.ndarray:
    """Convert a boolean SAM 3 mask into a 2-class map (0=fg, 1=bg)."""
    binary_mask = np.asarray(binary_mask, dtype=bool)
    out = np.ones(binary_mask.shape, dtype=np.int64)
    out[binary_mask] = 0
    return out


def predict_segmentation_sam3(
    handle: Dict[str, Any],
    image: Image.Image,
    mode: str = "pose_concept_prompted",
    pose_result: Optional[Dict[str, Any]] = None,
    classification_result: Optional[Dict[str, Any]] = None,
    kpt_threshold: float = 0.0,
) -> Dict[str, Any]:
    """Run SAM 3 segmentation and return the same dict shape as DeepLabV3 / SAM 2."""
    if mode == "prompted":
        binary = _sam3_prompted(handle, image)
    elif mode == "pose_prompted":
        binary = _sam3_pose_prompted(handle, image, pose_result, kpt_threshold=kpt_threshold)
    elif mode == "concept_prompted":
        binary = _sam3_concept_prompted(handle, image, classification_result)
    elif mode == "pose_concept_prompted":
        binary = _sam3_pose_concept_prompted(
            handle, image, pose_result, classification_result,
            kpt_threshold=kpt_threshold,
        )
    else:
        raise ValueError(
            f"Unknown SAM3 mode: {mode!r}. Use 'prompted', 'pose_prompted', "
            "'concept_prompted' or 'pose_concept_prompted'."
        )

    mask = sam3_mask_to_trimap(binary)
    return {
        "mask": mask,
        "mask_small": mask,
        "num_classes": 2,
        "binary_mask": binary,
        "backend": f"sam3_{mode}",
    }
