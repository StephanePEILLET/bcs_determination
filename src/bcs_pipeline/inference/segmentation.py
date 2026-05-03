"""Semantic segmentation inference helpers (DeepLabV3-ResNet50 trimap)."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Tuple

from bcs_pipeline.utils.device import get_best_device

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from bcs_pipeline.lightning_module.segmentation_module import LitSegmentationModule
from bcs_pipeline.inference.classification import IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger("bcs_pipeline")

SEG_IMAGE_SIZE: int = 256

# RGB palette per class. Foreground is rendered green, border yellow,
# background plain white (left transparent in the visualization).
SEG_PALETTE: List[Tuple[int, int, int]] = [
    (0, 200, 0),       # 0 - foreground (animal)
    (255, 255, 255),   # 1 - background
    (255, 200, 0),     # 2 - border
]


def load_segmentation_model(
    checkpoint_path: str,
    num_classes: int = 3,
    device: torch.device | None = None,
) -> LitSegmentationModule:
    """Load a trained DeepLabV3 checkpoint in eval mode.

    Uses ``strict=False`` because checkpoints saved with ``pretrained=True``
    include ``aux_classifier`` weights that may not be present when the model
    is re-created without pretrained weights.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if device is None:
        device = get_best_device()

    logger.info("Loading segmentation model from %s (device=%s)…", checkpoint_path, device)
    model = LitSegmentationModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        map_location=device,
    )
    model.to(device)
    model.eval()
    return model


def get_segmentation_transform(image_size: int = SEG_IMAGE_SIZE) -> transforms.Compose:
    """Resize to (image_size, image_size), ToTensor, ImageNet normalize."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def predict_segmentation(
    model,
    image: Image.Image,
    image_size: int = SEG_IMAGE_SIZE,
    device: torch.device | None = None,
    return_logits: bool = False,
) -> Dict:
    """Predict a trimap mask. The returned mask matches the input image size."""
    if device is None:
        device = next(model.parameters()).device

    transform = get_segmentation_transform(image_size)
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)  # (1, C, H, W)
        pred_small = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()  # (H, W)

    num_classes = int(logits.shape[1])

    # Resize mask back to original image size with NEAREST to preserve labels.
    mask_full = np.array(
        Image.fromarray(pred_small.astype(np.uint8)).resize(image.size, Image.NEAREST),
        dtype=np.int64,
    )

    result = {
        "mask": mask_full,
        "mask_small": pred_small.astype(np.int64),
        "num_classes": num_classes,
    }
    if return_logits:
        result["logits"] = logits.detach().cpu()
    return result


def mask_to_rgb(
    mask: np.ndarray,
    palette: List[Tuple[int, int, int]] = SEG_PALETTE,
) -> np.ndarray:
    """Colorize an integer mask (H, W) into an RGB array (H, W, 3)."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, colour in enumerate(palette):
        rgb[mask == cls_idx] = colour
    return rgb


def load_segmentation_backend(
    backend: str,
    checkpoint_path: str,
    device: torch.device | None = None,
    sam2_config: str | None = None,
    sam3_bpe_path: str | None = None,
):
    """Load a segmentation model for the requested backend.

    Returns an opaque handle: a ``LitSegmentationModule`` for ``"deeplab"``, a
    dict ``{"predictor", "auto_gen", ...}`` for ``"sam2"``, or a dict
    ``{"model", "processor", ...}`` for ``"sam3"``. Use
    :func:`predict_segmentation_with` to dispatch inference uniformly.
    """
    if backend == "deeplab":
        return load_segmentation_model(checkpoint_path, device=device)
    if backend == "sam2":
        from bcs_pipeline.inference.segmentation_sam2 import (
            DEFAULT_SAM2_CONFIG,
            load_sam2_model,
        )
        return load_sam2_model(
            checkpoint_path,
            config_path=sam2_config or DEFAULT_SAM2_CONFIG,
            device=device,
        )
    if backend == "sam3":
        from bcs_pipeline.inference.segmentation_sam3 import load_sam3_model
        return load_sam3_model(checkpoint_path, bpe_path=sam3_bpe_path, device=device)
    raise ValueError(
        f"Unknown segmentation backend: {backend!r}. Use 'deeplab', 'sam2' or 'sam3'."
    )


def predict_segmentation_with(
    backend: str,
    handle,
    image: Image.Image,
    *,
    image_size: int = SEG_IMAGE_SIZE,
    sam2_mode: str = "prompted",
    sam3_mode: str = "pose_concept_prompted",
    pose_result: dict | None = None,
    classification_result: dict | None = None,
    kpt_threshold: float = 0.3,
    device: torch.device | None = None,
) -> Dict:
    """Dispatch ``predict_segmentation`` to the chosen backend.

    Output shape is identical across backends so the downstream visualization
    code is unaware of the backend choice. ``pose_result`` is consumed by
    ``sam2`` (when ``sam2_mode == "pose_prompted"``) and by ``sam3``
    (``pose_prompted`` / ``pose_concept_prompted`` modes).
    ``classification_result`` is consumed by ``sam3`` for the concept-grounded
    text prompt.
    """
    if backend == "deeplab":
        result = predict_segmentation(handle, image, image_size=image_size, device=device)
        result["backend"] = "deeplab"
        return result
    if backend == "sam2":
        from bcs_pipeline.inference.segmentation_sam2 import predict_segmentation_sam2
        return predict_segmentation_sam2(
            handle, image,
            mode=sam2_mode,
            pose_result=pose_result,
            kpt_threshold=kpt_threshold,
        )
    if backend == "sam3":
        from bcs_pipeline.inference.segmentation_sam3 import predict_segmentation_sam3
        # SAM3 defaults to a permissive keypoint threshold (all visible kpts).
        sam3_kpt_threshold = 0.0 if kpt_threshold == 0.3 else kpt_threshold
        return predict_segmentation_sam3(
            handle, image,
            mode=sam3_mode,
            pose_result=pose_result,
            classification_result=classification_result,
            kpt_threshold=sam3_kpt_threshold,
        )
    raise ValueError(
        f"Unknown segmentation backend: {backend!r}. Use 'deeplab', 'sam2' or 'sam3'."
    )
