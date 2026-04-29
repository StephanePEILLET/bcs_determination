"""Unified inference orchestrator: classification + segmentation + pose."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from PIL import Image

from bcs_pipeline.inference.classification import (
    load_classification_model,
    load_class_names,
    predict_single,
)
from bcs_pipeline.inference.segmentation import (
    load_segmentation_backend,
    predict_segmentation_with,
)
from bcs_pipeline.inference.pose import load_pose_model, predict_pose
from bcs_pipeline.inference.visualization import render_combined, save_visualization
from bcs_pipeline.inference.coco_export import build_coco, save_coco

logger = logging.getLogger("bcs_pipeline")


def _open_image(image: Union[str, Path, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(str(image)).convert("RGB")


def run_full_inference(
    image: Union[str, Path, Image.Image],
    *,
    classification_ckpt: Optional[str] = None,
    segmentation_ckpt: Optional[str] = None,
    pose_ckpt: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    classification_model_name: str = "resnet50",
    classification_num_classes: int = 120,
    classification_image_size: int = 224,
    segmentation_image_size: int = 256,
    segmentation_backend: str = "deeplab",
    sam2_mode: str = "prompted",
    sam2_config: Optional[str] = None,
    data_dir: Optional[str] = None,
    top_k: int = 5,
    pose_conf_threshold: float = 0.25,
    coco_path: Optional[Union[str, Path]] = None,
    polygon_simplify: float = 2.0,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run the three pipelines whose checkpoints are provided and (optionally)
    save a combined PNG visualization plus a COCO-format JSON sidecar.

    Returns a dict ``{"classification", "segmentation", "pose", "output_path",
    "coco", "coco_path"}`` where missing branches are ``None``. ``coco`` is
    always built when at least one branch produced output.
    """
    pil_image = _open_image(image)

    classification_result: Optional[Dict] = None
    if classification_ckpt:
        cls_model = load_classification_model(
            classification_ckpt,
            model_name=classification_model_name,
            num_classes=classification_num_classes,
            device=device,
        )
        class_names = load_class_names(data_dir) if data_dir else None
        classification_result = predict_single(
            cls_model,
            pil_image,
            image_size=classification_image_size,
            class_names=class_names,
            top_k=top_k,
        )

    # When sam2 pose_prompted mode is requested, the pose result must be
    # available *before* segmentation runs.
    needs_pose_first = (
        segmentation_backend == "sam2"
        and sam2_mode == "pose_prompted"
        and pose_ckpt is not None
    )

    pose_result: Optional[Dict] = None
    if pose_ckpt and needs_pose_first:
        pose_model = load_pose_model(pose_ckpt)
        pose_result = predict_pose(
            pose_model, pil_image, conf_threshold=pose_conf_threshold,
        )

    segmentation_result: Optional[Dict] = None
    if segmentation_ckpt:
        seg_handle = load_segmentation_backend(
            segmentation_backend,
            segmentation_ckpt,
            device=device,
            sam2_config=sam2_config,
        )
        segmentation_result = predict_segmentation_with(
            segmentation_backend,
            seg_handle,
            pil_image,
            image_size=segmentation_image_size,
            sam2_mode=sam2_mode,
            pose_result=pose_result,
            device=device,
        )

    if pose_ckpt and pose_result is None:
        pose_model = load_pose_model(pose_ckpt)
        pose_result = predict_pose(
            pose_model, pil_image, conf_threshold=pose_conf_threshold,
        )

    saved_path: Optional[str] = None
    if output_path:
        composed = render_combined(
            pil_image,
            classification=classification_result,
            segmentation=segmentation_result,
            pose=pose_result,
        )
        saved_path = str(save_visualization(composed, output_path))
    elif not any([classification_result, segmentation_result, pose_result]):
        logger.warning("run_full_inference called without any checkpoint — nothing to do.")

    coco_dict: Optional[Dict[str, Any]] = None
    coco_saved_path: Optional[str] = None
    if any([classification_result, segmentation_result, pose_result]):
        coco_dict = build_coco(
            pil_image,
            image_path=str(image) if not isinstance(image, Image.Image) else None,
            classification=classification_result,
            segmentation=segmentation_result,
            pose=pose_result,
            polygon_simplify=polygon_simplify,
        )
        if coco_path:
            coco_saved_path = str(save_coco(coco_dict, coco_path))

    return {
        "classification": classification_result,
        "segmentation": segmentation_result,
        "pose": pose_result,
        "output_path": saved_path,
        "coco": coco_dict,
        "coco_path": coco_saved_path,
    }
