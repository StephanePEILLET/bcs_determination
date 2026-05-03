"""Inference utilities for the BCS Determination pipeline.

Exposes three task-specific pipelines (classification, segmentation, pose) and a
combined visualization, all re-exported here for backward compatibility with the
README example and ``app.py``.

>>> from bcs_pipeline.inference import load_model, predict_single, run_full_inference
"""

from bcs_pipeline.inference.classification import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_inference_transform,
    load_class_names,
    load_classification_model,
    load_combined_class_names,
    load_oxford_cat_class_names,
    predict_batch,
    predict_single,
)

# Backward-compatible alias (README documents `load_model`).
load_model = load_classification_model

from bcs_pipeline.inference.segmentation import (
    SEG_IMAGE_SIZE,
    SEG_PALETTE,
    get_segmentation_transform,
    load_segmentation_backend,
    load_segmentation_model,
    mask_to_rgb,
    predict_segmentation,
    predict_segmentation_with,
)
from bcs_pipeline.inference.segmentation_sam2 import (
    DEFAULT_SAM2_CONFIG,
    load_sam2_model,
    predict_segmentation_sam2,
    sam2_mask_to_trimap,
)
from bcs_pipeline.inference.segmentation_sam3 import (
    load_sam3_model,
    predict_segmentation_sam3,
    sam3_mask_to_trimap,
)
from bcs_pipeline.inference.pose import (
    load_pose_model,
    predict_pose,
)
from bcs_pipeline.inference.visualization import (
    DEFAULT_OVERLAY_COLORS,
    draw_label_banner,
    draw_pose,
    overlay_segmentation,
    render_combined,
    render_segmentation_layer,
    render_pose_layer,
    save_visualization,
)
from bcs_pipeline.inference.coco_export import (
    build_coco,
    mask_to_polygons,
    save_coco,
)
from bcs_pipeline.inference.pipeline import run_full_inference

__all__ = [
    # classification
    "load_model",
    "load_classification_model",
    "load_class_names",
    "load_combined_class_names",
    "load_oxford_cat_class_names",
    "get_inference_transform",
    "predict_single",
    "predict_batch",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    # segmentation (DeepLabV3)
    "load_segmentation_model",
    "get_segmentation_transform",
    "predict_segmentation",
    "mask_to_rgb",
    "SEG_PALETTE",
    "SEG_IMAGE_SIZE",
    # segmentation backend dispatch
    "load_segmentation_backend",
    "predict_segmentation_with",
    # segmentation (SAM2)
    "load_sam2_model",
    "predict_segmentation_sam2",
    "sam2_mask_to_trimap",
    "DEFAULT_SAM2_CONFIG",
    # segmentation (SAM3)
    "load_sam3_model",
    "predict_segmentation_sam3",
    "sam3_mask_to_trimap",
    # pose
    "load_pose_model",
    "predict_pose",
    # visualization
    "overlay_segmentation",
    "draw_pose",
    "draw_label_banner",
    "render_combined",
    "render_segmentation_layer",
    "render_pose_layer",
    "save_visualization",
    "DEFAULT_OVERLAY_COLORS",
    # pipeline
    "run_full_inference",
    # COCO export
    "build_coco",
    "mask_to_polygons",
    "save_coco",
]
