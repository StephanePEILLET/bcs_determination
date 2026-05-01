"""Shared inference orchestration and result formatting.

Provides the core logic for running classification + segmentation + pose
pipelines and formatting the result dict — used by both ``app.py`` and
``scripts/preload_db.py``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from bcs_pipeline.inference import (
    predict_pose,
    predict_segmentation_with,
    predict_single,
)


def run_core_inference(
    cls_model,
    class_names,
    seg_handle,
    seg_backend: str,
    pose_model,
    img: Image.Image,
    sam2_mode: str = "prompted",
    top_k: int = 5,
    conf_threshold: float = 0.25,
) -> tuple:
    img = img.convert("RGB")

    cls = predict_single(cls_model, img, class_names=class_names, top_k=top_k)

    needs_pose_first = seg_backend == "sam2" and sam2_mode == "pose_prompted"
    pose = predict_pose(pose_model, img, conf_threshold=conf_threshold) if needs_pose_first else None

    seg = predict_segmentation_with(
        seg_backend, seg_handle, img,
        sam2_mode=sam2_mode, pose_result=pose,
    )

    if pose is None:
        pose = predict_pose(pose_model, img, conf_threshold=conf_threshold)

    return cls, seg, pose


def format_inference_result(
    cls: Dict[str, Any],
    seg: Dict[str, Any],
    pose: Dict[str, Any],
    image_name: str,
    image_size: tuple,
    seg_backend: str = "deeplab",
) -> Dict[str, Any]:
    mask = seg["mask"]
    unique, counts = np.unique(mask, return_counts=True)
    seg_dist = [
        {"class": int(u), "pct": round(float(c) / mask.size * 100, 1)}
        for u, c in zip(unique, counts)
    ]

    return {
        "classification": {
            "class_name": cls.get("class_name"),
            "confidence": round(cls.get("confidence", 0.0) * 100, 2),
            "top_k": [
                {
                    "rank": i + 1,
                    "class_name": e.get("class_name"),
                    "confidence": round(e.get("confidence", 0.0) * 100, 2),
                }
                for i, e in enumerate(cls.get("top_k", []))
            ],
        },
        "segmentation": {
            "backend": seg.get("backend", seg_backend),
            "distribution": seg_dist,
        },
        "pose": {
            "num_detections": pose.get("num_detections", 0),
            "best_conf": round(float(pose["box_confs"][0]), 2) if pose.get("num_detections", 0) > 0 else None,
        },
        "pose_annotations": {
            "boxes": pose["boxes"].tolist(),
            "keypoints": pose["keypoints"].tolist(),
            "kpt_confs": pose["kpt_confs"].tolist(),
            "box_confs": pose["box_confs"].tolist(),
        },
        "image_size": list(image_size),
        "image_name": image_name,
    }
