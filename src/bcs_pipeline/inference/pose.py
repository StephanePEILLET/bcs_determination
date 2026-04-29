"""Pose detection inference helpers (Ultralytics YOLO custom-trained on dogs)."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger("bcs_pipeline")


def load_pose_model(
    checkpoint_path: str,
    device: Optional[str] = None,
):
    """Load an Ultralytics YOLO pose model from a ``.pt`` file."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Pose checkpoint not found: {checkpoint_path}")

    from ultralytics import YOLO

    logger.info("Loading pose model from %s…", checkpoint_path)
    model = YOLO(checkpoint_path)
    if device is not None:
        try:
            model.to(device)
        except Exception:
            # Some ultralytics builds manage device only at predict time.
            pass
    return model


def predict_pose(
    model,
    image: Image.Image,
    conf_threshold: float = 0.25,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Run pose detection on a single PIL image.

    Returns a dict with bounding boxes, keypoints, and the raw Results object.
    On no detection, returns empty arrays and ``num_detections=0``.
    """
    img_arr = np.array(image)

    kwargs = {"conf": conf_threshold, "verbose": False}
    if device is not None:
        kwargs["device"] = device
    results = model(img_arr, **kwargs)
    r = results[0]

    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        box_confs = r.boxes.conf.cpu().numpy().astype(np.float32)
        box_classes = r.boxes.cls.cpu().numpy().astype(int)
    else:
        boxes = np.zeros((0, 4), dtype=np.float32)
        box_confs = np.zeros((0,), dtype=np.float32)
        box_classes = np.zeros((0,), dtype=int)

    if r.keypoints is not None and r.keypoints.xy is not None and len(r.keypoints.xy) > 0:
        keypoints = r.keypoints.xy.cpu().numpy().astype(np.float32)  # (N, K, 2)
        if r.keypoints.conf is not None:
            kpt_confs = r.keypoints.conf.cpu().numpy().astype(np.float32)  # (N, K)
        else:
            kpt_confs = np.ones(keypoints.shape[:2], dtype=np.float32)
    else:
        keypoints = np.zeros((0, 0, 2), dtype=np.float32)
        kpt_confs = np.zeros((0, 0), dtype=np.float32)

    return {
        "boxes": boxes,
        "box_confs": box_confs,
        "box_classes": box_classes,
        "keypoints": keypoints,
        "kpt_confs": kpt_confs,
        "num_detections": int(boxes.shape[0]),
        "raw": r,
    }
