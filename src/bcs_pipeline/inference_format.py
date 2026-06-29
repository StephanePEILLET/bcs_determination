"""Shared inference orchestration and result formatting.

Provides the core logic for running classification + segmentation + pose
pipelines and formatting the result dict — used by both ``app.py`` and
``scripts/preload_db.py``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from bcs_pipeline.inference import (
    predict_pose,
    predict_segmentation_with,
    predict_single,
    predict_species,
)


def run_core_inference(
    cls_model,
    class_names,
    seg_handle,
    seg_backend: str,
    pose_model,
    img: Image.Image,
    sam2_mode: str = "prompted",
    sam3_mode: str = "pose_concept_prompted",
    top_k: int = 5,
    conf_threshold: float = 0.25,
    species_model=None,
    dog_breed: Optional[Tuple[Any, list]] = None,
    cat_breed: Optional[Tuple[Any, list]] = None,
) -> tuple:
    """Run the cascade: species → breed (routed) → pose? → segmentation.

    Parameters
    ----------
    cls_model, class_names :
        Fallback (combined) breed classifier, used when no species-specific
        breed model is available for the predicted species.
    species_model :
        Optional binary dog/cat classifier (cascade stage 1). When provided, its
        prediction routes the breed classifier (and, downstream, the BCS model).
    dog_breed, cat_breed :
        Optional ``(model, class_names)`` tuples for the dedicated dog-only and
        cat-only breed classifiers.

    Returns ``(cls, seg, pose, species)`` where ``species`` is the species dict
    (``{"species", "confidence", ...}``) or ``None`` when no species model ran.
    """
    img = img.convert("RGB")

    # ── Stage 1: species (optional) ──────────────────────────────────
    species = predict_species(species_model, img) if species_model is not None else None

    # ── Stage 2: breed, routed by species when a dedicated model exists ──
    routed = None
    if species is not None:
        if species["species"] == "dog" and dog_breed is not None:
            routed = dog_breed
        elif species["species"] == "cat" and cat_breed is not None:
            routed = cat_breed

    if routed is not None:
        breed_model, breed_names = routed
        cls = predict_single(breed_model, img, class_names=breed_names, top_k=top_k)
    else:
        cls = predict_single(cls_model, img, class_names=class_names, top_k=top_k)

    if species is not None:
        cls["species"] = species["species"]

    needs_pose_first = (
        (seg_backend == "sam2" and sam2_mode == "pose_prompted")
        or (
            seg_backend == "sam3"
            and sam3_mode in {"pose_prompted", "pose_concept_prompted"}
        )
    )
    pose = predict_pose(pose_model, img, conf_threshold=conf_threshold) if needs_pose_first else None

    seg = predict_segmentation_with(
        seg_backend, seg_handle, img,
        sam2_mode=sam2_mode,
        sam3_mode=sam3_mode,
        pose_result=pose,
        classification_result=cls,
    )

    if pose is None:
        pose = predict_pose(pose_model, img, conf_threshold=conf_threshold)

    return cls, seg, pose, species



def format_inference_result(
    cls: Dict[str, Any],
    seg: Dict[str, Any],
    pose: Dict[str, Any],
    image_name: str,
    image_size: tuple,
    seg_backend: str = "deeplab",
    bcs: Optional[Dict[str, Any]] = None,
    species: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    mask = seg["mask"]
    unique, counts = np.unique(mask, return_counts=True)
    seg_dist = [
        {"class": int(u), "pct": round(float(c) / mask.size * 100, 1)}
        for u, c in zip(unique, counts)
    ]

    return {
        "bcs": bcs,
        "species": (
            {
                "species": species.get("species"),
                "confidence": round(species.get("confidence", 0.0) * 100, 2),
            }
            if species
            else None
        ),
        "classification": {
            "class_name": cls.get("class_name"),
            "species": cls.get("species"),
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
