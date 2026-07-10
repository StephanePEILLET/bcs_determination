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
    load_cat_class_names,
    load_dog_class_names,
    predict_single,
)
from bcs_pipeline.inference.segmentation import (
    load_segmentation_backend,
    predict_segmentation_with,
)
from bcs_pipeline.inference.pose import load_pose_model, predict_pose
from bcs_pipeline.inference.bcs import load_bcs_model, load_bcs_models, predict_bcs
from bcs_pipeline.inference.species import load_species_model, predict_species
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
    bcs_ckpt: Optional[Union[str, Path]] = None,
    bcs_sex: Optional[str] = None,
    bcs_long_coat: Optional[object] = None,
    species_ckpt: Optional[str] = None,
    species_model_name: str = "vit",
    dog_breed_ckpt: Optional[str] = None,
    dog_breed_num_classes: int = 120,
    cat_breed_ckpt: Optional[str] = None,
    cat_breed_num_classes: int = 12,
    stanford_data_dir: Optional[str] = None,
    oxford_data_dir: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    classification_model_name: str = "resnet50",
    classification_num_classes: int = 120,
    classification_image_size: int = 224,
    segmentation_image_size: int = 256,
    segmentation_backend: str = "deeplab",
    sam2_mode: str = "prompted",
    sam2_config: Optional[str] = None,
    sam3_mode: str = "pose_concept_prompted",
    sam3_bpe_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    top_k: int = 5,
    pose_conf_threshold: float = 0.25,
    coco_path: Optional[Union[str, Path]] = None,
    polygon_simplify: float = 2.0,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run the cascade pipelines whose checkpoints are provided and (optionally)
    save a combined PNG visualization plus a COCO-format JSON sidecar.

    The cascade is: species → breed (routed by species) → segmentation → BCS
    (routed by species). Returns a dict ``{"classification", "segmentation",
    "pose", "bcs", "species", "output_path", "coco", "coco_path"}`` where missing
    branches are ``None``. ``coco`` is always built when at least one branch
    produced output. The ``bcs`` branch runs only when ``bcs_ckpt`` is given; it
    consumes the segmentation mask (when available) to isolate the silhouette
    before scoring, and routes to the dog/cat model by predicted species.
    """
    pil_image = _open_image(image)

    # ── Cascade stage 1: species (optional) ──────────────────────────
    species_result: Optional[Dict] = None
    if species_ckpt:
        species_model = load_species_model(
            species_ckpt, model_name=species_model_name, device=device,
        )
        species_result = predict_species(species_model, pil_image)
    species_name = species_result["species"] if species_result else None

    # ── Cascade stage 2: breed, routed by species when a dedicated model exists ──
    classification_result: Optional[Dict] = None
    if species_name == "dog" and dog_breed_ckpt:
        breed_model = load_classification_model(
            dog_breed_ckpt, model_name=classification_model_name,
            num_classes=dog_breed_num_classes, device=device,
        )
        breed_names = load_dog_class_names(stanford_data_dir) if stanford_data_dir else None
        classification_result = predict_single(
            breed_model, pil_image, image_size=classification_image_size,
            class_names=breed_names, top_k=top_k,
        )
    elif species_name == "cat" and cat_breed_ckpt:
        breed_model = load_classification_model(
            cat_breed_ckpt, model_name=classification_model_name,
            num_classes=cat_breed_num_classes, device=device,
        )
        breed_names = load_cat_class_names(oxford_data_dir) if oxford_data_dir else None
        classification_result = predict_single(
            breed_model, pil_image, image_size=classification_image_size,
            class_names=breed_names, top_k=top_k,
        )
    elif classification_ckpt:
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
    if classification_result is not None and species_name is not None:
        classification_result["species"] = species_name


    # Pose detection must run *before* segmentation when the chosen backend
    # consumes pose prompts (SAM 2 pose_prompted, or SAM 3 pose_/pose_concept_prompted).
    needs_pose_first = pose_ckpt is not None and (
        (segmentation_backend == "sam2" and sam2_mode == "pose_prompted")
        or (
            segmentation_backend == "sam3"
            and sam3_mode in {"pose_prompted", "pose_concept_prompted"}
        )
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
            sam3_bpe_path=sam3_bpe_path,
        )
        segmentation_result = predict_segmentation_with(
            segmentation_backend,
            seg_handle,
            pil_image,
            image_size=segmentation_image_size,
            sam2_mode=sam2_mode,
            sam3_mode=sam3_mode,
            pose_result=pose_result,
            classification_result=classification_result,
            device=device,
        )

    if pose_ckpt and pose_result is None:
        pose_model = load_pose_model(pose_ckpt)
        pose_result = predict_pose(
            pose_model, pil_image, conf_threshold=pose_conf_threshold,
        )

    # BCS regression: consume the segmentation mask (silhouette) when present,
    # routing to the dog/cat model by predicted species. ``bcs_ckpt`` may be a
    # base dir with cat/dog sub-folders, a per-species dir, or a single ckpt.
    bcs_result: Optional[Dict] = None
    if bcs_ckpt:
        seg_mask = segmentation_result.get("mask") if segmentation_result else None
        bcs_base = Path(bcs_ckpt)
        if (bcs_base / "cat").is_dir() or (bcs_base / "dog").is_dir():
            registry = load_bcs_models(bcs_base, device=device)
            species_key = species_name or "cat"
            bcs_handle = registry.get(species_key)
            # Fall back to the cat model only when the species is *unknown*
            # (no species classifier ran). A recognised species without its own
            # model (e.g. dog) must NOT be scored by the cat model — it falls
            # through to the "unavailable" branch below.
            if bcs_handle is None and species_name is None:
                bcs_handle = registry.get("cat")
            if bcs_handle is not None:
                bcs_result = predict_bcs(
                    bcs_handle, pil_image, mask=seg_mask,
                    sex=bcs_sex, long_coat=bcs_long_coat, device=device,
                )
                bcs_result["model_species"] = (
                    species_key if registry.get(species_key) else "cat"
                )
            elif species_key == "dog":
                bcs_result = {"bcs": None, "category": None, "unavailable_for": "dog"}
        else:
            bcs_handle = load_bcs_model(bcs_ckpt, device=device)
            bcs_result = predict_bcs(
                bcs_handle, pil_image, mask=seg_mask,
                sex=bcs_sex, long_coat=bcs_long_coat, device=device,
            )


    saved_path: Optional[str] = None
    if output_path:
        composed = render_combined(
            pil_image,
            classification=classification_result,
            segmentation=segmentation_result,
            pose=pose_result,
            bcs=bcs_result,
        )
        saved_path = str(save_visualization(composed, output_path))
    elif not any([classification_result, segmentation_result, pose_result, bcs_result]):
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
        "bcs": bcs_result,
        "species": species_result,
        "output_path": saved_path,
        "coco": coco_dict,
        "coco_path": coco_saved_path,
    }
