#!/usr/bin/env python
"""Pre-load the SQLite database with inference results for all dataset images.

Scans Stanford Dogs, Oxford-IIIT Pet, and Reddit example directories, runs
classification + segmentation + pose inference on every image, and persists
the results via ``save_run()`` so that the web app can serve them instantly.

The script is **idempotent**: images already present in the database (matched
by ``image_name + dataset + group_name + seg_backend``) are skipped unless
``--force`` is passed.

Usage
-----
.. code-block:: bash

    python scripts/preload_db.py
    python scripts/preload_db.py --seg-backend sam2 --sam2-mode automatic
    python scripts/preload_db.py --force
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bcs_pipeline.db import (
    InferenceRun,
    init_db,
    save_run,
)
from bcs_pipeline.inference import (
    load_classification_model,
    load_combined_class_names,
    load_pose_model,
    load_segmentation_backend,
    predict_pose,
    predict_segmentation_with,
    predict_single,
    render_combined,
    render_segmentation_layer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
)
logger = logging.getLogger("preload_db")

REPO_ROOT = Path(__file__).resolve().parent.parent

CLASSIFICATION_CKPT_DIR = REPO_ROOT / "checkpoints" / "classification" / "resnet50_dogs_cats"
CLASSIFICATION_NUM_CLASSES = 132
DEEPLAB_CKPT = REPO_ROOT / "checkpoints" / "segmentation" / "deeplabv3_resnet50_last-v1.ckpt"
SAM2_CKPT = REPO_ROOT / "checkpoints" / "sam2.1_hiera_large.pt"
POSE_CKPT = REPO_ROOT / "checkpoints" / "pose" / "yolo_best.pt"

STANFORD_ROOT = REPO_ROOT / "data" / "stanford_dogs" / "images"
STANFORD_IMAGES = STANFORD_ROOT / "Images"
OXFORD_ROOT = REPO_ROOT / "data" / "Oxford-IIIT_pet_dataset"
OXFORD_IMAGES = OXFORD_ROOT / "images"
REDDIT_DIR = REPO_ROOT / "data" / "Reddit_example"

OUTPUT_DIR = REPO_ROOT / "data" / "outputs"
DEFAULT_DB_PATH = REPO_ROOT / "data" / "bcs_app.db"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

STATS = {"processed": 0, "skipped": 0, "errors": 0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-load the database with inference results for all dataset images.",
    )
    parser.add_argument(
        "--seg-backend",
        choices=["deeplab", "sam2"],
        default="deeplab",
        help="Segmentation backend (default: deeplab).",
    )
    parser.add_argument(
        "--sam2-mode",
        choices=["prompted", "automatic", "pose_prompted"],
        default="prompted",
        help="SAM2 prompting mode (default: prompted).",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of top classification predictions (default: 5).",
    )
    parser.add_argument(
        "--conf-threshold", type=float, default=0.25,
        help="YOLO confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--db-path", type=str, default=str(DEFAULT_DB_PATH),
        help="Path to the SQLite database file.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process images already present in the database.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Only process these datasets (e.g. --datasets Reddit 'Stanford Dogs'). "
             "Default: all datasets.",
    )
    return parser.parse_args()


def _resolve_classifier_ckpt() -> Optional[Path]:
    if not CLASSIFICATION_CKPT_DIR.is_dir():
        return None
    last = CLASSIFICATION_CKPT_DIR / "last.ckpt"
    if last.is_file():
        return last
    candidates = sorted(
        CLASSIFICATION_CKPT_DIR.glob("*.ckpt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _list_images(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    )


def _collect_stanford() -> List[Tuple[Path, str, str, Optional[str]]]:
    """Return [(image_path, dataset, group_name, ground_truth), ...]."""
    entries: List[Tuple[Path, str, str, Optional[str]]] = []
    if not STANFORD_IMAGES.is_dir():
        logger.warning("Stanford Dogs directory not found: %s", STANFORD_IMAGES)
        return entries
    for breed_dir in sorted(p for p in STANFORD_IMAGES.iterdir() if p.is_dir()):
        breed_name = (
            breed_dir.name.split("-", 1)[1]
            if "-" in breed_dir.name
            else breed_dir.name
        )
        for img_path in _list_images(breed_dir):
            entries.append((img_path, "Stanford Dogs", breed_name, breed_name))
    return entries


def _collect_oxford() -> List[Tuple[Path, str, str, Optional[str]]]:
    entries: List[Tuple[Path, str, str, Optional[str]]] = []
    if not OXFORD_IMAGES.is_dir():
        logger.warning("Oxford-IIIT Pet images directory not found: %s", OXFORD_IMAGES)
        return entries
    for img_path in _list_images(OXFORD_IMAGES):
        prefix = "_".join(img_path.stem.split("_")[:-1])
        ground_truth = prefix.replace("_", " ")
        entries.append((img_path, "Oxford-IIIT Pet", prefix, ground_truth))
    return entries


def _collect_reddit() -> List[Tuple[Path, str, str, Optional[str]]]:
    entries: List[Tuple[Path, str, str, Optional[str]]] = []
    if not REDDIT_DIR.is_dir():
        logger.warning("Reddit example directory not found: %s", REDDIT_DIR)
        return entries
    for img_path in _list_images(REDDIT_DIR):
        entries.append((img_path, "Reddit", "all", None))
    return entries


def _is_already_processed(session, image_name: str, dataset: str,
                          group_name: str, seg_backend: str) -> bool:
    q = (
        session.query(InferenceRun)
        .filter(
            InferenceRun.image_name == image_name,
            InferenceRun.dataset == dataset,
            InferenceRun.group_name == group_name,
            InferenceRun.seg_backend == seg_backend,
        )
        .limit(1)
    )
    return session.query(q.exists()).scalar()


def _run_inference(
    img,
    image_name: str,
    cls_model,
    class_names,
    seg_handle,
    seg_backend: str,
    pose_model,
    sam2_mode: str = "prompted",
    top_k: int = 5,
    conf_threshold: float = 0.25,
) -> Dict[str, Any]:
    from PIL import Image

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

    mask = seg["mask"]
    unique, counts = np.unique(mask, return_counts=True)
    seg_dist = [
        {"class": int(u), "pct": round(float(c) / mask.size * 100, 1)}
        for u, c in zip(unique, counts)
    ]

    return {
        "image_name": image_name,
        "image_size": list(img.size),
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
    }


def main() -> None:
    args = parse_args()
    start_time = time.time()

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║       🐾  Body Pawsitive — Pre-load Database        ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    engine, session_local = init_db(args.db_path)
    session = session_local()

    filter_datasets = set(args.datasets) if args.datasets else None

    all_entries = []
    for collector, name in [
        (_collect_stanford, "Stanford Dogs"),
        (_collect_oxford, "Oxford-IIIT Pet"),
        (_collect_reddit, "Reddit"),
    ]:
        if filter_datasets and name not in filter_datasets:
            logger.info("Skipping dataset: %s", name)
            continue
        entries = collector()
        logger.info("  %-20s → %d images", name, len(entries))
        all_entries.extend(entries)

    if not all_entries:
        logger.warning("No images found. Check that data directories are populated.")
        session.close()
        engine.dispose()
        return

    total = len(all_entries)
    print(f"\n  Total images to process: {total}")
    print(f"  Segmentation backend   : {args.seg_backend}")
    print(f"  SAM2 mode              : {args.sam2_mode}")
    print()

    # ── Load models ──────────────────────────────────────────────────────────
    cls_model = None
    class_names = None
    seg_handle = None
    pose_model = None

    ckpt = _resolve_classifier_ckpt()
    if ckpt is not None:
        logger.info("Loading classification model from %s ...", ckpt)
        cls_model = load_classification_model(str(ckpt), num_classes=CLASSIFICATION_NUM_CLASSES)
        class_names = load_combined_class_names(str(STANFORD_ROOT), str(OXFORD_ROOT))
        logger.info("Classification model loaded (%d classes).", len(class_names) if class_names else 0)
    else:
        logger.warning("No classification checkpoint found — skipping classification.")

    seg_ckpt_path = str(SAM2_CKPT) if args.seg_backend == "sam2" else str(DEEPLAB_CKPT)
    seg_ckpt_file = SAM2_CKPT if args.seg_backend == "sam2" else DEEPLAB_CKPT
    if seg_ckpt_file.is_file():
        logger.info("Loading segmentation model (%s) from %s ...", args.seg_backend, seg_ckpt_file)
        seg_handle = load_segmentation_backend(args.seg_backend, seg_ckpt_path)
        logger.info("Segmentation model loaded.")
    else:
        logger.warning("Segmentation checkpoint not found (%s) — skipping segmentation.", seg_ckpt_file)

    if POSE_CKPT.is_file():
        logger.info("Loading pose model from %s ...", POSE_CKPT)
        pose_model = load_pose_model(str(POSE_CKPT))
        logger.info("Pose model loaded.")
    else:
        logger.warning("Pose checkpoint not found (%s) — skipping pose.", POSE_CKPT)

    if cls_model is None and seg_handle is None and pose_model is None:
        logger.error("No models could be loaded — nothing to do.")
        session.close()
        engine.dispose()
        sys.exit(1)

    # ── Process images ───────────────────────────────────────────────────────
    from PIL import Image

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print()
    for idx, (img_path, dataset, group_name, ground_truth) in enumerate(all_entries, 1):
        image_name = img_path.name
        tag = f"[{idx}/{total}]"

        if not args.force and _is_already_processed(
            session, image_name, dataset, group_name, args.seg_backend
        ):
            STATS["skipped"] += 1
            continue

        try:
            img = Image.open(str(img_path)).convert("RGB")
        except Exception:
            logger.exception("%s Cannot open image: %s", tag, img_path)
            STATS["errors"] += 1
            continue

        try:
            result = _run_inference(
                img, image_name,
                cls_model=cls_model,
                class_names=class_names,
                seg_handle=seg_handle,
                seg_backend=args.seg_backend,
                pose_model=pose_model,
                sam2_mode=args.sam2_mode,
                top_k=args.top_k,
                conf_threshold=args.conf_threshold,
            )

            save_run(
                session, OUTPUT_DIR, result,
                source_type="dataset",
                dataset=dataset,
                group_name=group_name,
                ground_truth=ground_truth,
                seg_backend=args.seg_backend,
                sam2_mode=args.sam2_mode,
            )

            STATS["processed"] += 1
            if STATS["processed"] % 50 == 0 or idx == total:
                logger.info(
                    "%s %s → %s (%.1f%% done)",
                    tag, image_name, result["classification"].get("class_name", "?"),
                    idx / total * 100,
                )

        except Exception:
            logger.exception("%s Inference failed: %s", tag, img_path)
            STATS["errors"] += 1
            session.rollback()

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    session.close()
    engine.dispose()

    print(f"\n{'━' * 50}")
    print(f"  Pre-load complete in {elapsed:.1f}s")
    print(f"  Processed : {STATS['processed']}")
    print(f"  Skipped   : {STATS['skipped']} (already in DB)")
    print(f"  Errors    : {STATS['errors']}")
    print(f"  Database  : {args.db_path}")
    print(f"{'━' * 50}\n")


if __name__ == "__main__":
    main()
