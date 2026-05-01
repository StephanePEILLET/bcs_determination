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
import logging
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bcs_pipeline.app_checkpoints import (
    CLASSIFICATION_CKPT_DIR,
    CLASSIFICATION_MODEL_NAME,
    CLASSIFICATION_NUM_CLASSES,
    DEEPLAB_CKPT,
    POSE_CKPT,
    REPO_ROOT,
    SAM2_CKPT,
    resolve_classifier_ckpt,
)
from bcs_pipeline.datasets import (
    STANFORD_ROOT,
    OXFORD_ROOT,
    collect_all_images,
)
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
)
from bcs_pipeline.inference_format import (
    format_inference_result,
    run_core_inference,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
)
logger = logging.getLogger("preload_db")

OUTPUT_DIR = REPO_ROOT / "data" / "outputs"
DEFAULT_DB_PATH = REPO_ROOT / "data" / "bcs_app.db"

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
) -> dict:
    from PIL import Image

    img = img.convert("RGB")
    cls, seg, pose = run_core_inference(
        cls_model, class_names, seg_handle, seg_backend, pose_model, img,
        sam2_mode=sam2_mode, top_k=top_k, conf_threshold=conf_threshold,
    )
    return format_inference_result(cls, seg, pose, image_name, img.size, seg_backend)


def main() -> None:
    args = parse_args()
    start_time = time.time()

    print("\n\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557")
    print("\u2551       \U0001F43E  Body Pawsitive \u2014 Pre-load Database        \u2551")
    print("\u255A\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D\n")

    engine, session_local = init_db(args.db_path)
    session = session_local()

    filter_datasets = set(args.datasets) if args.datasets else None

    all_entries = collect_all_images()

    if filter_datasets:
        all_entries = [e for e in all_entries if e[1] in filter_datasets]

    for name in ["Stanford Dogs", "Oxford-IIIT Pet", "Reddit"]:
        count = sum(1 for e in all_entries if e[1] == name)
        logger.info("  %-20s \u2192 %d images", name, count)

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

    cls_model = None
    class_names = None
    seg_handle = None
    pose_model = None

    ckpt = resolve_classifier_ckpt()
    if ckpt is not None:
        logger.info("Loading classification model (%s, %d classes) from %s ...",
                    CLASSIFICATION_MODEL_NAME, CLASSIFICATION_NUM_CLASSES,
                    ckpt.relative_to(REPO_ROOT))
        cls_model = load_classification_model(
            str(ckpt),
            model_name=CLASSIFICATION_MODEL_NAME,
            num_classes=CLASSIFICATION_NUM_CLASSES,
        )
        class_names = load_combined_class_names(str(STANFORD_ROOT), str(OXFORD_ROOT))
        logger.info("Classification model loaded (%d classes).", len(class_names) if class_names else 0)
    else:
        logger.warning("No classification checkpoint found in %s \u2014 skipping classification.",
                       CLASSIFICATION_CKPT_DIR)

    seg_ckpt_path = str(SAM2_CKPT) if args.seg_backend == "sam2" else str(DEEPLAB_CKPT)
    seg_ckpt_file = SAM2_CKPT if args.seg_backend == "sam2" else DEEPLAB_CKPT
    if seg_ckpt_file.is_file():
        logger.info("Loading segmentation model (%s) from %s ...", args.seg_backend, seg_ckpt_file)
        seg_handle = load_segmentation_backend(args.seg_backend, seg_ckpt_path)
        logger.info("Segmentation model loaded.")
    else:
        logger.warning("Segmentation checkpoint not found (%s) \u2014 skipping segmentation.", seg_ckpt_file)

    if POSE_CKPT.is_file():
        logger.info("Loading pose model from %s ...", POSE_CKPT)
        pose_model = load_pose_model(str(POSE_CKPT))
        logger.info("Pose model loaded.")
    else:
        logger.warning("Pose checkpoint not found (%s) \u2014 skipping pose.", POSE_CKPT)

    if cls_model is None and seg_handle is None and pose_model is None:
        logger.error("No models could be loaded \u2014 nothing to do.")
        session.close()
        engine.dispose()
        sys.exit(1)

    from PIL import Image

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print()
    for idx, (img_path, dataset, group_name, gt) in enumerate(all_entries, 1):
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
                ground_truth=gt,
                seg_backend=args.seg_backend,
                sam2_mode=args.sam2_mode,
            )

            STATS["processed"] += 1
            if STATS["processed"] % 50 == 0 or idx == total:
                logger.info(
                    "%s %s \u2192 %s (%.1f%% done)",
                    tag, image_name, result["classification"].get("class_name", "?"),
                    idx / total * 100,
                )

        except Exception:
            logger.exception("%s Inference failed: %s", tag, img_path)
            STATS["errors"] += 1
            session.rollback()

    elapsed = time.time() - start_time
    session.close()
    engine.dispose()

    print(f"\n{'\u2501' * 50}")
    print(f"  Pre-load complete in {elapsed:.1f}s")
    print(f"  Processed : {STATS['processed']}")
    print(f"  Skipped   : {STATS['skipped']} (already in DB)")
    print(f"  Errors    : {STATS['errors']}")
    print(f"  Database  : {args.db_path}")
    print(f"{'\u2501' * 50}\n")


if __name__ == "__main__":
    main()
