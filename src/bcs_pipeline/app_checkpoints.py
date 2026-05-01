"""Single source of truth for the checkpoints served by the web app.

Edit this file to swap which trained models the app (``app.py``) and the
preload script (``scripts/preload_db.py``) load at startup. Each constant is
paired with a short comment describing the architecture and training scope.

Layout under ``checkpoints/`` (relative to the repo root):

    classification/
        vit_dogs_cats/last.ckpt           ← active classifier (ViT, 132 classes)
        resnet50_dogs_cats/last.ckpt      ← previous classifier (ResNet50, 132 classes)
        vit_adam_cosine_annealing/...     ← Stanford-only ViT (120 classes)
        legacy/                           ← archived / experimental ckpts
    segmentation/
        deeplabv3_resnet50_last-v1.ckpt   ← active DeepLabV3 trimap segmenter
        sam2.1_hiera_large.pt             ← Meta SAM 2.1 (Hiera-Large), zero-shot
    pose/
        yolo_best.pt                      ← active Ultralytics YOLO pose model
        train/                            ← raw Ultralytics training output
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS_ROOT = REPO_ROOT / "checkpoints"


# ─── Classification ─────────────────────────────────────────────────────────
# Dogs+cats breed classifier.
# Architecture: Vision Transformer (ViT-B/16, google/vit-base-patch16-224-in21k).
# Training:     Stanford Dogs (120) + Oxford-IIIT cat breeds (12) → 132 classes.
# Best epoch:   67, val_acc=0.91, test top-1=89.6% (see notebooks/evaluate_combined_classifier.ipynb).
CLASSIFICATION_MODEL_NAME = "vit"  # must match the architecture saved in the .ckpt ("vit" or "resnet50")
CLASSIFICATION_CKPT_DIR = CHECKPOINTS_ROOT / "classification" / "vit_dogs_cats"
CLASSIFICATION_NUM_CLASSES = 132


# ─── Segmentation ───────────────────────────────────────────────────────────
# Fine-tuned DeepLabV3 (ResNet-50 backbone) trained on Oxford-IIIT pet trimap
# (3 classes: background / pet / boundary).
DEEPLAB_CKPT = CHECKPOINTS_ROOT / "segmentation" / "deeplabv3_resnet50_last-v1.ckpt"

# Meta SAM 2.1 zero-shot segmenter (Hiera-Large variant). Pretrained, not
# fine-tuned in this project. Downloaded by ``scripts/setup_and_run.sh``.
SAM2_CKPT = CHECKPOINTS_ROOT / "segmentation" / "sam2.1_hiera_large.pt"


# ─── Pose ───────────────────────────────────────────────────────────────────
# Ultralytics YOLOv26-pose, fine-tuned for dog/cat keypoint detection.
# Symlink/copy of the best epoch from ``pose/train/weights/best.pt``.
POSE_CKPT = CHECKPOINTS_ROOT / "pose" / "yolo_best.pt"


def resolve_classifier_ckpt() -> Optional[Path]:
    """Return the active classification checkpoint, or ``None`` if missing.

    Prefers ``CLASSIFICATION_CKPT_DIR/last.ckpt``; otherwise picks the most
    recently-modified ``*.ckpt`` in that directory. Lets users drop a freshly
    trained checkpoint into the directory without having to rename it.
    """
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


def describe_active_models() -> str:
    """Human-readable summary of the resolved checkpoints (for startup logs)."""
    cls_ckpt = resolve_classifier_ckpt()
    cls_status = str(cls_ckpt.relative_to(REPO_ROOT)) if cls_ckpt else "MISSING"
    return (
        f"Active models:\n"
        f"  classification : {CLASSIFICATION_MODEL_NAME} "
        f"({CLASSIFICATION_NUM_CLASSES} classes) ← {cls_status}\n"
        f"  segmentation   : DeepLabV3 ← {DEEPLAB_CKPT.relative_to(REPO_ROOT)}\n"
        f"  segmentation   : SAM 2.1   ← {SAM2_CKPT.relative_to(REPO_ROOT)}\n"
        f"  pose           : YOLO      ← {POSE_CKPT.relative_to(REPO_ROOT)}"
    )
