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


def _resolve_file(root: Path, filename: str) -> Path:
    """Resolve a checkpoint path by searching recursively under *root*.

    If the exact ``root / filename`` exists, return it directly.  Otherwise,
    perform a recursive glob for ``**/filename`` under *root* and return the
    most recently modified match.  Returns the original ``root / filename``
    (which may not exist) as a fallback so callers can still do
    ``.is_file()`` checks.
    """
    direct = root / filename
    if direct.is_file():
        return direct
    candidates = sorted(
        root.glob(f"**/{filename}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else direct


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
DEEPLAB_CKPT = _resolve_file(CHECKPOINTS_ROOT, "deeplabv3_resnet50_last-v1.ckpt")

# Meta SAM 2.1 zero-shot segmenter (Hiera-Large variant). Pretrained, not
# fine-tuned in this project. Downloaded by ``scripts/setup_and_run.sh``.
SAM2_CKPT = _resolve_file(CHECKPOINTS_ROOT, "sam2.1_hiera_large.pt")

# Meta SAM 3 zero-shot, concept-grounded segmenter. Hosted on HuggingFace
# (``facebook/sam3``); the checkpoint and BPE vocabulary are fetched by
# ``scripts/setup_and_run.sh`` after ``hf auth login``. The exact filename
# is gated behind HF auth — adjust if HuggingFace renames the artifact.
SAM3_CKPT = _resolve_file(CHECKPOINTS_ROOT, "sam3_image_model.pt")
SAM3_BPE = _resolve_file(CHECKPOINTS_ROOT, "bpe_simple_vocab_16e6.txt.gz")


# ─── Pose ───────────────────────────────────────────────────────────────────
# Ultralytics YOLOv26-pose, fine-tuned for dog/cat keypoint detection.
# Symlink/copy of the best epoch from ``pose/train/weights/best.pt``.
POSE_CKPT = _resolve_file(CHECKPOINTS_ROOT, "yolo_best.pt")


def resolve_classifier_ckpt() -> Optional[Path]:
    """Return the active classification checkpoint, or ``None`` if missing.

    Prefers ``CLASSIFICATION_CKPT_DIR/last.ckpt``; otherwise picks the most
    recently-modified ``*.ckpt`` in that directory.  Also falls back to a
    recursive search under ``CHECKPOINTS_ROOT/classification`` in case the
    checkpoint directory is nested deeper.
    """
    if CLASSIFICATION_CKPT_DIR.is_dir():
        last = CLASSIFICATION_CKPT_DIR / "last.ckpt"
        if last.is_file():
            return last
        candidates = sorted(
            CLASSIFICATION_CKPT_DIR.glob("*.ckpt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]

    cls_root = CHECKPOINTS_ROOT / "classification"
    if cls_root.is_dir():
        candidates = sorted(
            cls_root.glob("**/last.ckpt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]

    return None


def describe_active_models() -> str:
    """Human-readable summary of the resolved checkpoints (for startup logs)."""
    cls_ckpt = resolve_classifier_ckpt()
    cls_status = str(cls_ckpt.relative_to(REPO_ROOT)) if cls_ckpt else "MISSING"
    sam3_status = (
        str(SAM3_CKPT.relative_to(REPO_ROOT)) if SAM3_CKPT.is_file() else "MISSING (optional)"
    )
    return (
        f"Active models:\n"
        f"  classification : {CLASSIFICATION_MODEL_NAME} "
        f"({CLASSIFICATION_NUM_CLASSES} classes) ← {cls_status}\n"
        f"  segmentation   : DeepLabV3 ← {DEEPLAB_CKPT.relative_to(REPO_ROOT)}\n"
        f"  segmentation   : SAM 2.1   ← {SAM2_CKPT.relative_to(REPO_ROOT)}\n"
        f"  segmentation   : SAM 3     ← {sam3_status}\n"
        f"  pose           : YOLO      ← {POSE_CKPT.relative_to(REPO_ROOT)}"
    )
