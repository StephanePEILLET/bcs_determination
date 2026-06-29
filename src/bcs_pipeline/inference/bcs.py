"""BCS regression inference helpers.

Predicts a continuous Body Condition Score (1–9 scale) from an animal image.

The model is a frozen ViT breed-classification backbone followed by a small MLP
regression head, trained with Leave-One-Cat-Out cross-validation on the OGR
dataset (see ``scripts/train_bcs_regression.py``). Because LOCO-CV produces one
head per held-out cat rather than a single "final" model, inference **ensembles
all fold heads** (averaging their predictions) over a single shared frozen
backbone — cheap, since only the tiny heads differ between folds.

The backbone expects the animal **silhouette only**: background pixels are set
to neutral gray using the upstream segmentation mask before normalization, which
matches the training preprocessing in :class:`bcs_pipeline.data.bcs_datamodule.BCSDataset`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from bcs_pipeline.utils.device import get_best_device
from bcs_pipeline.data.bcs_datamodule import BCSDataset, IMAGENET_MEAN, IMAGENET_STD
from bcs_pipeline.lightning_module.bcs_regression_module import (
    BCSRegressionHead,
    LitBCSRegression,
)

logger = logging.getLogger("bcs_pipeline")

# BCS clinical scale bounds (1 = emaciated … 9 = obese).
BCS_MIN, BCS_MAX = 1.0, 9.0


def get_bcs_transform() -> transforms.Compose:
    """Resize(256) → CenterCrop(224) → ToTensor → ImageNet normalize.

    Must mirror the validation transform in ``BCSDataset`` so the backbone sees
    the same field of view at train and inference time.
    """
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def bcs_category(score: float) -> str:
    """Map a 1–9 BCS to the WSAVA condition band (French label)."""
    if score < 4.0:
        return "Maigreur / sous-poids"
    if score <= 5.0:
        return "Idéal"
    return "Surpoids / obésité"


def _find_fold_checkpoints(checkpoint_dir: Union[str, Path]) -> List[Path]:
    """Collect per-fold checkpoints under *checkpoint_dir*.

    Accepts either a directory containing ``fold_*/best.ckpt`` (the LOCO-CV
    layout) or a single ``.ckpt`` file. Prefers ``best.ckpt`` over ``last.ckpt``.
    """
    path = Path(checkpoint_dir)
    if path.is_file():
        return [path]
    folds = sorted(path.glob("fold_*/best.ckpt"))
    if not folds:
        folds = sorted(path.glob("fold_*/last.ckpt"))
    if not folds:
        folds = sorted(path.glob("**/*.ckpt"))
    return folds


def load_bcs_model(
    checkpoint_dir: Union[str, Path],
    *,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load the BCS regression ensemble (shared backbone + per-fold heads).

    Returns an opaque handle consumed by :func:`predict_bcs`. Hyperparameters
    (``model_name``, ``num_classes``, ``hidden_dim`` …) are read from the first
    checkpoint so the handle stays in sync with however the model was trained.
    """
    if device is None:
        device = get_best_device()

    fold_ckpts = _find_fold_checkpoints(checkpoint_dir)
    if not fold_ckpts:
        raise FileNotFoundError(
            f"No BCS regression checkpoints found under {checkpoint_dir}"
        )

    # The first checkpoint carries both the (frozen, shared) backbone weights
    # and that fold's head, plus the training hyper-parameters.
    first = torch.load(str(fold_ckpts[0]), map_location=device, weights_only=False)
    hp = first.get("hyper_parameters", {}) or {}
    model_name = hp.get("model_name", "vit")
    num_classes = int(hp.get("num_classes", 132))
    embedding_dim = int(hp.get("embedding_dim", 768))
    hidden_dim = int(hp.get("hidden_dim", 128))
    dropout = float(hp.get("dropout", 0.3))
    target_mean = float(hp.get("target_mean", 5.0))

    # Build the module with a randomly-initialised backbone, then overwrite both
    # backbone and head from the checkpoint state dict (``backbone_ckpt=None``
    # avoids re-reading the original classification checkpoint by absolute path).
    base = LitBCSRegression(
        backbone_ckpt=None,
        model_name=model_name,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        target_mean=target_mean,
    )
    sd0 = first.get("state_dict", first)
    missing, unexpected = base.load_state_dict(sd0, strict=False)
    backbone_missing = [k for k in missing if k.startswith("backbone.")]
    if backbone_missing:
        logger.warning(
            "BCS backbone load: %d missing key(s), e.g. %s",
            len(backbone_missing),
            backbone_missing[:3],
        )
    base.to(device).eval()

    # Collect every fold's head for ensembling over the shared backbone.
    heads: List[BCSRegressionHead] = []
    for fck in fold_ckpts:
        ckpt = torch.load(str(fck), map_location=device, weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        head_sd = {k[len("head."):]: v for k, v in sd.items() if k.startswith("head.")}
        if not head_sd:
            continue
        head = BCSRegressionHead(
            embedding_dim, hidden_dim, dropout, target_mean=target_mean
        )
        head.load_state_dict(head_sd)
        head.to(device).eval()
        heads.append(head)

    if not heads:
        raise RuntimeError(
            f"BCS checkpoints under {checkpoint_dir} contained no 'head.*' weights."
        )

    logger.info(
        "Loaded BCS regression ensemble: %d fold head(s) from %s (device=%s)",
        len(heads),
        checkpoint_dir,
        device,
    )
    return {
        "module": base,
        "heads": heads,
        "device": device,
        "transform": get_bcs_transform(),
        "num_folds": len(heads),
    }


# Supported BCS species branches. Each maps to a ``<base_dir>/<species>/`` folder
# containing ``fold_*/best.ckpt`` ensembles. Dog is a placeholder until dog BCS
# data is collected (see scripts/train_bcs_regression.py --species dog).
BCS_SPECIES = ("cat", "dog")


def _species_has_folds(species_dir: Path) -> bool:
    """True when *species_dir* holds at least one fold checkpoint."""
    if not species_dir.is_dir():
        return False
    return (
        any(species_dir.glob("fold_*/best.ckpt"))
        or any(species_dir.glob("fold_*/last.ckpt"))
        or any(species_dir.glob("**/*.ckpt"))
    )


def load_bcs_models(
    base_dir: Union[str, Path],
    *,
    device: Optional[torch.device] = None,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Load the per-species BCS ensembles under *base_dir*.

    Expected layout ``<base_dir>/{cat,dog}/fold_*/best.ckpt``. For backward
    compatibility, if no ``cat/`` sub-folder exists but flat ``fold_*`` folders
    are present directly under *base_dir*, those legacy checkpoints are treated
    as the **cat** model.

    Returns ``{"cat": handle_or_None, "dog": handle_or_None}``. A ``None`` entry
    means that species has no trained model yet (e.g. the dog placeholder), and
    callers should degrade gracefully ("BCS unavailable for <species>").
    """
    base = Path(base_dir)
    handles: Dict[str, Optional[Dict[str, Any]]] = {sp: None for sp in BCS_SPECIES}

    for species in BCS_SPECIES:
        species_dir = base / species
        if _species_has_folds(species_dir):
            handles[species] = load_bcs_model(species_dir, device=device)
        elif species == "cat" and _species_has_folds(base):
            # Legacy flat layout (checkpoints/bcs_regression/fold_*/) = cat model.
            logger.info("Using legacy flat BCS checkpoints under %s as the cat model.", base)
            handles[species] = load_bcs_model(base, device=device)
        else:
            logger.info("No BCS model found for species '%s' under %s.", species, species_dir)

    return handles



def predict_bcs(
    handle: Dict[str, Any],
    image: Union[str, Path, Image.Image],
    mask: Optional[np.ndarray] = None,
    *,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Predict the Body Condition Score for *image*.

    Parameters
    ----------
    handle : dict
        Returned by :func:`load_bcs_model`.
    image : path or PIL.Image
        The animal photo (raw, un-masked).
    mask : np.ndarray, optional
        Segmentation map (``0 = foreground/animal``, non-zero = background), as
        produced by any segmentation backend. When provided, the background is
        set to neutral gray before inference — matching training. When ``None``
        the full image is used (degraded accuracy).

    Returns a dict ``{"bcs", "bcs_raw", "std", "num_folds", "scale", "category",
    "masked"}``. ``bcs`` is clamped to [1, 9]; ``std`` is the inter-fold
    dispersion (a rough uncertainty proxy).
    """
    module = handle["module"]
    device = device or handle["device"]
    transform = handle["transform"]

    if isinstance(image, Image.Image):
        pil = image.convert("RGB")
    else:
        pil = Image.open(str(image)).convert("RGB")

    if mask is not None:
        pil = BCSDataset._apply_mask(pil, np.asarray(mask))

    x = transform(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        feats = module.extract_features(x).float()  # (1, embedding_dim)
        preds = torch.stack([head(feats) for head in handle["heads"]], dim=0)  # (F, 1)
        mean = preds.mean(dim=0)
        std = preds.std(dim=0) if preds.shape[0] > 1 else torch.zeros_like(mean)

    raw = float(mean.item())
    score = float(min(BCS_MAX, max(BCS_MIN, raw)))
    return {
        "bcs": round(score, 2),
        "bcs_raw": round(raw, 4),
        "std": round(float(std.item()), 4),
        "num_folds": int(handle["num_folds"]),
        "scale": [int(BCS_MIN), int(BCS_MAX)],
        "category": bcs_category(score),
        "masked": mask is not None,
    }
