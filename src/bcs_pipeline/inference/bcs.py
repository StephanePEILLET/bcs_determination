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
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from bcs_pipeline.data.bcs_datamodule import IMAGENET_MEAN, IMAGENET_STD, BCSDataset
from bcs_pipeline.lightning_module.bcs_classification_module import (
    BCSClassificationHead,
    LitBCSClassification,
    covariates_to_vector,
    encode_bcs_covariates,
)
from bcs_pipeline.lightning_module.bcs_regression_module import (
    BCSRegressionHead,
    LitBCSRegression,
)
from bcs_pipeline.utils.device import get_best_device

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

    # Task detection: a classification checkpoint carries a ``bcs_classes`` hparam
    # (the ordered BCS scores its head predicts). Anything else is legacy
    # regression, kept working unchanged for backward compatibility.
    bcs_classes = hp.get("bcs_classes")
    task = "classification" if bcs_classes else "regression"

    covariate_names = list(hp.get("covariate_names") or [])

    if task == "classification":
        bcs_classes = [float(c) for c in bcs_classes]
        # Build the module with a randomly-initialised backbone, then overwrite
        # both backbone and head from the checkpoint state dict.
        base = LitBCSClassification(
            bcs_classes=bcs_classes,
            backbone_ckpt=None,
            model_name=model_name,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            covariate_names=covariate_names,
        )
    else:
        target_mean = float(hp.get("target_mean", 5.0))
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
    heads: List[nn.Module] = []
    for fck in fold_ckpts:
        ckpt = torch.load(str(fck), map_location=device, weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        head_sd = {k[len("head."):]: v for k, v in sd.items() if k.startswith("head.")}
        if not head_sd:
            continue
        if task == "classification":
            head = BCSClassificationHead(
                embedding_dim, hidden_dim, dropout,
                num_bcs_classes=len(bcs_classes),
                covariate_dim=len(covariate_names),
            )
        else:
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
        "Loaded BCS %s ensemble: %d fold head(s) from %s (device=%s)",
        task,
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
        "task": task,
        "bcs_classes": bcs_classes if task == "classification" else None,
        "covariate_names": covariate_names,
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
    sex: Optional[str] = None,
    long_coat: Optional[object] = None,
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
    sex : {"M", "F"}, optional
        Animal sex metadata. Used only by classification models trained with
        covariate conditioning; ``None`` = unknown (encoded as all-zeros, which
        the metadata-dropout training makes safe).
    long_coat : optional
        Long-coat metadata (truthy/falsy or ``"Y"``/``"N"``); ``None`` = unknown.

    Returns a dict always carrying ``{"bcs", "std", "num_folds", "scale",
    "category", "masked", "task"}``. ``bcs`` is clamped to [1, 9]; ``std`` is the
    inter-fold dispersion (a rough uncertainty proxy).

    For **classification** models it additionally returns ``bcs_class`` (the
    argmax BCS score), ``confidence`` (the argmax probability), ``probs`` (a
    ``{score: probability}`` map) and ``covariates_used`` (the metadata actually
    fed in). ``bcs`` is then the probability-weighted expected score. For
    **regression** models it also returns ``bcs_raw``.
    """
    module = handle["module"]
    device = device or handle["device"]
    transform = handle["transform"]
    task = handle.get("task", "regression")
    covariate_names = handle.get("covariate_names") or []

    if isinstance(image, Image.Image):
        pil = image.convert("RGB")
    else:
        pil = Image.open(str(image)).convert("RGB")

    if mask is not None:
        pil = BCSDataset._apply_mask(pil, np.asarray(mask))

    x = transform(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        feats = module.extract_features(x).float()  # (1, embedding_dim)

        if task == "classification":
            classes = torch.tensor(
                handle["bcs_classes"], dtype=torch.float32, device=feats.device
            )
            # Encode optional metadata into the covariate vector the head expects.
            cov = None
            covariates_used: Dict[str, float] = {}
            if covariate_names:
                covariates_used = encode_bcs_covariates(sex=sex, long_coat=long_coat)
                cov_vec = covariates_to_vector(covariate_names, covariates_used)
                cov = torch.tensor([cov_vec], dtype=torch.float32, device=feats.device)
            # Average per-fold logits, then softmax over the ensemble.
            logits = torch.stack(
                [head(feats, cov) for head in handle["heads"]], dim=0
            ).mean(dim=0)  # (1, C)
            probs = F.softmax(logits, dim=-1).squeeze(0)  # (C,)
            expected = float((probs * classes).sum().item())
            score = float(min(BCS_MAX, max(BCS_MIN, expected)))
            top_idx = int(probs.argmax().item())
            # Inter-fold dispersion of the expected score (uncertainty proxy).
            per_fold_exp = torch.stack(
                [
                    (F.softmax(head(feats, cov), dim=-1).squeeze(0) * classes).sum()
                    for head in handle["heads"]
                ]
            )
            std = float(per_fold_exp.std().item()) if per_fold_exp.numel() > 1 else 0.0
            return {
                "bcs": round(score, 2),
                "bcs_class": round(float(classes[top_idx].item()), 1),
                "confidence": round(float(probs[top_idx].item()), 4),
                "probs": {
                    str(round(float(c.item()), 1)): round(float(p.item()), 4)
                    for c, p in zip(classes, probs)
                },
                "std": round(std, 4),
                "num_folds": int(handle["num_folds"]),
                "scale": [int(BCS_MIN), int(BCS_MAX)],
                "category": bcs_category(score),
                "masked": mask is not None,
                "task": "classification",
                "covariates_used": {k: v for k, v in covariates_used.items() if v},
            }

        # ── Regression (legacy) path ─────────────────────────────────────
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
        "task": "regression",
    }
