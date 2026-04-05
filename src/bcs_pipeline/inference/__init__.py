"""
Inference utilities for BCS Determination pipeline.

This module contains all the logic needed to load a trained model, pre-process
an image, and produce predictions.  Both the CLI ``inference.py`` script and
the FastAPI ``app.py`` should delegate to the helpers defined here.

Key functions
-------------
* :func:`load_model` – Load a ``LitBcsDetermination`` from a ``.ckpt`` file.
* :func:`load_class_names` – Extract human-readable breed names from the
  dataset directory structure.
* :func:`get_inference_transform` – Return the deterministic validation
  transform pipeline.
* :func:`predict_single` – Run inference on a **single PIL image**.
* :func:`predict_batch` – Run inference on a **batch of tensors**.

Usage
-----
>>> from bcs_pipeline.inference import load_model, predict_single
>>> model = load_model("checkpoints/best.ckpt", model_name="resnet50")
>>> result = predict_single(model, image, class_names=["Chihuahua", ...])
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

logger = logging.getLogger("bcs_pipeline")

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────
def load_model(
    checkpoint_path: str,
    model_name: str = "resnet50",
    num_classes: int = 120,
    device: torch.device | None = None,
):
    """Load a trained ``LitBcsDetermination`` from a checkpoint.

    Parameters
    ----------
    checkpoint_path:
        Path to the ``.ckpt`` file produced by PyTorch Lightning.
    model_name:
        Architecture used during training (``"resnet50"`` or ``"vit"``).
    num_classes:
        Number of output classes.
    device:
        Target device.  Defaults to CUDA if available, else CPU.

    Returns
    -------
    LitBcsDetermination
        Model in **eval** mode on the requested device.

    Raises
    ------
    FileNotFoundError
        If *checkpoint_path* does not exist.
    RuntimeError
        If the checkpoint cannot be loaded.
    """
    # Lazy import to avoid circular dependencies at module level
    from bcs_pipeline.lightning_module.bcs_determination_module import LitBcsDetermination

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading model from %s (device=%s)…", checkpoint_path, device)
    model = LitBcsDetermination.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        num_classes=num_classes,
        map_location=device,
    )
    model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────
# Class-name helpers
# ─────────────────────────────────────────────────────────────────────
def load_class_names(data_dir: str) -> Optional[List[str]]:
    """Extract sorted, human-readable class names from the dataset directory.

    The Stanford Dogs dataset folder names follow the pattern
    ``n02085620-Chihuahua``.  This function strips the synset prefix and
    returns only the breed name (e.g. ``"Chihuahua"``).

    Parameters
    ----------
    data_dir:
        Root directory that contains an ``Images/`` sub-folder.

    Returns
    -------
    list[str] | None
        Sorted breed names, or ``None`` if the directory does not exist.
    """
    images_dir = os.path.join(data_dir, "Images")
    if not os.path.isdir(images_dir):
        logger.warning("Images directory not found: %s", images_dir)
        return None

    classes = sorted(d.name for d in os.scandir(images_dir) if d.is_dir())
    clean = ["-".join(c.split("-")[1:]) for c in classes]
    logger.debug("Loaded %d class names from %s", len(clean), images_dir)
    return clean


# ─────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────
def get_inference_transform(image_size: int = 224) -> transforms.Compose:
    """Return the deterministic transform pipeline used for validation / inference.

    Parameters
    ----------
    image_size:
        Spatial size of the output tensor.

    Returns
    -------
    torchvision.transforms.Compose
    """
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─────────────────────────────────────────────────────────────────────
# Prediction helpers
# ─────────────────────────────────────────────────────────────────────
def predict_single(
    model,
    image: Image.Image,
    image_size: int = 224,
    class_names: Optional[List[str]] = None,
    top_k: int = 5,
    device: torch.device | None = None,
) -> Dict:
    """Run inference on a single PIL image.

    Parameters
    ----------
    model:
        A ``LitBcsDetermination`` (or any ``nn.Module`` that accepts
        a ``(1, 3, H, W)`` tensor and returns logits).
    image:
        Input image in RGB mode.
    image_size:
        Resolution expected by the model.
    class_names:
        Optional list mapping class indices to human-readable labels.
    top_k:
        Number of top predictions to return.
    device:
        Device on which the model resides.

    Returns
    -------
    dict
        ``{"class_id": int, "class_name": str | None, "confidence": float,
        "top_k": [{"class_id", "class_name", "confidence"}, ...]}``.
    """
    if device is None:
        device = next(model.parameters()).device

    transform = get_inference_transform(image_size)
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        top_probs, top_indices = probs.topk(top_k, dim=1)

    top_class = top_indices[0, 0].item()
    top_prob = top_probs[0, 0].item()

    top_k_results = []
    for i in range(top_k):
        idx = top_indices[0, i].item()
        top_k_results.append({
            "class_id": idx,
            "class_name": class_names[idx] if class_names and idx < len(class_names) else None,
            "confidence": top_probs[0, i].item(),
        })

    return {
        "class_id": top_class,
        "class_name": class_names[top_class] if class_names and top_class < len(class_names) else None,
        "confidence": top_prob,
        "top_k": top_k_results,
    }


def predict_batch(
    model,
    batch: torch.Tensor,
    class_names: Optional[List[str]] = None,
) -> List[Dict]:
    """Run inference on a pre-processed batch of tensors.

    Parameters
    ----------
    model:
        A model in eval mode.
    batch:
        Tensor of shape ``(B, 3, H, W)``, already normalised.
    class_names:
        Optional class-name mapping.

    Returns
    -------
    list[dict]
        One result dict per image in the batch (same schema as
        :func:`predict_single` without the ``top_k`` key).
    """
    device = next(model.parameters()).device
    batch = batch.to(device)

    with torch.no_grad():
        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        top_probs, top_classes = probs.max(dim=1)

    results = []
    for i in range(batch.size(0)):
        idx = top_classes[i].item()
        results.append({
            "class_id": idx,
            "class_name": class_names[idx] if class_names and idx < len(class_names) else None,
            "confidence": top_probs[i].item(),
        })
    return results
