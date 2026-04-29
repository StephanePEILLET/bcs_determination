"""Classification inference helpers (ResNet50 / ViT on Stanford Dogs)."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from bcs_pipeline.lightning_module.classification_module import LitClassificationModule

logger = logging.getLogger("bcs_pipeline")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _remap_resnet_fc_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Older ResNet checkpoints stored ``net.backbone.fc.{weight,bias}`` (a plain
    Linear). The current ``ResNetTransfer`` wraps fc in
    ``nn.Sequential(Dropout, Linear)``, so the keys become
    ``net.backbone.fc.1.{weight,bias}``. This shim makes legacy checkpoints
    loadable.
    """
    rename = {
        "net.backbone.fc.weight": "net.backbone.fc.1.weight",
        "net.backbone.fc.bias": "net.backbone.fc.1.bias",
    }
    if not any(k in state_dict for k in rename):
        return state_dict
    out = {}
    for k, v in state_dict.items():
        out[rename.get(k, k)] = v
    return out


def load_classification_model(
    checkpoint_path: str,
    model_name: str = "resnet50",
    num_classes: int = 120,
    device: torch.device | None = None,
    pretrained: bool = False,
) -> LitClassificationModule:
    """Load a trained classification checkpoint in eval mode.

    Parameters
    ----------
    pretrained :
        Whether to initialize with pretrained ImageNet weights before loading
        the checkpoint.  Defaults to ``False`` since the checkpoint already
        contains the trained weights — setting it to ``True`` would trigger an
        unnecessary download.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading classification model from %s (device=%s)…", checkpoint_path, device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = _remap_resnet_fc_keys(ckpt["state_dict"])

    model = LitClassificationModule(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys when loading checkpoint: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys when loading checkpoint: %s", unexpected)

    model.to(device)
    model.eval()
    return model


def load_class_names(data_dir: str) -> Optional[List[str]]:
    """Extract sorted breed names from a Stanford Dogs ``Images/`` directory.

    Strips the ``n02085620-`` synset prefix from each folder name.
    """
    images_dir = os.path.join(data_dir, "Images")
    if not os.path.isdir(images_dir):
        logger.warning("Images directory not found: %s", images_dir)
        return None

    classes = sorted(d.name for d in os.scandir(images_dir) if d.is_dir())
    clean = ["-".join(c.split("-")[1:]) for c in classes]
    logger.debug("Loaded %d class names from %s", len(clean), images_dir)
    return clean


def get_inference_transform(image_size: int = 224) -> transforms.Compose:
    """Return the deterministic transform used at validation / inference."""
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def predict_single(
    model,
    image: Image.Image,
    image_size: int = 224,
    class_names: Optional[List[str]] = None,
    top_k: int = 5,
    device: torch.device | None = None,
) -> Dict:
    """Run inference on a single PIL image and return top-k predictions."""
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
    """Run inference on a pre-processed tensor batch (B, 3, H, W)."""
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
