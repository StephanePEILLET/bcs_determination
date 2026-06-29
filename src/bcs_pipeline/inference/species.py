"""Species (dog vs cat) inference helpers — stage 1 of the cascade.

A tiny binary classifier whose prediction routes the rest of the pipeline:
the predicted species selects which breed classifier and which BCS regressor
run downstream. Reuses the generic classification loader / predictor since the
species model shares the same ``LitClassificationModule`` architecture, just
with ``num_classes=2``.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
from PIL import Image

from bcs_pipeline.inference.classification import (
    load_classification_model,
    predict_single,
)

logger = logging.getLogger("bcs_pipeline")

# Label order must match SpeciesClassificationDataModule (0=dog, 1=cat).
SPECIES_CLASS_NAMES = ["dog", "cat"]


def load_species_model(
    checkpoint_path: str,
    model_name: str = "vit",
    device: Optional[torch.device] = None,
):
    """Load the binary species classifier in eval mode."""
    return load_classification_model(
        checkpoint_path,
        model_name=model_name,
        num_classes=len(SPECIES_CLASS_NAMES),
        device=device,
    )


def predict_species(
    model,
    image: Image.Image,
    image_size: int = 224,
    device: Optional[torch.device] = None,
) -> Dict:
    """Predict the species of *image*.

    Returns ``{"species": "dog"|"cat", "confidence": float, "top_k": [...]}``.
    """
    result = predict_single(
        model,
        image,
        image_size=image_size,
        class_names=SPECIES_CLASS_NAMES,
        top_k=len(SPECIES_CLASS_NAMES),
        device=device,
    )
    return {
        "species": result["class_name"],
        "confidence": result["confidence"],
        "top_k": result["top_k"],
    }
