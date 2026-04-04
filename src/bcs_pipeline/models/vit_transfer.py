import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification

class ViTTransfer(nn.Module):
    """
    Vision Transformer wrapper for PyTorch Lightning module.
    Automatically downloads the pretrained weights and injects a new classification head.
    """
    def __init__(self, num_classes: int = 120, model_name: str = "google/vit-base-patch16-224-in21k"):
        super().__init__()
        # ignore_mismatched_sizes=True drops the classfier head trained on 21k classes
        # and re-initializes one for `num_classes` automatically.
        self.vit = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transformers models return an object, we only want the bare logits
        outputs = self.vit(x)
        return outputs.logits
