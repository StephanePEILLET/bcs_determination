import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification


class ViTTransfer(nn.Module):
    """
    Vision Transformer wrapper for PyTorch Lightning module.
    Automatically downloads the pretrained weights and injects a new classification head.
    Includes dropout before the classifier head.
    """
    def __init__(
        self,
        num_classes: int = 120,
        model_name: str = "google/vit-base-patch16-224-in21k",
        dropout: float = 0.0,
        pretrained: bool = True,
    ):
        super().__init__()
        if pretrained:
            self.vit = AutoModelForImageClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout,
            )
        else:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(
                model_name,
                num_labels=num_classes,
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout,
            )
            self.vit = AutoModelForImageClassification.from_config(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transformers models return an object, we only want the bare logits
        outputs = self.vit(x)
        return outputs.logits
