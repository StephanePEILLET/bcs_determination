import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import StochasticDepth


class ResNetTransfer(nn.Module):
    """
    ResNet50 model pre-trained on ImageNet, adapted for transfer learning.
    Includes dropout before the classifier and optional stochastic depth.
    """
    def __init__(
        self,
        num_classes: int = 120,
        pretrained: bool = True,
        dropout: float = 0.0,
        stochastic_depth: float = 0.0,
    ):
        super().__init__()

        # Load optionally pre-trained model
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = resnet50(weights=weights)

        # Inject stochastic depth into residual blocks
        if stochastic_depth > 0.0:
            self._inject_stochastic_depth(stochastic_depth)

        # Replace the final fully connected layer with dropout + linear
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, num_classes),
        )

    def _inject_stochastic_depth(self, drop_rate: float):
        """Apply linearly increasing stochastic depth across residual blocks."""
        blocks = []
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Sequential) and name.startswith("layer"):
                blocks.extend(list(module.children()))

        total_blocks = len(blocks)
        for i, block in enumerate(blocks):
            # Linear schedule: 0 at first block, drop_rate at last block
            block_drop_rate = drop_rate * (i / max(total_blocks - 1, 1))
            original_forward = block.forward

            sd = StochasticDepth(p=block_drop_rate, mode="row")

            def make_forward(orig_fwd, stoch_depth, blk):
                def new_forward(x):
                    identity = x
                    out = orig_fwd(x)
                    if hasattr(blk, 'downsample') and blk.downsample is not None:
                        identity = blk.downsample(x)
                    # Apply stochastic depth to the residual branch only
                    out = stoch_depth(out) + identity
                    return out
                return new_forward

            # Only apply to blocks that have a residual shortcut
            if hasattr(block, 'downsample'):
                block.forward = make_forward(original_forward, sd, block)

    def forward(self, x):
        return self.backbone(x)
