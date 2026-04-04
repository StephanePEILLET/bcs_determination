import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNetTransfer(nn.Module):
    """
    ResNet50 model pre-trained on ImageNet, adapted for transfer learning.
    """
    def __init__(self, num_classes: int = 120, pretrained: bool = True):
        super().__init__()
        
        # Load optionally pre-trained model
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = resnet50(weights=weights)
        
        # Replace the final fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)
