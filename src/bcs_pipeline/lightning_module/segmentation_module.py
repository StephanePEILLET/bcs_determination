import torch
import pytorch_lightning as pl
from torch import nn
from torchmetrics import JaccardIndex

class LitSegmentationModule(pl.LightningModule):
    """
    Lightning module skeleton for Image Segmentation.
    Features tracking of IoU (Jaccard Index) instead of Accuracy.
    """

    def __init__(
        self,
        model_name: str = "unet",
        lr: float = 1e-3,
        num_classes: int = 2,  # e.g. background vs dog
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # TODO: Implement dynamic model selection for segmentation (UNet, DeepLabV3, etc)
        # Using a dummy Conv2d that outputs (N, num_classes, H, W) for now.
        # This will be replaced by a real pretrained segmentation model structure.
        self.net = nn.Conv2d(3, num_classes, kernel_size=3, padding=1)
        
        # 2D cross entropy in PyTorch lacks deterministic implementations on GPU sometimes
        torch.use_deterministic_algorithms(False)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        # JaccardIndex is often used for IoU in segmentation
        self.train_iou = JaccardIndex(task="multiclass", num_classes=num_classes)
        self.val_iou = JaccardIndex(task="multiclass", num_classes=num_classes)
        self.test_iou = JaccardIndex(task="multiclass", num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _shared_step(self, batch, batch_idx, prefix: str):
        x, y = batch
        logits = self(x)
        
        loss = self.criterion(logits, y)
        
        # Calculate metric
        preds = torch.argmax(logits, dim=1)
        if prefix == "train":
            iou = self.train_iou(preds, y)
        elif prefix == "val":
            iou = self.val_iou(preds, y)
        else:
            iou = self.test_iou(preds, y)

        self.log(f"{prefix}/loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{prefix}/iou", iou, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
