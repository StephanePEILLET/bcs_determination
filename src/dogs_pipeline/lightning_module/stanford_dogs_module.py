from typing import Any, Dict

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from dogs_pipeline.models.resnet_transfer import ResNetTransfer
from dogs_pipeline.models.vit_transfer import ViTTransfer

class LitStanfordDogs(LightningModule):
    """
    LightningModule for Stanford Dogs Classification.
    """
    def __init__(
        self,
        model_name: str = "resnet50",
        num_classes: int = 120,
        lr: float = 0.001,
        optimizer_name: str = "adam",
        scheduler_config: dict = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        if model_name == "resnet50":
            self.net = ResNetTransfer(num_classes=num_classes, pretrained=True)
        elif model_name == "vit":
            self.net = ViTTransfer(num_classes=num_classes)
        else:
            raise ValueError(f"Model {model_name} not supported.")

        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train_loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        # Log sample images from the first batch to TensorBoard
        if batch_idx == 0 and self.logger is not None:
            try:
                from pytorch_lightning.loggers import TensorBoardLogger
                import torchvision
                if isinstance(self.logger, TensorBoardLogger):
                    x, _ = batch
                    # Render a grid of the first 8 images
                    grid = torchvision.utils.make_grid(x[:8], normalize=True)
                    self.logger.experiment.add_image("val/predictions_sample", grid, self.global_step)
            except Exception:
                pass

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val_acc_best", self.val_acc_best.compute())

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test_loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer_name = self.hparams.optimizer_name.lower()
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-5)
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=1e-5
            )
        else:
            raise ValueError(f"Optimizer {optimizer_name} not supported.")

        if not self.hparams.scheduler_config or self.hparams.scheduler_config.get("name") == "none":
            return {"optimizer": optimizer}

        scheduler_name = self.hparams.scheduler_config.get("name")
        if scheduler_name == "cosine_annealing":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.scheduler_config.get("T_max", 10)
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        elif scheduler_name == "reduce_lr_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=3
            )
            monitor = self.hparams.scheduler_config.get("monitor", "val_loss")
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": monitor}}
        else:
            raise ValueError(f"Scheduler {scheduler_name} not supported.")
