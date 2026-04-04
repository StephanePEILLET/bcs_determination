from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassPrecisionRecallCurve,
)

from bcs_pipeline.models.resnet_transfer import ResNetTransfer
from bcs_pipeline.models.vit_transfer import ViTTransfer


# ──────────────────────────────────────────────────────────────────
# Mixup / CutMix utilities
# ──────────────────────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.2):
    """Apply Mixup augmentation: convex combination of two random samples."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix augmentation: cut and paste patches between samples."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    _, _, H, W = x.shape
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    cy = np.random.randint(H)
    cx = np.random.randint(W)

    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # Adjust lambda to the actual area ratio
    lam = 1 - ((y2 - y1) * (x2 - x1)) / (H * W)
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for mixed labels."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ──────────────────────────────────────────────────────────────────
# Lightning Module
# ──────────────────────────────────────────────────────────────────
class LitBcsDetermination(LightningModule):
    """
    LightningModule for BCS Determination with:
    - Mixup / CutMix augmentation
    - Label smoothing
    - Dropout & stochastic depth
    - Comprehensive TensorBoard logging
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        num_classes: int = 120,
        lr: float = 0.001,
        optimizer_name: str = "adam",
        weight_decay: float = 1e-4,
        scheduler_config: dict = None,
        regularization: dict = None,
        tensorboard: dict = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ── Regularization config ──
        reg = regularization or {}
        self.dropout = reg.get("dropout", 0.0)
        self.label_smoothing = reg.get("label_smoothing", 0.0)
        self.mixup_alpha = reg.get("mixup_alpha", 0.0)
        self.cutmix_alpha = reg.get("cutmix_alpha", 0.0)
        self.mixup_prob = reg.get("mixup_prob", 0.5)
        self.stochastic_depth = reg.get("stochastic_depth", 0.0)

        # ── TensorBoard config ──
        tb = tensorboard or {}
        self.log_graph = tb.get("log_graph", False)
        self.log_weight_histograms = tb.get("log_weight_histograms", False)
        self.histogram_every_n_epochs = tb.get("histogram_every_n_epochs", 5)
        self.log_images = tb.get("log_images", True)
        self.log_confusion_matrix = tb.get("log_confusion_matrix", False)
        self.log_pr_curve = tb.get("log_pr_curve", False)

        # ── Build network ──
        if model_name == "resnet50":
            self.net = ResNetTransfer(
                num_classes=num_classes,
                pretrained=True,
                dropout=self.dropout,
                stochastic_depth=self.stochastic_depth,
            )
        elif model_name == "vit":
            self.net = ViTTransfer(
                num_classes=num_classes,
                dropout=self.dropout,
            )
        else:
            raise ValueError(f"Model {model_name} not supported.")

        # ── Loss with label smoothing ──
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        # ── Metrics ──
        self.train_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.val_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.test_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")

        self.train_acc_top5 = MulticlassAccuracy(num_classes=num_classes, top_k=5, average="micro")
        self.val_acc_top5 = MulticlassAccuracy(num_classes=num_classes, top_k=5, average="micro")
        self.test_acc_top5 = MulticlassAccuracy(num_classes=num_classes, top_k=5, average="micro")

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_acc_best = MaxMetric()

        # Confusion matrix (computed on validation)
        if self.log_confusion_matrix:
            self.val_confmat = MulticlassConfusionMatrix(num_classes=num_classes)

        # Buffers for PR curves
        if self.log_pr_curve:
            self._val_probs = []
            self._val_targets = []

    # ──────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        return self.net(x)

    # ──────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────
    def on_train_start(self):
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

        # Log model graph
        if self.log_graph and self.logger is not None:
            try:
                sample = torch.randn(1, 3, 224, 224, device=self.device)
                self.logger.experiment.add_graph(self, sample)
            except Exception:
                pass

    def training_step(self, batch: Any, batch_idx: int):
        x, y = batch

        # ── Apply Mixup or CutMix ──
        use_mix = self.training and (self.mixup_alpha > 0 or self.cutmix_alpha > 0)
        if use_mix:
            if self.cutmix_alpha > 0 and (self.mixup_alpha <= 0 or np.random.rand() > self.mixup_prob):
                x, y_a, y_b, lam = cutmix_data(x, y, self.cutmix_alpha)
            else:
                x, y_a, y_b, lam = mixup_data(x, y, self.mixup_alpha)
            logits = self.forward(x)
            loss = mixup_criterion(self.criterion, logits, y_a, y_b, lam)
        else:
            logits = self.forward(x)
            loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)

        # Metrics (use original y for accuracy, not mixed labels)
        self.train_loss(loss)
        self.train_acc(preds, y)
        self.train_acc_top5(logits, y)

        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc_top5", self.train_acc_top5, on_step=False, on_epoch=True)

        return loss

    # ──────────────────────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────────────────────
    def validation_step(self, batch: Any, batch_idx: int):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)

        self.val_loss(loss)
        self.val_acc(preds, y)
        self.val_acc_top5(logits, y)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc_top5", self.val_acc_top5, on_step=False, on_epoch=True)

        # Confusion matrix update
        if self.log_confusion_matrix:
            self.val_confmat.update(preds, y)

        # PR curve buffer
        if self.log_pr_curve:
            self._val_probs.append(probs.detach().cpu())
            self._val_targets.append(y.detach().cpu())

        # Log sample images from the first batch
        if batch_idx == 0 and self.log_images and self.logger is not None:
            self._log_prediction_images(x, y, preds, probs)

    def on_validation_epoch_end(self):
        # Best accuracy tracking
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute())

        # ── Overfitting gap ──
        train_loss = self.trainer.callback_metrics.get("train/loss_epoch")
        val_loss = self.trainer.callback_metrics.get("val/loss")
        if train_loss is not None and val_loss is not None:
            self.log("monitor/overfit_gap", val_loss - train_loss)

        train_acc = self.trainer.callback_metrics.get("train/acc_epoch")
        val_acc = self.trainer.callback_metrics.get("val/acc")
        if train_acc is not None and val_acc is not None:
            self.log("monitor/generalization_gap", train_acc - val_acc)

        # ── Confusion matrix to TensorBoard ──
        if self.log_confusion_matrix and self.logger is not None:
            self._log_confusion_matrix()

        # ── PR curves to TensorBoard ──
        if self.log_pr_curve and self.logger is not None and self._val_probs:
            self._log_pr_curves()

        # ── Weight histograms ──
        if (
            self.log_weight_histograms
            and self.logger is not None
            and (self.current_epoch + 1) % self.histogram_every_n_epochs == 0
        ):
            self._log_weight_histograms()

    # ──────────────────────────────────────────────────────────────
    # Test
    # ──────────────────────────────────────────────────────────────
    def test_step(self, batch: Any, batch_idx: int):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.test_loss(loss)
        self.test_acc(preds, y)
        self.test_acc_top5(logits, y)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc_top5", self.test_acc_top5, on_step=False, on_epoch=True)

    # ──────────────────────────────────────────────────────────────
    # Optimizer & scheduler
    # ──────────────────────────────────────────────────────────────
    def configure_optimizers(self):
        wd = self.hparams.weight_decay
        optimizer_name = self.hparams.optimizer_name.lower()

        if optimizer_name == "adam":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=wd)
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=wd
            )
        else:
            raise ValueError(f"Optimizer {optimizer_name} not supported.")

        if not self.hparams.scheduler_config or self.hparams.scheduler_config.get("name") == "none":
            return {"optimizer": optimizer}

        scheduler_name = self.hparams.scheduler_config.get("name")
        if scheduler_name == "cosine_annealing":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.scheduler_config.get("T_max", 10),
                eta_min=self.hparams.scheduler_config.get("eta_min", 0),
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

    # ──────────────────────────────────────────────────────────────
    # TensorBoard logging helpers
    # ──────────────────────────────────────────────────────────────
    def _get_tb_writer(self):
        """Return the SummaryWriter if a TensorBoardLogger is available."""
        from pytorch_lightning.loggers import TensorBoardLogger
        if isinstance(self.logger, TensorBoardLogger):
            return self.logger.experiment
        # Multi-logger scenario
        if hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "__iter__"):
            for exp in self.logger.experiment:
                if hasattr(exp, "add_scalar"):
                    return exp
        return None

    def _log_prediction_images(self, x, y, preds, probs):
        """Log a grid of sample images with predicted vs true labels."""
        try:
            writer = self._get_tb_writer()
            if writer is None:
                return

            n = min(8, x.size(0))
            grid = torchvision.utils.make_grid(x[:n], normalize=True, nrow=4)
            writer.add_image("val/prediction_samples", grid, self.global_step)

            # Log per-image text
            for i in range(n):
                conf = probs[i, preds[i]].item() * 100
                tag = f"val/sample_{i}_pred{preds[i].item()}_true{y[i].item()}_{conf:.0f}pct"
                writer.add_scalar(tag, conf, self.global_step)
        except Exception:
            pass

    def _log_confusion_matrix(self):
        """Render the confusion matrix as an image and send to TensorBoard."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            writer = self._get_tb_writer()
            if writer is None:
                return

            cm = self.val_confmat.compute().detach().cpu().numpy()
            self.val_confmat.reset()

            fig, ax = plt.subplots(figsize=(12, 12))
            ax.imshow(cm, interpolation="nearest", cmap="Blues")
            ax.set_title(f"Confusion Matrix — Epoch {self.current_epoch}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            fig.tight_layout()

            writer.add_figure("val/confusion_matrix", fig, self.current_epoch)
            plt.close(fig)
        except Exception:
            pass

    def _log_pr_curves(self):
        """Log precision-recall curves to TensorBoard."""
        try:
            writer = self._get_tb_writer()
            if writer is None:
                return

            all_probs = torch.cat(self._val_probs, dim=0)
            all_targets = torch.cat(self._val_targets, dim=0)

            num_classes = all_probs.size(1)
            # Log aggregate PR curve (one-vs-rest for top 10 most frequent classes)
            target_counts = torch.bincount(all_targets, minlength=num_classes)
            top_classes = torch.argsort(target_counts, descending=True)[:10]

            for cls_idx in top_classes:
                cls = cls_idx.item()
                binary_targets = (all_targets == cls).int()
                binary_probs = all_probs[:, cls]
                writer.add_pr_curve(
                    f"pr_curve/class_{cls}",
                    labels=binary_targets,
                    predictions=binary_probs,
                    global_step=self.current_epoch,
                )

            self._val_probs.clear()
            self._val_targets.clear()
        except Exception:
            pass

    def _log_weight_histograms(self):
        """Log weight and gradient histograms for all layers."""
        try:
            writer = self._get_tb_writer()
            if writer is None:
                return

            for name, param in self.named_parameters():
                if param.requires_grad:
                    writer.add_histogram(f"weights/{name}", param.data, self.current_epoch)
                    if param.grad is not None:
                        writer.add_histogram(f"gradients/{name}", param.grad, self.current_epoch)
        except Exception:
            pass
