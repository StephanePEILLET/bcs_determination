"""
Lightning module for image segmentation.

Uses a DeepLabV3-ResNet50 backbone from *torchvision* with pretrained COCO
weights.  Provides comprehensive metrics (IoU, Dice, Pixel Accuracy) and
rich TensorBoard visualisations (input images, ground-truth masks, predicted
masks, and colour-coded overlays).
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from torch import nn
from torchmetrics import JaccardIndex, MeanMetric
from PIL import Image

from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights


# ──────────────────────────────────────────────────────────────────────
# Dice Loss
# ──────────────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    """Generalised Dice loss for multi-class segmentation."""

    def __init__(self, num_classes: int, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets.long(), self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (probs * targets_one_hot).sum(dim=dims)
        cardinality = (probs + targets_one_hot).sum(dim=dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_per_class.mean()


# ──────────────────────────────────────────────────────────────────────
# Lightning Module
# ──────────────────────────────────────────────────────────────────────
class LitSegmentationModule(pl.LightningModule):
    """Segmentation LightningModule with DeepLabV3-ResNet50.

    Parameters
    ----------
    model_name:
        Currently only ``"deeplabv3_resnet50"`` is supported.
    lr:
        Learning rate.
    num_classes:
        Number of segmentation classes (including background).
    weight_ce:
        Weight for CrossEntropy term in the combined loss.
    weight_dice:
        Weight for Dice loss term.
    tensorboard:
        Dict of TensorBoard logging options (``log_images``, etc.).
    """

    def __init__(
        self,
        model_name: str = "deeplabv3_resnet50",
        lr: float = 1e-3,
        num_classes: int = 3,
        weight_ce: float = 1.0,
        weight_dice: float = 1.0,
        tensorboard: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        tb = tensorboard or {}
        self.log_images = tb.get("log_images", True)
        self.log_weight_histograms = tb.get("log_weight_histograms", False)
        self.histogram_every_n_epochs = tb.get("histogram_every_n_epochs", 5)

        # ── Model ──
        self.net = deeplabv3_resnet50(
            weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1,
        )
        self.net.classifier[4] = nn.Conv2d(
            256, num_classes, kernel_size=1,
        )

        # ── Loss ──
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(num_classes)
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

        # ── Metrics (train / val / test) ──
        task = "multiclass"
        self.train_iou = JaccardIndex(task=task, num_classes=num_classes)
        self.val_iou = JaccardIndex(task=task, num_classes=num_classes)
        self.test_iou = JaccardIndex(task=task, num_classes=num_classes)

        self.train_dice = MeanMetric()
        self.val_dice = MeanMetric()
        self.test_dice = MeanMetric()

        self.train_pixel_acc = MeanMetric()
        self.val_pixel_acc = MeanMetric()
        self.test_pixel_acc = MeanMetric()

        self.train_loss_agg = MeanMetric()
        self.val_loss_agg = MeanMetric()
        self.test_loss_agg = MeanMetric()

        self.val_iou_best = 0.0

        # Colour palette for mask visualisation (RGB per class)
        self._palette = self._build_palette(num_classes)

    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _build_palette(num_classes: int) -> list:
        base = [
            (0, 0, 0),        # class 0 – foreground (pet)
            (255, 255, 255),   # class 1 – background
            (128, 128, 128),   # class 2 – border
        ]
        for i in range(len(base), num_classes):
            np.random.seed(i + 42)
            base.append(tuple(np.random.randint(0, 256, 3).tolist()))
        return base[:num_classes]

    # ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if isinstance(out, dict):
            return out["out"]
        return out

    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_dice(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
        dice_sum = 0.0
        for c in range(num_classes):
            pred_c = (preds == c).float()
            tgt_c = (targets == c).float()
            inter = (pred_c * tgt_c).sum()
            dice_sum += (2.0 * inter + 1e-6) / (pred_c.sum() + tgt_c.sum() + 1e-6)
        return (dice_sum / num_classes).item()

    @staticmethod
    def _compute_pixel_acc(preds: torch.Tensor, targets: torch.Tensor) -> float:
        correct = (preds == targets).float().sum()
        total = targets.numel()
        return (correct / (total + 1e-8)).item()

    # ──────────────────────────────────────────────────────────────
    def _shared_step(self, batch, batch_idx: int, prefix: str):
        x, y = batch
        logits = self(x)
        y = y.long()

        loss_ce = self.ce_loss(logits, y)
        loss_dice = self.dice_loss(logits, y)
        loss = self.weight_ce * loss_ce + self.weight_dice * loss_dice

        preds = torch.argmax(logits, dim=1)

        nc = self.hparams.num_classes
        dice_val = self._compute_dice(preds, y, nc)
        pix_acc = self._compute_pixel_acc(preds, y)

        if prefix == "train":
            iou = self.train_iou(preds, y)
            self.train_dice(dice_val)
            self.train_pixel_acc(pix_acc)
            self.train_loss_agg(loss)
        elif prefix == "val":
            iou = self.val_iou(preds, y)
            self.val_dice(dice_val)
            self.val_pixel_acc(pix_acc)
            self.val_loss_agg(loss)
        else:
            iou = self.test_iou(preds, y)
            self.test_dice(dice_val)
            self.test_pixel_acc(pix_acc)
            self.test_loss_agg(loss)

        self.log(f"{prefix}/loss", loss, prog_bar=True, sync_dist=True, on_step=(prefix == "train"), on_epoch=True)
        self.log(f"{prefix}/iou", iou, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log(f"{prefix}/dice", dice_val, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log(f"{prefix}/pixel_acc", pix_acc, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)

        return loss

    # ──────────────────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, batch_idx, "val")
        if batch_idx == 0 and self.log_images and self.logger is not None:
            self._log_segmentation_images(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, batch_idx, "test")
        if batch_idx == 0 and self.log_images and self.logger is not None:
            self._log_segmentation_images(batch, "test")

    # ──────────────────────────────────────────────────────────────
    def on_validation_epoch_end(self):
        iou = self.val_iou.compute()
        if iou.item() > self.val_iou_best:
            self.val_iou_best = iou.item()
        self.log("val/iou_best", self.val_iou_best)

        train_loss = self.trainer.callback_metrics.get("train/loss_epoch")
        val_loss = self.trainer.callback_metrics.get("val/loss")
        if train_loss is not None and val_loss is not None:
            self.log("monitor/overfit_gap", val_loss - train_loss)

        if (
            self.log_weight_histograms
            and self.logger is not None
            and (self.current_epoch + 1) % self.histogram_every_n_epochs == 0
        ):
            self._log_weight_histograms()

    # ──────────────────────────────────────────────────────────────
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    # ──────────────────────────────────────────────────────────────
    # TensorBoard helpers
    # ──────────────────────────────────────────────────────────────
    def _get_tb_writer(self):
        from pytorch_lightning.loggers import TensorBoardLogger
        if isinstance(self.logger, TensorBoardLogger):
            return self.logger.experiment
        if hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "__iter__"):
            for exp in self.logger.experiment:
                if hasattr(exp, "add_scalar"):
                    return exp
        return None

    def _mask_to_rgb(self, mask: torch.Tensor) -> torch.Tensor:
        """Convert a (H, W) int mask to a (3, H, W) RGB float tensor."""
        h, w = mask.shape
        rgb = torch.zeros(3, h, w, dtype=torch.float32, device=mask.device)
        for cls_idx, colour in enumerate(self._palette):
            r, g, b = colour
            mask_c = (mask == cls_idx)
            rgb[0][mask_c] = r / 255.0
            rgb[1][mask_c] = g / 255.0
            rgb[2][mask_c] = b / 255.0
        return rgb

    def _log_segmentation_images(self, batch, prefix: str):
        """Log a grid of segmentation results to TensorBoard."""
        try:
            writer = self._get_tb_writer()
            if writer is None:
                return

            x, y = batch
            logits = self(x)
            preds = torch.argmax(logits, dim=1)

            n = min(6, x.size(0))
            inv_norm = transforms_inv_normalize()

            rows = []
            for i in range(n):
                img = inv_norm(x[i].cpu())
                gt_rgb = self._mask_to_rgb(y[i].cpu())
                pred_rgb = self._mask_to_rgb(preds[i].cpu())

                overlay_gt = overlay_mask_on_image(img, y[i].cpu(), alpha=0.4, palette=self._palette)
                overlay_pred = overlay_mask_on_image(img, preds[i].cpu(), alpha=0.4, palette=self._palette)

                row = torch.stack([img, gt_rgb, pred_rgb, overlay_gt, overlay_pred])
                rows.append(row)

            grid = torch.cat(rows, dim=0)
            writer.add_images(
                f"{prefix}/segmentation_grid",
                grid,
                self.global_step,
                dataformats="NCHW",
            )
        except Exception as exc:
            logger_dbg = logging.getLogger("bcs_pipeline")
            logger_dbg.warning("Failed to log segmentation images: %s", exc)

    def _log_weight_histograms(self):
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


# ──────────────────────────────────────────────────────────────────────
# Utility helpers (module-level)
# ──────────────────────────────────────────────────────────────────────
def transforms_inv_normalize():
    """Return a transform that undoes ImageNet normalisation (for display)."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    def _inv(t: torch.Tensor) -> torch.Tensor:
        t = t * std + mean
        return t.clamp(0, 1)
    return _inv


def overlay_mask_on_image(
    img: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.4,
    palette: list | None = None,
) -> torch.Tensor:
    """Blend a colour-coded segmentation mask over the image.

    Parameters
    ----------
    img : (3, H, W) float tensor in [0, 1]
    mask : (H, W) int tensor of class indices
    alpha : blending factor
    palette : list of (R, G, B) tuples per class

    Returns
    -------
    (3, H, W) float tensor in [0, 1]
    """
    if palette is None:
        palette = [(0, 0, 0), (255, 255, 255), (128, 128, 128)]

    h, w = mask.shape
    overlay = torch.zeros(3, h, w, dtype=torch.float32)
    for cls_idx, colour in enumerate(palette):
        if cls_idx >= len(palette):
            break
        r, g, b = colour
        m = (mask == cls_idx)
        overlay[0][m] = r / 255.0
        overlay[1][m] = g / 255.0
        overlay[2][m] = b / 255.0

    blended = (1 - alpha) * img + alpha * overlay
    return blended.clamp(0, 1)
