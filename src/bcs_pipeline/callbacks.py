"""
Callback factory for BCS Determination pipeline.

This module centralises the creation of all PyTorch Lightning callbacks used
during training.  Every callback is configurable via Hydra's ``DictConfig`` so
that ``train.py`` stays lightweight and ``inference.py`` (or any other entry
point) can cherry-pick the helpers it needs.

Supported callbacks
-------------------
* **ModelCheckpoint** – saves the *k*-best checkpoints ranked by validation
  accuracy and always keeps the latest one.
* **EarlyStopping** – stops training when the monitored metric stalls for
  ``patience`` epochs.
* **LearningRateMonitor** – logs the current learning-rate at every optimiser
  step (only when a logger is active).

Usage
-----
>>> from bcs_pipeline.callbacks import build_callbacks
>>> callbacks = build_callbacks(cfg, checkpoint_dir=Path("experiments/run1/checkpoints"))
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pytorch_lightning as pl
from omegaconf import DictConfig

logger = logging.getLogger("bcs_pipeline")


def build_checkpoint_callback(
    checkpoint_dir: Path,
    monitor: str = "val/acc",
    mode: str = "max",
    save_top_k: int = 3,
    save_last: bool = True,
) -> pl.callbacks.ModelCheckpoint:
    # ... (existing build_checkpoint_callback logic untouched)
    callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch={epoch:02d}-val_acc={val/acc:.2f}-{step}",
        monitor=monitor,
        mode=mode,
        save_top_k=save_top_k,
        save_last=save_last,
        verbose=True,
    )
    logger.debug(
        "ModelCheckpoint: monitor=%s, mode=%s, top_k=%d, dir=%s",
        monitor, mode, save_top_k, checkpoint_dir,
    )
    return callback


def build_early_stopping_callback(
    monitor: str = "val/acc",
    patience: int = 5,
    mode: str = "max",
) -> pl.callbacks.EarlyStopping:
    callback = pl.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        mode=mode,
        verbose=True,
    )
    logger.debug("EarlyStopping: monitor=%s, patience=%d, mode=%s", monitor, patience, mode)
    return callback


def build_lr_monitor(logging_interval: str = "step") -> pl.callbacks.LearningRateMonitor:
    return pl.callbacks.LearningRateMonitor(logging_interval=logging_interval)


class TransferLearningFinetune(pl.callbacks.Callback):
    """
    Two-phase transfer learning callback to prevent catastrophic forgetting.
    Phase 1: Freezes the pretrained backbone and trains only the new classification head.
    Phase 2: Unfreezes the backbone at a specified epoch and reduces the learning rate for all parameters.
    """
    def __init__(self, unfreeze_at_epoch: int = 5, backbone_lr_multiplier: float = 0.1):
        super().__init__()
        self.unfreeze_epoch = unfreeze_at_epoch
        self.backbone_lr_multiplier = backbone_lr_multiplier
        self._backbone_frozen = False
        
    def _get_backbone_and_head(self, pl_module):
        if hasattr(pl_module.net, "backbone"): # ResNetTransfer
            return pl_module.net.backbone, getattr(pl_module.net.backbone, "fc", None)
        elif hasattr(pl_module.net, "vit"): # ViTTransfer
            return getattr(pl_module.net.vit, "vit", None), getattr(pl_module.net.vit, "classifier", None)
        return None, None

    def on_fit_start(self, trainer, pl_module):
        if self.unfreeze_epoch > 0:
            backbone, head = self._get_backbone_and_head(pl_module)
            if backbone is not None:
                # 1. Freeze the entire backbone
                for param in backbone.parameters():
                    param.requires_grad = False
                # 2. Keep the classification head unfrozen
                if head is not None:
                    for param in head.parameters():
                        param.requires_grad = True
                
                self._backbone_frozen = True
                logger.info("TransferLearning: Backbone frozen for Phase 1. Only training classifier head.")

    def on_train_epoch_start(self, trainer, pl_module):
        # Trigger Phase 2 at the specified epoch
        if self._backbone_frozen and trainer.current_epoch == self.unfreeze_epoch:
            backbone, _ = self._get_backbone_and_head(pl_module)
            if backbone is not None:
                # 1. Unfreeze the backbone
                for param in backbone.parameters():
                    param.requires_grad = True
                self._backbone_frozen = False
                
                # 2. Scale down the learning rate
                for opt in trainer.optimizers:
                    for param_group in opt.param_groups:
                        param_group["lr"] *= self.backbone_lr_multiplier
                        
                logger.info(
                    f"TransferLearning: Phase 2 triggered at epoch {trainer.current_epoch}. "
                    f"Backbone UNFREEZED. Learning rate multiplied by {self.backbone_lr_multiplier}."
                )


def build_finetuning_callback(cfg: DictConfig) -> pl.callbacks.Callback | None:
    finetuning_config = cfg.get("finetuning", None)
    if not finetuning_config:
        return None
    
    unfreeze_epoch = finetuning_config.get("unfreeze_at_epoch", 5)
    lr_multiplier = finetuning_config.get("backbone_lr_multiplier", 0.1)
    
    logger.debug(f"TransferLearningFinetune: unfreeze_at_epoch={unfreeze_epoch}, lr_multiplier={lr_multiplier}")
    return TransferLearningFinetune(unfreeze_at_epoch=unfreeze_epoch, backbone_lr_multiplier=lr_multiplier)


def build_callbacks(cfg: DictConfig, checkpoint_dir: Path) -> List[pl.callbacks.Callback]:
    """Build the full list of training callbacks from the Hydra config."""
    callbacks: List[pl.callbacks.Callback] = []
    
    # Identify the task to determine the monitored metric
    task = cfg.get("task", "classification")
    if task == "segmentation":
        monitor_metric = "val/iou"
        mode = "max"
        filename_pattern = "epoch={epoch:02d}-val_iou={val/iou:.2f}-{step}"
    else:
        monitor_metric = "val/acc"
        mode = "max"
        filename_pattern = "epoch={epoch:02d}-val_acc={val/acc:.2f}-{step}"

    # 1. Model checkpointing
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=filename_pattern,
        monitor=monitor_metric,
        mode=mode,
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_cb)

    # 2. Early stopping
    if cfg.get("patience", 0) > 0:
        callbacks.append(build_early_stopping_callback(
            monitor=monitor_metric,
            patience=cfg.patience,
            mode=mode
        ))

    # 3. Learning-rate monitor (only useful when a logger is active)
    if cfg.get("use_tensorboard", True) or cfg.get("use_wandb", False):
        callbacks.append(build_lr_monitor())

    # 4. Transfer Learning Fine-tuning
    finetuning_cb = build_finetuning_callback(cfg)
    if finetuning_cb is not None:
        callbacks.append(finetuning_cb)

    logger.info(
        "Callbacks ready: %s (Monitoring: %s)",
        ", ".join(type(c).__name__ for c in callbacks),
        monitor_metric
    )
    return callbacks
