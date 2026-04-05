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
    """Create a ``ModelCheckpoint`` callback.

    Parameters
    ----------
    checkpoint_dir:
        Directory where ``.ckpt`` files are written.
    monitor:
        Metric name to watch (must match a ``self.log(...)`` key in the
        LightningModule).
    mode:
        ``"max"`` for accuracy-like metrics, ``"min"`` for loss-like.
    save_top_k:
        How many best checkpoints to keep on disk.
    save_last:
        Always save the checkpoint of the most recent epoch.

    Returns
    -------
    pl.callbacks.ModelCheckpoint
    """
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
    """Create an ``EarlyStopping`` callback.

    Parameters
    ----------
    monitor:
        Metric name to watch.
    patience:
        Number of epochs with no improvement after which training is stopped.
    mode:
        ``"max"`` for accuracy-like metrics, ``"min"`` for loss-like.

    Returns
    -------
    pl.callbacks.EarlyStopping
    """
    callback = pl.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        mode=mode,
        verbose=True,
    )
    logger.debug("EarlyStopping: monitor=%s, patience=%d, mode=%s", monitor, patience, mode)
    return callback


def build_lr_monitor(logging_interval: str = "step") -> pl.callbacks.LearningRateMonitor:
    """Create a ``LearningRateMonitor`` callback.

    Parameters
    ----------
    logging_interval:
        ``"step"`` or ``"epoch"``.

    Returns
    -------
    pl.callbacks.LearningRateMonitor
    """
    return pl.callbacks.LearningRateMonitor(logging_interval=logging_interval)


def build_callbacks(cfg: DictConfig, checkpoint_dir: Path) -> List[pl.callbacks.Callback]:
    """Build the full list of training callbacks from the Hydra config.

    This is the **single entry-point** called by ``train.py``.  It reads
    relevant fields from *cfg* and delegates to the specialised builders above.

    Parameters
    ----------
    cfg:
        Hydra ``DictConfig`` (or any dict-like) containing at least
        ``patience``, ``use_tensorboard``, and optionally ``use_wandb``.
    checkpoint_dir:
        Where checkpoint files should be stored.

    Returns
    -------
    list[pl.callbacks.Callback]
        Ordered list of callbacks ready to be passed to ``pl.Trainer``.
    """
    callbacks: List[pl.callbacks.Callback] = []

    # 1. Model checkpointing
    callbacks.append(build_checkpoint_callback(checkpoint_dir=checkpoint_dir))

    # 2. Early stopping
    callbacks.append(build_early_stopping_callback(patience=cfg.patience))

    # 3. Learning-rate monitor (only useful when a logger is active)
    if cfg.use_tensorboard or getattr(cfg, "use_wandb", False):
        callbacks.append(build_lr_monitor())

    logger.info(
        "Callbacks ready: %s",
        ", ".join(type(c).__name__ for c in callbacks),
    )
    return callbacks
