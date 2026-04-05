"""
Trainer factory for BCS Determination pipeline.

Constructs a fully-configured ``pytorch_lightning.Trainer`` from the Hydra
configuration, reusing the callback and logger factories defined in
:mod:`bcs_pipeline.callbacks` and :mod:`bcs_pipeline.loggers`.

Usage
-----
>>> from bcs_pipeline.trainer_factory import build_trainer
>>> trainer = build_trainer(cfg, experiment_dirs)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import pytorch_lightning as pl
from omegaconf import DictConfig

from bcs_pipeline.callbacks import build_callbacks
from bcs_pipeline.loggers import build_loggers

logger = logging.getLogger("bcs_pipeline")


def build_trainer(
    cfg: DictConfig,
    experiment_dirs: Dict[str, Path],
) -> pl.Trainer:
    """Construct a ``pl.Trainer`` from the Hydra configuration.

    This is the **single entry-point** called by ``train.py``.  It wires
    together callbacks, loggers, and all other trainer options so that the
    training script itself contains virtually no boiler-plate.

    Parameters
    ----------
    cfg:
        Full Hydra configuration.
    experiment_dirs:
        Mapping returned by
        :func:`bcs_pipeline.utils.config_utils.setup_experiment_dirs`.

    Returns
    -------
    pl.Trainer
    """
    # Build components
    callbacks = build_callbacks(cfg, checkpoint_dir=experiment_dirs["checkpoints"])
    loggers = build_loggers(cfg, experiment_dirs)

    trainer_kwargs = dict(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.get("strategy", "auto"),
        sync_batchnorm=cfg.trainer.get("sync_batchnorm", False),
        fast_dev_run=cfg.trainer.get("fast_dev_run", False),
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=cfg.log_every_n_steps,
        val_check_interval=cfg.val_check_interval,
        gradient_clip_val=cfg.get("gradient_clip_val", None),
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True,
        precision=cfg.precision,
    )

    logger.info("Building Trainer with %d callbacks and %d loggers.", len(callbacks), len(loggers))
    return pl.Trainer(**trainer_kwargs)


def get_checkpoint_callback(trainer: pl.Trainer) -> pl.callbacks.ModelCheckpoint | None:
    """Retrieve the ``ModelCheckpoint`` callback from a trainer instance.

    Useful after training to access ``best_model_path`` and
    ``best_model_score``.

    Parameters
    ----------
    trainer:
        A trainer that was built with :func:`build_trainer`.

    Returns
    -------
    pl.callbacks.ModelCheckpoint | None
    """
    for cb in trainer.callbacks:
        if isinstance(cb, pl.callbacks.ModelCheckpoint):
            return cb
    return None
