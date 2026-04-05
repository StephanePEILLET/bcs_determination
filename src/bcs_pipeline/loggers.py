"""
Logger factory for BCS Determination pipeline.

This module provides helpers to instantiate PyTorch Lightning loggers
(TensorBoard, Weights & Biases) from the Hydra configuration.

Centralising logger creation keeps ``train.py`` minimal and makes it
trivial to add new logger back-ends (e.g. MLflow, Neptune) in the future.

Usage
-----
>>> from bcs_pipeline.loggers import build_loggers
>>> loggers = build_loggers(cfg, experiment_dirs)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import pytorch_lightning as pl
from omegaconf import DictConfig

logger = logging.getLogger("bcs_pipeline")


def build_tensorboard_logger(
    save_dir: Path,
    name: str = "bcs_determination",
    version: str | None = None,
) -> pl.loggers.TensorBoardLogger:
    """Create a ``TensorBoardLogger``.

    Parameters
    ----------
    save_dir:
        Root directory for TensorBoard event files.
    name:
        Experiment name (sub-folder inside *save_dir*).
    version:
        Optional version string.  ``None`` lets Lightning auto-increment.

    Returns
    -------
    pl.loggers.TensorBoardLogger
    """
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=str(save_dir),
        name=name,
        version=version,
    )
    logger.debug("TensorBoardLogger: save_dir=%s, name=%s", save_dir, name)
    return tb_logger


def build_wandb_logger(
    project: str,
    name: str,
    save_dir: Path,
    log_model: bool = True,
) -> pl.loggers.WandbLogger | None:
    """Create a ``WandbLogger``, returning *None* if ``wandb`` is not installed.

    Parameters
    ----------
    project:
        W&B project name.
    name:
        Run name displayed in the W&B dashboard.
    save_dir:
        Local directory where W&B stores artefacts.
    log_model:
        Whether to log model checkpoints as W&B artefacts.

    Returns
    -------
    pl.loggers.WandbLogger | None
        The logger instance, or ``None`` if the ``wandb`` package is missing.
    """
    try:
        wandb_logger = pl.loggers.WandbLogger(
            project=project,
            name=name,
            save_dir=str(save_dir),
            log_model=log_model,
        )
        logger.debug("WandbLogger: project=%s, name=%s", project, name)
        return wandb_logger
    except ImportError:
        logger.warning("wandb is not installed — skipping W&B logging.")
        return None


def build_loggers(cfg: DictConfig, experiment_dirs: Dict[str, Path]) -> List:
    """Build the complete list of PL loggers from the Hydra configuration.

    This is the **single entry-point** called by ``train.py``.

    Parameters
    ----------
    cfg:
        Hydra configuration.  Expected keys: ``use_tensorboard``,
        ``use_wandb``, ``wandb_project``, ``model_name``.
    experiment_dirs:
        Mapping returned by
        :func:`bcs_pipeline.utils.config_utils.setup_experiment_dirs`.

    Returns
    -------
    list
        List of PL logger instances (may be empty if all logging is disabled).
    """
    loggers: List = []

    if cfg.use_tensorboard:
        loggers.append(
            build_tensorboard_logger(save_dir=experiment_dirs["tensorboard"])
        )

    if getattr(cfg, "use_wandb", False):
        wb = build_wandb_logger(
            project=cfg.wandb_project,
            name=f"{cfg.model_name}_bcs",
            save_dir=experiment_dirs["wandb"],
        )
        if wb is not None:
            loggers.append(wb)

    logger.info(
        "Loggers ready: %s",
        ", ".join(type(lg).__name__ for lg in loggers) or "(none)",
    )
    return loggers
