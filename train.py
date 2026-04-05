"""
BCS Determination – Training entry-point.

This script is intentionally kept **lightweight**: all heavy-lifting
(callbacks, loggers, trainer construction, stats) is delegated to reusable
modules inside ``src/bcs_pipeline/``.

Usage
-----
.. code-block:: bash

    # Default config (configs/config.yaml)
    python train.py

    # Override hyper-parameters on the fly
    python train.py data_dir=/data/stanford_dogs batch_size=64 max_epochs=50

    # Full reproducibility: set seed, accelerator, precision
    python train.py seed=123 trainer.accelerator=gpu precision=16-mixed
"""

import os
import sys
import logging
from pathlib import Path

# ── Make the ``src/`` package importable ─────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig

from bcs_pipeline.data.classification_datamodule import StanfordClassificationDataModule
from bcs_pipeline.data.segmentation_datamodule import StanfordSegmentationDataModule
from bcs_pipeline.lightning_module.classification_module import LitClassificationModule
from bcs_pipeline.lightning_module.segmentation_module import LitSegmentationModule
from bcs_pipeline.trainer_factory import build_trainer, get_checkpoint_callback
from bcs_pipeline.utils.config_utils import (
    setup_experiment_dirs,
    validate_config,
    save_config_snapshot,
)
from bcs_pipeline.utils.logging_utils import (
    setup_logging,
    log_experiment_info,
    print_config,
    print_config_rich,
)
from bcs_pipeline.utils.dataset_stats import (
    compute_all_stats,
    display_stats_rich,
    log_stats,
    save_stats_json,
)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> float:
    """Run one training experiment and return the best validation accuracy.

    The function orchestrates the following steps:

    1. Validate the Hydra configuration.
    2. Create experiment directories (checkpoints, logs, splits, stats, …).
    3. **Save a config snapshot** (YAML + JSON) for full reproducibility.
    4. Seed all RNGs globally.
    5. Set up the DataModule with **stratified splits** (persisted to JSON).
    6. **Compute and display dataset statistics** per class (Rich tables).
    7. Build the model.
    8. Build the Trainer (callbacks + loggers wired automatically).
    9. Train and evaluate.

    Parameters
    ----------
    cfg:
        Hydra-managed configuration (merged from YAML + CLI overrides).

    Returns
    -------
    float
        Best ``val/acc`` observed during training (used as the Optuna
        objective when running hyper-parameter sweeps).
    """
    # ── 1. Validate & set up experiment artefacts ────────────────────
    if not validate_config(cfg):
        raise ValueError("Configuration validation failed. Check your config.yaml.")

    experiment_dirs = setup_experiment_dirs(cfg)

    logger = setup_logging(
        log_file=experiment_dirs["logs"] / "train.log",
        level=logging.INFO,
    )
    log_experiment_info(logger, cfg, experiment_dirs)
    print_config_rich(cfg)
    print_config(cfg, logger=logger)

    # ── 2. Save config snapshot for reproducibility ──────────────────
    snapshot_paths = save_config_snapshot(cfg, experiment_dirs)
    logger.info("Config snapshot saved → %s", snapshot_paths["yaml"])

    # ── 3. Reproducibility ───────────────────────────────────────────
    pl.seed_everything(cfg.seed, workers=True)

    # ── 4. Data (stratified split + manifest) ────────────────────────
    logger.info("Setting up data module (stratified splits)…")
    task = cfg.get("task", "classification")

    if task == "segmentation":
        data_module = StanfordSegmentationDataModule(
            data_dir=cfg.data_dir,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            image_size=cfg.image_size,
            val_split=cfg.get("val_split", 0.1),
            test_split=cfg.get("test_split", 0.1),
            seed=cfg.seed,
        )
    else:
        data_module = StanfordClassificationDataModule(
            data_dir=cfg.data_dir,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            image_size=cfg.image_size,
            val_split=cfg.get("val_split", 0.1),
            test_split=cfg.get("test_split", 0.1),
            seed=cfg.seed,
            split_dir=str(experiment_dirs["splits"]),
        )
    data_module.prepare_data()
    data_module.setup()

    # ── 5. Dataset statistics ────────────────────────────────────────
    if task == "classification":
        logger.info("Computing dataset statistics…")
        stats = compute_all_stats(data_module)
        display_stats_rich(stats)
        log_stats(stats, log=logger)
        save_stats_json(stats, experiment_dirs["stats"] / "dataset_stats.json")

    # ── 6. Model ─────────────────────────────────────────────────────
    logger.info("Setting up model…")
    if task == "segmentation":
        model = LitSegmentationModule(
            model_name=cfg.model_name,
            lr=cfg.lr,
            num_classes=2,
        )
    else:
        model = LitClassificationModule(
            model_name=cfg.model_name,
            num_classes=cfg.num_classes,
            lr=cfg.lr,
            optimizer_name=cfg.optimizer_name,
            weight_decay=cfg.get("weight_decay", 1e-4),
            scheduler_config=cfg.scheduler_config,
            regularization=cfg.get("regularization", {}),
            tensorboard=cfg.get("tensorboard", {}),
        )

    # ── 7. Trainer (callbacks + loggers wired automatically) ─────────
    logger.info("Setting up trainer…")
    trainer = build_trainer(cfg, experiment_dirs)

    # ── 8. Train ─────────────────────────────────────────────────────
    logger.info("Starting training…")
    ckpt_path = cfg.trainer.get("resume_from_checkpoint")
    if ckpt_path and not os.path.exists(ckpt_path):
        logger.warning("Checkpoint %s not found – starting from scratch.", ckpt_path)
        ckpt_path = None

    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)

    # ── 9. Evaluate ──────────────────────────────────────────────────
    logger.info("Running final evaluation…")
    trainer.test(model, datamodule=data_module)

    # ── 10. Report ───────────────────────────────────────────────────
    ckpt_cb = get_checkpoint_callback(trainer)
    if ckpt_cb:
        logger.info("Best model saved at: %s", ckpt_cb.best_model_path)
        val_acc = ckpt_cb.best_model_score
        return float(val_acc.item()) if val_acc is not None else 0.0

    logger.info("Training completed!")
    return 0.0


if __name__ == "__main__":
    train()
