#!/usr/bin/env python3
"""
Stanford Dogs Training Pipeline
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf
from rich import print as rprint
from rich.tree import Tree
from rich.syntax import Syntax
from pytorch_lightning.profilers import PyTorchProfiler

from dogs_pipeline.data.stanford_dogs_datamodule import StanfordDogsDataModule
from dogs_pipeline.lightning_module.stanford_dogs_module import LitStanfordDogs
from dogs_pipeline.utils.config_utils import setup_experiment_dirs, validate_config
from dogs_pipeline.utils.logging_utils import setup_logging, log_experiment_info


@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> float:
    # Setup experiment directories
    experiment_dirs = setup_experiment_dirs(cfg)
    
    # Setup logging
    logger = setup_logging(
        log_file=experiment_dirs["logs"] / "train.log",
        level=logging.INFO
    )
    
    # Log experiment info (this calls wandb initialization conditionally inside)
    log_experiment_info(logger, cfg, experiment_dirs)
    
    # Print config tree
    tree = Tree("TRAINING CONFIG", guide_style="bold bright_blue")
    yaml_config = OmegaConf.to_yaml(cfg, resolve=True)
    syntax = Syntax(yaml_config, "yaml", theme="monokai", line_numbers=False)
    tree.add(syntax)
    rprint(tree)
    
    # Set random seeds for reproducibility
    pl.seed_everything(cfg.seed, workers=True)
    
    # Instantiate data module
    logger.info("Setting up data module...")
    data_module = StanfordDogsDataModule(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        image_size=cfg.image_size,
    )
    
    # Instantiate model
    logger.info("Setting up model...")
    model = LitStanfordDogs(
        model_name=cfg.model_name,
        num_classes=cfg.num_classes,
        lr=cfg.lr,
        optimizer_name=cfg.optimizer_name,
        scheduler_config=cfg.scheduler_config
    )
    
    # Setup callbacks
    callbacks = []
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=experiment_dirs["checkpoints"],
        filename="epoch={epoch:02d}-val_acc={val_acc:.2f}-{step}",
        monitor="val_acc",
        mode="max",
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val_acc",
        patience=cfg.patience,
        mode="max",
        verbose=True
    )
    callbacks.append(early_stopping_callback)
    
    if cfg.use_tensorboard or getattr(cfg, "use_wandb", False):
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)
    
    # Setup loggers
    loggers = []
    if cfg.use_tensorboard:
        tensorboard_logger = pl.loggers.TensorBoardLogger(
            save_dir=experiment_dirs["tensorboard"],
            name="stanford_dogs",
            version=None
        )
        loggers.append(tensorboard_logger)
    
    if getattr(cfg, "use_wandb", False):
        try:
            wandb_logger = pl.loggers.WandbLogger(
                project=cfg.wandb_project,
                name=f"{cfg.model_name}_dogs",
                save_dir=experiment_dirs["wandb"],
                log_model=True
            )
            loggers.append(wandb_logger)
        except ImportError:
            logger.warning("wandb is not installed, skipping wandb logging")
    
    # Setup trainer
    logger.info("Setting up trainer...")
    trainer_kwargs = dict(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.get("strategy", "auto"),
        sync_batchnorm=cfg.trainer.get("sync_batchnorm", False),
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=cfg.log_every_n_steps,
        val_check_interval=cfg.val_check_interval,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True,
        precision=cfg.precision
    )
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Train the model
    logger.info("Starting training...")
    ckpt_path = cfg.trainer.get("resume_from_checkpoint")
    if ckpt_path and not os.path.exists(ckpt_path):
        logger.warning(f"Checkpoint {ckpt_path} not found. Starting from scratch.")
        ckpt_path = None
        
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)
    
    # Test the model
    logger.info("Running final evaluation...")
    trainer.test(model, datamodule=data_module)
    
    # Log final results
    logger.info("Training completed!")
    logger.info(f"Best model saved at: {checkpoint_callback.best_model_path}")
    
    val_acc = checkpoint_callback.best_model_score
    return float(val_acc.item()) if val_acc is not None else 0.0

if __name__ == "__main__":
    train()
