"""
BCS Determination Training Pipeline
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf

from bcs_pipeline.data.stanford_bcs_datamodule import StanfordBcsDataModule
from bcs_pipeline.lightning_module.bcs_determination_module import LitBcsDetermination
from bcs_pipeline.utils.config_utils import setup_experiment_dirs, validate_config
from bcs_pipeline.utils.logging_utils import setup_logging, log_experiment_info, print_config, print_config_rich


@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> float:
    # Validate configuration
    if not validate_config(cfg):
        raise ValueError("Configuration validation failed. Check your config.yaml.")

    # Setup experiment directories
    experiment_dirs = setup_experiment_dirs(cfg)
    
    # Setup logging
    logger = setup_logging(
        log_file=experiment_dirs["logs"] / "train.log",
        level=logging.INFO
    )
    
    # Log experiment info (this calls wandb initialization conditionally inside)
    log_experiment_info(logger, cfg, experiment_dirs)
    
    # Print config to console (Rich tree) and to log file
    print_config_rich(cfg)
    print_config(cfg, logger=logger)
    
    # Set random seeds for reproducibility
    pl.seed_everything(cfg.seed, workers=True)
    
    # Instantiate data module
    logger.info("Setting up data module...")
    data_module = StanfordBcsDataModule(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        image_size=cfg.image_size,
    )
    
    # Instantiate model
    logger.info("Setting up model...")
    model = LitBcsDetermination(
        model_name=cfg.model_name,
        num_classes=cfg.num_classes,
        lr=cfg.lr,
        optimizer_name=cfg.optimizer_name,
        weight_decay=cfg.get("weight_decay", 1e-4),
        scheduler_config=cfg.scheduler_config,
        regularization=cfg.get("regularization", {}),
        tensorboard=cfg.get("tensorboard", {}),
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
            name="bcs_determination",
            version=None
        )
        loggers.append(tensorboard_logger)
    
    if getattr(cfg, "use_wandb", False):
        try:
            wandb_logger = pl.loggers.WandbLogger(
                project=cfg.wandb_project,
                name=f"{cfg.model_name}_bcs",
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
        fast_dev_run=cfg.trainer.get("fast_dev_run", False),
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=cfg.log_every_n_steps,
        val_check_interval=cfg.val_check_interval,
        gradient_clip_val=cfg.get("gradient_clip_val", None),
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
