"""
Configuration utilities for the Stanford BCS pipeline.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from omegaconf import DictConfig, OmegaConf
import hydra
from .config_validation import validate_hydra_config, print_config_validation_summary


def load_config(config_path: str = "configs", config_name: str = "config") -> DictConfig:
    """
    Load configuration using Hydra.
    
    Args:
        config_path: Path to config directory
        config_name: Name of the config file
        
    Returns:
        Loaded configuration
    """
    return OmegaConf.load(os.path.join(config_path, f"{config_name}.yaml"))


def validate_config(cfg: DictConfig) -> bool:
    """
    Validate configuration using Pydantic.
    
    Args:
        cfg: Configuration object from Hydra
        
    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        # Convert DictConfig to dict and validate with Pydantic
        validated_config = validate_hydra_config(cfg)
        
        # Print validation summary
        print_config_validation_summary(validated_config)
        
        return True
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


def get_experiment_dir(cfg: DictConfig) -> Path:
    """
    Get experiment directory path.
    
    Args:
        cfg: Configuration
        
    Returns:
        Path to experiment directory
    """
    base_dir = Path("experiments")
    experiment_name = f"{cfg.model_name}_{cfg.optimizer_name}_{cfg.scheduler_config['name']}"  
    return base_dir / experiment_name


def setup_experiment_dirs(cfg: DictConfig) -> Dict[str, Path]:
    """
    Setup experiment directories for outputs.
    
    Args:
        cfg: Configuration object
        
    Returns:
        Dictionary with experiment directory paths
    """
    # Create experiment name based on configuration
    experiment_name = f"{cfg.model_name}_{cfg.optimizer_name}_{cfg.scheduler_config['name']}"
    
    # Base experiments directory
    experiments_dir = Path("experiments") / experiment_name
    
    # Create subdirectories
    dirs = {
        "checkpoints": experiments_dir / "checkpoints",
        "logs": experiments_dir / "logs",
        "tensorboard": experiments_dir / "tensorboard",
        "wandb": experiments_dir / "wandb",
        "hydra": experiments_dir / "hydra"
    }
    
    # Create directories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def get_config_summary(cfg: DictConfig) -> dict:
    """
    Get a summary of the configuration.
    
    Args:
        cfg: Configuration object
        
    Returns:
        Dictionary with configuration summary
    """
    return {
        "model": cfg.model_name,
        "optimizer": cfg.optimizer_name,
        "scheduler": cfg.scheduler_config["name"],
        "lr": cfg.lr,
        "batch_size": cfg.batch_size,
        "max_epochs": cfg.max_epochs,
        "patience": cfg.patience,
        "precision": cfg.precision,
        "accelerator": cfg.trainer.accelerator,
        "use_tensorboard": cfg.use_tensorboard,
        "use_wandb": cfg.use_wandb
    } 