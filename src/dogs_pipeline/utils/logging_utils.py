"""
Logging utilities for the MNIST pipeline.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from rich.logging import RichHandler
from rich.console import Console


def setup_logging(
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    use_rich: bool = True
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file
        level: Logging level
        use_rich: Whether to use rich formatting
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger("dogs_pipeline")
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    if use_rich:
        formatter = logging.Formatter(
            "%(message)s",
            datefmt="[%X]"
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    # Console handler
    if use_rich:
        console_handler = RichHandler(
            console=Console(),
            show_time=True,
            show_path=False
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "dogs_pipeline") -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_experiment_info(logger: logging.Logger, cfg, experiment_dirs: dict):
    """
    Log experiment information.
    
    Args:
        logger: Logger instance
        cfg: Configuration
        experiment_dirs: Experiment directories
    """
    logger.info("=" * 50)
    logger.info("EXPERIMENT SETUP")
    logger.info("=" * 50)
    logger.info(f"Model: {cfg.model_name}")
    logger.info(f"Optimizer: {cfg.optimizer_name}")
    logger.info(f"Scheduler: {cfg.scheduler_config.name}")
    logger.info(f"Learning Rate: {cfg.lr}")
    logger.info(f"Batch Size: {cfg.batch_size}")
    logger.info(f"Max Epochs: {cfg.max_epochs}")
    logger.info("=" * 50)
    logger.info("DIRECTORIES:")
    for name, path in experiment_dirs.items():
        logger.info(f"  {name}: {path}")
    logger.info("=" * 50) 