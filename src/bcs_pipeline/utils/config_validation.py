"""
Configuration validation using Pydantic and dataclasses.

This module provides type-safe configuration validation for the Stanford BCS pipeline.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
import torch


@dataclass
class SchedulerConfig:
    """Configuration for learning rate schedulers."""
    name: Literal["none", "cosine_annealing", "reduce_lr_on_plateau"]
    T_max: Optional[int] = None
    monitor: str = "val_loss"
    factor: float = 0.1
    patience: int = 3
    verbose: bool = True


@dataclass
class FinetuningConfig:
    """Configuration for two-phase transfer learning fine-tuning."""
    unfreeze_at_epoch: int = 5
    backbone_lr_multiplier: float = 0.1


@dataclass
class TrainerConfig:
    """Configuration for PyTorch Lightning trainer."""
    accelerator: Literal["auto", "cpu", "gpu", "tpu"] = "auto"
    devices: Union[int, Literal["auto"]] = "auto"


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    checkpoint_path: Optional[str] = None
    batch_size: int = 256
    save_results: bool = True


class BCSConfig(BaseModel):
    """
    Main configuration class for the Stanford BCS pipeline.
    
    This class validates all configuration parameters using Pydantic.
    """
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra="ignore",  # Ignore extra fields from Hydra (e.g. n_trials, hydra internals)
        use_enum_values=True
    )
    
    # Model configuration
    task: Literal["classification", "segmentation"] = Field(
        default="classification",
        description="Task to perform (classification or segmentation)"
    )
    model_name: Literal["resnet50", "vit"] = Field(
        default="resnet50",
        description="Name of the model architecture to use"
    )
    
    # Training configuration
    lr: float = Field(
        default=0.001,
        gt=0.0,
        le=1.0,
        description="Learning rate"
    )
    optimizer_name: Literal["adam", "sgd"] = Field(
        default="adam",
        description="Optimizer to use"
    )
    scheduler_config: Dict[str, Any] = Field(
        default_factory=lambda: {"name": "cosine_annealing", "T_max": 10},
        description="Learning rate scheduler configuration"
    )
    
    # Data configuration
    data_dir: str = Field(
        default="data/",
        description="Directory for Stanford BCS dataset"
    )
    image_size: int = Field(
        default=224,
        gt=0,
        description="Image size for training"
    )
    num_classes: int = Field(
        default=120,
        gt=0,
        description="Number of classes in the dataset"
    )
    batch_size: int = Field(
        default=64,
        gt=0,
        le=1024,
        description="Batch size for training"
    )
    num_workers: int = Field(
        default=4,
        ge=0,
        le=32,
        description="Number of workers for data loading"
    )
    val_split: float = Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
        description="Fraction of data for validation (stratified)"
    )
    test_split: float = Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
        description="Fraction of data for test (stratified)"
    )
    
    # Training parameters
    finetuning: Optional[Union[Dict[str, Any], FinetuningConfig]] = Field(
        default=None,
        description="Configuration for transfer learning fine-tuning"
    )
    max_epochs: int = Field(
        default=10,
        gt=0,
        le=1000,
        description="Maximum number of training epochs"
    )
    patience: int = Field(
        default=5,
        gt=0,
        le=100,
        description="Patience for early stopping"
    )
    log_every_n_steps: int = Field(
        default=50,
        gt=0,
        description="Log metrics every N steps"
    )
    val_check_interval: float = Field(
        default=1.0,
        gt=0.0,
        description="Validation check interval"
    )
    precision: Literal["16-mixed", "32", "64"] = Field(
        default="16-mixed",
        description="Training precision"
    )
    
    # Logging configuration
    use_tensorboard: bool = Field(
        default=True,
        description="Enable TensorBoard logging"
    )
    use_wandb: bool = Field(
        default=True,
        description="Enable Weights & Biases logging"
    )
    wandb_project: str = Field(
        default="stanford_bcs_classification",
        description="Weights & Biases project name"
    )
    
    # Trainer configuration
    trainer: Dict[str, Any] = Field(
        default_factory=lambda: {"accelerator": "auto", "devices": "auto"},
        description="PyTorch Lightning trainer configuration"
    )
    
    # Inference configuration
    inference: Dict[str, Any] = Field(
        default_factory=lambda: {"checkpoint_path": None, "batch_size": 256, "save_results": True},
        description="Inference configuration"
    )
    
    # Reproducibility
    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility"
    )
    
    @field_validator('scheduler_config')
    @classmethod
    def validate_scheduler_config(cls, v):
        """Validate scheduler configuration based on scheduler type."""
        if v.get("name") == "cosine_annealing" and v.get("T_max") is None:
            raise ValueError("T_max must be specified for cosine_annealing scheduler")
        return v
    
    @model_validator(mode='after')
    def validate_cross_fields(self):
        """Validate cross-field dependencies."""
        # Validate batch size based on model
        if self.model_name == "vit" and self.batch_size > 128:
            raise ValueError("Batch size too large for ViT. Use <= 128.")
        
        # Validate learning rate based on optimizer
        if self.optimizer_name == "sgd" and self.lr > 0.1:
            raise ValueError("Learning rate too high for SGD. Use <= 0.1.")
        
        # Validate trainer configuration
        if self.trainer.get("accelerator") == "gpu" and not torch.cuda.is_available():
            raise ValueError("GPU requested but CUDA is not available")
        
        return self


def validate_hydra_config(cfg_dict: dict) -> BCSConfig:
    """
    Validate Hydra configuration dictionary.
    
    Args:
        cfg_dict: Configuration dictionary from Hydra
        
    Returns:
        Validated BCSConfig object
        
    Raises:
        ValueError: If configuration is invalid
    """
    try:
        # Convert nested dataclasses to dicts for Pydantic
        config_dict = _convert_dataclasses_to_dict(cfg_dict)
        return BCSConfig(**config_dict)
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}")


def _convert_dataclasses_to_dict(obj):
    """Recursively convert dataclasses to dictionaries."""
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _convert_dataclasses_to_dict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, dict):
        return {k: _convert_dataclasses_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_dataclasses_to_dict(item) for item in obj]
    else:
        return obj


def get_config_schema() -> dict:
    """
    Get the JSON schema for the configuration.
    
    Returns:
        JSON schema dictionary
    """
    return BCSConfig.model_json_schema()


def print_config_validation_summary(config: BCSConfig) -> None:
    """
    Print a summary of the validated configuration.
    
    Args:
        config: Validated BCSConfig object
    """
    print("=" * 50)
    print("CONFIGURATION VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Model: {config.model_name}")
    print(f"Optimizer: {config.optimizer_name}")
    print(f"Scheduler: {config.scheduler_config.get('name', 'unknown')}")
    print(f"Learning Rate: {config.lr}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Max Epochs: {config.max_epochs}")
    print(f"Precision: {config.precision}")
    print(f"Accelerator: {config.trainer.get('accelerator', 'unknown')}")
    print(f"TensorBoard: {config.use_tensorboard}")
    print(f"Weights & Biases: {config.use_wandb}")
    print("=" * 50) 