"""
Configuration utilities for the BCS Determination pipeline.

Handles experiment directory creation, configuration validation, and
**config snapshot saving** so that every experiment can be fully
reproduced later.

Key functions
-------------
* :func:`setup_experiment_dirs` – create the directory tree for an experiment.
* :func:`validate_config` – run Pydantic validation on the Hydra config.
* :func:`save_config_snapshot` – persist the *resolved* config to YAML + JSON.
* :func:`get_config_summary` – one-liner summary dict.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf

from .config_validation import validate_hydra_config, print_config_validation_summary


# ──────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────
def load_config(config_path: str = "configs", config_name: str = "config") -> DictConfig:
    """Load a YAML configuration file.

    Parameters
    ----------
    config_path:
        Directory containing the config file.
    config_name:
        Base name (without ``.yaml``).

    Returns
    -------
    DictConfig
    """
    return OmegaConf.load(os.path.join(config_path, f"{config_name}.yaml"))


# ──────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────
def validate_config(cfg: DictConfig) -> bool:
    """Validate configuration through Pydantic.

    Parameters
    ----------
    cfg:
        Hydra-managed configuration.

    Returns
    -------
    bool
        ``True`` if the config is valid.
    """
    try:
        validated = validate_hydra_config(cfg)
        print_config_validation_summary(validated)
        return True
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────
# Experiment directories
# ──────────────────────────────────────────────────────────────────────
def get_experiment_name(cfg: DictConfig) -> str:
    """Derive a deterministic experiment name from the config.

    Parameters
    ----------
    cfg:
        Hydra config.

    Returns
    -------
    str
        e.g. ``"resnet50_adam_cosine_annealing"``.
    """
    return f"{cfg.model_name}_{cfg.optimizer_name}_{cfg.scheduler_config['name']}"


def setup_experiment_dirs(cfg: DictConfig) -> Dict[str, Path]:
    """Create the full experiment directory tree.

    The tree includes sub-directories for checkpoints, logs, TensorBoard,
    W&B artefacts, the Hydra override snapshot, and the **split manifests**.

    Parameters
    ----------
    cfg:
        Hydra config.

    Returns
    -------
    dict[str, Path]
        Mapping from purpose → directory path.
    """
    experiment_name = get_experiment_name(cfg)
    experiments_dir = Path("experiments") / experiment_name

    dirs: Dict[str, Path] = {
        "root": experiments_dir,
        "checkpoints": experiments_dir / "checkpoints",
        "logs": experiments_dir / "logs",
        "tensorboard": experiments_dir / "tensorboard",
        "wandb": experiments_dir / "wandb",
        "hydra": experiments_dir / "hydra",
        "splits": experiments_dir / "splits",
        "stats": experiments_dir / "stats",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return dirs


# ──────────────────────────────────────────────────────────────────────
# Config snapshot (for reproducibility)
# ──────────────────────────────────────────────────────────────────────
def save_config_snapshot(cfg: DictConfig, experiment_dirs: Dict[str, Path]) -> Dict[str, Path]:
    """Save the **fully-resolved** config to both YAML and JSON.

    This snapshot captures *exactly* the configuration used for a particular
    run (including CLI overrides and Hydra interpolations).  It is essential
    for experiment reproducibility.

    Parameters
    ----------
    cfg:
        Hydra config.
    experiment_dirs:
        Mapping returned by :func:`setup_experiment_dirs`.

    Returns
    -------
    dict[str, Path]
        ``{"yaml": ..., "json": ...}`` paths to the saved files.
    """
    hydra_dir = experiment_dirs["hydra"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # YAML (human-readable)
    yaml_path = hydra_dir / f"config_{timestamp}.yaml"
    OmegaConf.save(cfg, yaml_path)

    # JSON (machine-readable, easier to diff)
    json_path = hydra_dir / f"config_{timestamp}.json"
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["_snapshot_meta"] = {
        "saved_at": datetime.now().isoformat(),
        "experiment_name": get_experiment_name(cfg),
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(cfg_dict, fh, indent=2, ensure_ascii=False, default=str)

    return {"yaml": yaml_path, "json": json_path}


# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────
def get_config_summary(cfg: DictConfig) -> Dict[str, Any]:
    """Return a flat summary dict of the most important config values.

    Parameters
    ----------
    cfg:
        Hydra config.

    Returns
    -------
    dict
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
        "use_wandb": getattr(cfg, "use_wandb", False),
    }