"""
Logging utilities for the Stanford BCS pipeline.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import prodict
import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf
from prodict import Prodict
from rich import get_console
from rich.console import Console
from rich.logging import RichHandler
from rich.style import Style
from rich.tree import Tree


class _PrettySafeLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))


_PrettySafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", _PrettySafeLoader.construct_python_tuple)


def resolve_tuple(*args):
    """Resolves a value in a config file with a structure like ${tuple:1,2} to a tuple (1, 2)."""
    return tuple(args)


OmegaConf.register_new_resolver("tuple", resolve_tuple, replace=True)


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
    logger = logging.getLogger("bcs_pipeline")
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


def get_logger(name: str = "bcs_pipeline") -> logging.Logger:
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
    logger.info(f"Scheduler: {cfg.scheduler_config['name']}")
    logger.info(f"Learning Rate: {cfg.lr}")
    logger.info(f"Batch Size: {cfg.batch_size}")
    logger.info(f"Max Epochs: {cfg.max_epochs}")


def print_config(config: DictConfig | prodict.Prodict | str, logger: Optional[logging.Logger] = None) -> None:
    """
    Prints a yaml configuration file to the console and/or logger.

    Args:
        config: string or dict, path of the yaml file or previously imported configuration file.
        logger: logger instance.
    """
    if isinstance(config, Prodict):
        config = config.to_dict(is_recursive=True)
    elif isinstance(config, str):
        config = read_config(config)
    elif isinstance(config, DictConfig):
        config = OmegaConf.to_container(config)

    if logger:
        logger.info(yaml.dump(config, indent=4, default_flow_style=False, sort_keys=False, allow_unicode=True))
    else:
        yaml.dump(config, sys.stdout, indent=4, default_flow_style=False, sort_keys=False, allow_unicode=True)


def read_config(file: str) -> DictConfig:
    """
    Reads a yaml configuration file.

    Args:
        file: str, path of the yaml configuration file.

    Returns:
        dict (DictConfig), imported yaml file.
    """
    if not os.path.exists(file):
        raise FileNotFoundError(f"ERROR: Cannot find the file {file}\n")
    try:
        with open(file, "r", encoding="utf-8") as f:
            config = yaml.load(f, Loader=_PrettySafeLoader)
    except yaml.YAMLError as e:
        raise RuntimeError(f"ERROR: Cannot load the file {file}\n") from e

    return OmegaConf.create(config)


def write_config(data: DictConfig | prodict.Prodict, outfile: str) -> None:
    """
    Writes the dictionary data to a yaml file.

    Args:
        data: dict (DictConfig), data to be stored as a yaml file.
        outfile: str, path of the output file.
    """
    with open(outfile, "w", encoding="utf-8") as f:
        if isinstance(data, Prodict):
            yaml.dump(data.to_dict(is_recursive=True), f, indent=4, default_flow_style=None, sort_keys=False, allow_unicode=True)
        elif isinstance(data, DictConfig):
            OmegaConf.save(config=data, f=f)
        else:
            yaml.dump(data, f, indent=4, default_flow_style=None, sort_keys=False, allow_unicode=True)


def print_config_rich(config: DictConfig) -> None:
    """Print content of given config using Rich library and its tree structure.

    Args:
        config: Config to print to console using a Rich tree.
    """
    def walk_config(tree: Tree, config: DictConfig):
        """Recursive function to accumulate branch."""
        for group_name, group_option in config.items():
            if isinstance(group_option, dict):
                branch = tree.add(str(group_name), style=Style(color="pink1", bold=True))
                walk_config(branch, group_option)
            elif isinstance(group_option, ListConfig):
                if not group_option:
                    tree.add(f"{group_name}: []", style=Style(color="pink1", bold=True))
                else:
                    tree.add(f"{str(group_name)}: {group_option}", style=Style(color="pink1", bold=True))
            else:
                if group_name == "_target_":
                    tree.add(f"{str(group_name)}: {group_option}", style=Style(color="white", italic=True, bold=True))
                else:
                    tree.add(f"{str(group_name)}: {group_option}", style=Style(color="pink1", bold=True))

    tree = Tree(
        ":rainbow: Configuration Tree :sun_behind_cloud:",
        style=Style(color="white", bold=True, encircle=True),
        guide_style=Style(color="sky_blue1", bold=True),
        expanded=True,
        highlight=True,
    )
    walk_config(tree, config)
    get_console().print(tree)