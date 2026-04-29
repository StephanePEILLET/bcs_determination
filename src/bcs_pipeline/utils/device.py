"""Device detection utility for BCS pipeline.

Provides a single ``get_best_device()`` function that returns the best
available PyTorch device: CUDA → MPS (Apple Silicon) → CPU.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger("bcs_pipeline")


def get_best_device() -> torch.device:
    """Return the best available device: CUDA → MPS → CPU.

    On Apple Silicon Macs, ``torch.backends.mps.is_available()`` will return
    ``True`` and the MPS backend will be used for GPU acceleration.

    Returns
    -------
    torch.device
        The selected device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info("Selected device: %s", device)
    return device
