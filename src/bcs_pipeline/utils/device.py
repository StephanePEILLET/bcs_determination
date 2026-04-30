"""Device detection utility for BCS pipeline.

Provides a single ``get_best_device()`` function that returns the best
available PyTorch device: CUDA → MPS (Apple Silicon) → CPU.

The device can be overridden via the ``BCS_DEVICE`` environment variable::

    BCS_DEVICE=cpu python app.py       # force CPU
    BCS_DEVICE=mps python app.py       # force MPS
    BCS_DEVICE=cuda python app.py      # force CUDA
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger("bcs_pipeline")


def get_best_device() -> torch.device:
    """Return the best available device: CUDA → MPS → CPU.

    If the environment variable ``BCS_DEVICE`` is set (e.g. ``cpu``, ``cuda``,
    ``mps``), it takes precedence over automatic detection.  This allows the
    setup script to force CPU mode when no GPU is available or when a specific
    backend causes issues.

    On Apple Silicon Macs, ``torch.backends.mps.is_available()`` will return
    ``True`` and the MPS backend will be used for GPU acceleration.

    Returns
    -------
    torch.device
        The selected device.
    """
    # Allow explicit override via environment variable
    forced = os.environ.get("BCS_DEVICE", "").strip().lower()
    if forced:
        device = torch.device(forced)
        logger.info("Device forced via BCS_DEVICE=%s", forced)
        return device

    # Auto-detection: CUDA → MPS → CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info("Selected device: %s", device)
    return device
