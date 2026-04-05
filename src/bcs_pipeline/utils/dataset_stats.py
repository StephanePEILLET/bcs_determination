"""
Dataset statistics computation and display for BCS Determination pipeline.

Computes per-class and global statistics on the train / val / test splits
and renders them with **Rich** tables in the console (and optionally logs
them to a file).

Key functions
-------------
* :func:`compute_split_stats` – per-class counts + global metrics for one split.
* :func:`compute_all_stats` – convenience wrapper over all three splits.
* :func:`display_stats_rich` – pretty Rich tables in the console.
* :func:`save_stats_json` – persist the stats dict to JSON.
* :func:`log_stats` – write a plain-text version to a ``logging.Logger``.

Usage
-----
>>> from bcs_pipeline.utils.dataset_stats import compute_all_stats, display_stats_rich
>>> stats = compute_all_stats(data_module)
>>> display_stats_rich(stats)
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

logger = logging.getLogger("bcs_pipeline")


# ──────────────────────────────────────────────────────────────────────
# Core computation
# ──────────────────────────────────────────────────────────────────────
def compute_split_stats(
    labels: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """Compute per-class and aggregate statistics for a single split.

    Parameters
    ----------
    labels:
        1-D array of integer class labels.
    class_names:
        Ordered list mapping class index → human-readable name.

    Returns
    -------
    dict
        ``{"total_samples", "num_classes", "per_class": [...],
        "min_count", "max_count", "mean_count", "median_count",
        "std_count", "imbalance_ratio"}``.
    """
    counter = Counter(labels.tolist())
    num_classes = len(class_names)

    per_class: List[Dict[str, Any]] = []
    counts: List[int] = []
    for idx in range(num_classes):
        count = counter.get(idx, 0)
        counts.append(count)
        # Extract short breed name (strip synset prefix like "n02085620-")
        raw_name = class_names[idx]
        short_name = "-".join(raw_name.split("-")[1:]) if "-" in raw_name else raw_name
        per_class.append({
            "class_id": idx,
            "class_name": short_name,
            "raw_name": raw_name,
            "count": count,
            "percentage": round(100.0 * count / len(labels), 2) if len(labels) > 0 else 0.0,
        })

    counts_arr = np.array(counts)
    min_c = int(counts_arr.min()) if len(counts_arr) > 0 else 0
    max_c = int(counts_arr.max()) if len(counts_arr) > 0 else 0

    return {
        "total_samples": int(len(labels)),
        "num_classes": num_classes,
        "per_class": per_class,
        "min_count": min_c,
        "max_count": max_c,
        "mean_count": round(float(counts_arr.mean()), 2),
        "median_count": round(float(np.median(counts_arr)), 2),
        "std_count": round(float(counts_arr.std()), 2),
        "imbalance_ratio": round(max_c / max(min_c, 1), 2),
    }


def compute_all_stats(data_module) -> Dict[str, Dict[str, Any]]:
    """Compute statistics for **train**, **val**, and **test** splits.

    Parameters
    ----------
    data_module:
        A :class:`StanfordBcsDataModule` that has already been ``setup()``.

    Returns
    -------
    dict
        ``{"train": {...}, "val": {...}, "test": {...}}``.
    """
    stats: Dict[str, Dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        labels = data_module.get_split_labels(split)
        stats[split] = compute_split_stats(labels, data_module.classes)
    return stats


# ──────────────────────────────────────────────────────────────────────
# Rich display
# ──────────────────────────────────────────────────────────────────────
def _make_summary_table(stats: Dict[str, Dict[str, Any]]) -> Table:
    """Build a Rich table summarising all splits at a glance."""
    table = Table(
        title="📊 Dataset Split Summary",
        title_style="bold magenta",
        header_style="bold cyan",
        border_style="bright_blue",
        show_lines=True,
    )
    table.add_column("Split", style="bold white", justify="center")
    table.add_column("Samples", justify="right")
    table.add_column("Classes", justify="right")
    table.add_column("Min / class", justify="right")
    table.add_column("Max / class", justify="right")
    table.add_column("Mean ± Std", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("Imbalance ratio", justify="right")

    for split in ("train", "val", "test"):
        s = stats[split]
        table.add_row(
            split.upper(),
            str(s["total_samples"]),
            str(s["num_classes"]),
            str(s["min_count"]),
            str(s["max_count"]),
            f"{s['mean_count']:.1f} ± {s['std_count']:.1f}",
            str(s["median_count"]),
            f"{s['imbalance_ratio']:.2f}",
        )
    return table


def _make_class_table(split_name: str, split_stats: Dict[str, Any], top_n: int = 10) -> Table:
    """Build a Rich table showing the top-N and bottom-N classes for a split."""
    per_class = sorted(split_stats["per_class"], key=lambda x: x["count"], reverse=True)

    table = Table(
        title=f"🔍 {split_name.upper()} – Top / Bottom {top_n} classes",
        title_style="bold green",
        header_style="bold yellow",
        border_style="bright_green",
        show_lines=False,
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("Class", style="white")
    table.add_column("Count", justify="right")
    table.add_column("Pct (%)", justify="right")
    table.add_column("Bar", min_width=20)

    max_count = split_stats["max_count"] or 1

    # Top N
    for i, entry in enumerate(per_class[:top_n], 1):
        bar_len = int(20 * entry["count"] / max_count)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        table.add_row(
            str(i), entry["class_name"],
            str(entry["count"]), f"{entry['percentage']:.1f}",
            f"[green]{bar}[/green]",
        )

    if len(per_class) > 2 * top_n:
        table.add_row("…", "…", "…", "…", "…", style="dim")

    # Bottom N
    for i, entry in enumerate(per_class[-top_n:], len(per_class) - top_n + 1):
        bar_len = int(20 * entry["count"] / max_count)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        table.add_row(
            str(i), entry["class_name"],
            str(entry["count"]), f"{entry['percentage']:.1f}",
            f"[red]{bar}[/red]",
        )

    return table


def display_stats_rich(
    stats: Dict[str, Dict[str, Any]],
    top_n: int = 10,
    console: Console | None = None,
) -> None:
    """Render dataset statistics as pretty Rich tables.

    Parameters
    ----------
    stats:
        Output of :func:`compute_all_stats`.
    top_n:
        How many top / bottom classes to show per split.
    console:
        Optional existing Console instance (for testing).
    """
    if console is None:
        console = Console()

    console.print()
    console.print(Panel(
        Text("BCS DETERMINATION – Dataset Statistics", style="bold white"),
        style="bold bright_blue",
        expand=False,
    ))
    console.print()

    # Global summary
    console.print(_make_summary_table(stats))
    console.print()

    # Per-split details (train and val only – test stays hidden)
    for split in ("train", "val"):
        console.print(_make_class_table(split, stats[split], top_n=top_n))
        console.print()


# ──────────────────────────────────────────────────────────────────────
# Logging (plain text for log files)
# ──────────────────────────────────────────────────────────────────────
def log_stats(
    stats: Dict[str, Dict[str, Any]],
    log: logging.Logger | None = None,
) -> None:
    """Write a plain-text summary of dataset stats to a logger.

    Parameters
    ----------
    stats:
        Output of :func:`compute_all_stats`.
    log:
        Logger instance.  Falls back to the module-level ``bcs_pipeline`` logger.
    """
    if log is None:
        log = logger

    log.info("=" * 60)
    log.info("DATASET STATISTICS")
    log.info("=" * 60)

    for split in ("train", "val", "test"):
        s = stats[split]
        log.info(
            "  %-5s │ samples=%-6d │ classes=%-3d │ min=%-3d │ max=%-3d │ "
            "mean=%.1f ± %.1f │ imbalance=%.2f",
            split.upper(), s["total_samples"], s["num_classes"],
            s["min_count"], s["max_count"], s["mean_count"],
            s["std_count"], s["imbalance_ratio"],
        )

    log.info("=" * 60)


# ──────────────────────────────────────────────────────────────────────
# JSON persistence
# ──────────────────────────────────────────────────────────────────────
def save_stats_json(
    stats: Dict[str, Dict[str, Any]],
    path: Path | str,
) -> Path:
    """Save statistics to a JSON file for later analysis.

    Parameters
    ----------
    stats:
        Output of :func:`compute_all_stats`.
    path:
        Destination file path.

    Returns
    -------
    Path
        The path of the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    logger.info("Dataset statistics saved → %s", path)
    return path
