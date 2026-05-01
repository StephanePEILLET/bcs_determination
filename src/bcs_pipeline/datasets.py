"""Dataset discovery and path resolution for Body Pawsitive.

Centralises the constants, file-listing helpers, and group/image
resolution logic shared between the web app (``app.py``) and the
command-line preloader (``scripts/preload_db.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from bcs_pipeline.app_checkpoints import REPO_ROOT

STANFORD_ROOT = REPO_ROOT / "data/stanford_dogs/images"
STANFORD_IMAGES = STANFORD_ROOT / "Images"
OXFORD_ROOT = REPO_ROOT / "data/Oxford-IIIT_pet_dataset"
OXFORD_IMAGES = OXFORD_ROOT / "images"
REDDIT_DIR = REPO_ROOT / "data/Reddit_example"

IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def list_image_files(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    )


def collect_all_images() -> List[Tuple[Path, str, str, Optional[str]]]:
    entries: List[Tuple[Path, str, str, Optional[str]]] = []

    if STANFORD_IMAGES.is_dir():
        for breed_dir in sorted(p for p in STANFORD_IMAGES.iterdir() if p.is_dir()):
            breed_name = (
                breed_dir.name.split("-", 1)[1]
                if "-" in breed_dir.name
                else breed_dir.name
            )
            for img_path in list_image_files(breed_dir):
                entries.append((img_path, "Stanford Dogs", breed_name, breed_name))

    if OXFORD_IMAGES.is_dir():
        for img_path in list_image_files(OXFORD_IMAGES):
            prefix = "_".join(img_path.stem.split("_")[:-1])
            ground_truth = prefix.replace("_", " ")
            entries.append((img_path, "Oxford-IIIT Pet", prefix, ground_truth))

    if REDDIT_DIR.is_dir():
        for img_path in list_image_files(REDDIT_DIR):
            entries.append((img_path, "Reddit", "all", None))

    return entries


def _stanford_groups() -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    if not STANFORD_IMAGES.is_dir():
        return groups
    for breed_dir in sorted(p for p in STANFORD_IMAGES.iterdir() if p.is_dir()):
        breed_name = (
            breed_dir.name.split("-", 1)[1]
            if "-" in breed_dir.name
            else breed_dir.name
        )
        groups[breed_name] = [p.name for p in list_image_files(breed_dir)]
    return groups


def _oxford_groups() -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for img_path in list_image_files(OXFORD_IMAGES):
        prefix = "_".join(img_path.stem.split("_")[:-1])
        groups.setdefault(prefix, []).append(img_path.name)
    return {k: sorted(v) for k, v in sorted(groups.items())}


def _reddit_groups() -> Dict[str, List[str]]:
    files = list_image_files(REDDIT_DIR)
    return {"all": [f.name for f in files]} if files else {}


def get_datasets() -> Dict[str, Dict[str, List[str]]]:
    return {
        "Reddit": _reddit_groups(),
        "Stanford Dogs": _stanford_groups(),
        "Oxford-IIIT Pet": _oxford_groups(),
    }


def resolve_image_path(dataset: str, group: str, filename: str) -> Optional[Path]:
    if dataset == "Reddit":
        return REDDIT_DIR / filename
    if dataset == "Stanford Dogs":
        for breed_dir in STANFORD_IMAGES.iterdir():
            if not breed_dir.is_dir():
                continue
            breed_name = (
                breed_dir.name.split("-", 1)[1]
                if "-" in breed_dir.name
                else breed_dir.name
            )
            if breed_name == group:
                return breed_dir / filename
        return None
    if dataset == "Oxford-IIIT Pet":
        return OXFORD_IMAGES / filename
    return None


def ground_truth(dataset: str, group: str) -> Optional[str]:
    if dataset == "Reddit":
        return None
    if dataset == "Stanford Dogs":
        return group if group else None
    if dataset == "Oxford-IIIT Pet":
        return group.replace("_", " ") if group else None
    return None
