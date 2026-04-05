"""
LightningDataModule for the Stanford Dogs (BCS) dataset.

Key features
------------
* **Stratified splitting** – uses ``sklearn.model_selection.StratifiedShuffleSplit``
  so that every class is represented proportionally in train / val / test.
* **Reproducible splits** – split indices are persisted to a JSON *manifest*
  file.  When a manifest already exists for the same ``(val_split, test_split,
  seed)`` combination, it is reloaded verbatim, guaranteeing identical data
  across runs.
* **Deterministic workers** – a per-worker ``seed_worker`` function and an
  explicit ``Generator`` are passed to every ``DataLoader``.

Usage
-----
>>> dm = StanfordBcsDataModule(data_dir="data/stanford_dogs", split_dir="experiments/run1/splits")
>>> dm.prepare_data()   # downloads if needed
>>> dm.setup()          # creates or loads the stratified split
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import urlretrieve

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

logger = logging.getLogger("bcs_pipeline")

DATASET_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"


# ──────────────────────────────────────────────────────────────────────
# Dataset wrapper that applies a specific transform per split
# ──────────────────────────────────────────────────────────────────────
class _SubsetWithTransform(torch.utils.data.Dataset):
    """Wraps a ``torch.utils.data.Subset`` and applies a transform lazily.

    Parameters
    ----------
    subset:
        A ``Subset`` or any map-style dataset returning ``(image, label)``.
    transform:
        torchvision transform pipeline to apply to each image.
    """

    def __init__(self, subset: Subset, transform: transforms.Compose | None = None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index: int):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self) -> int:
        return len(self.subset)


# ──────────────────────────────────────────────────────────────────────
# Worker seeding for reproducible DataLoaders
# ──────────────────────────────────────────────────────────────────────
def _seed_worker(worker_id: int) -> None:
    """Seed each DataLoader worker for reproducibility.

    Called automatically by ``DataLoader(worker_init_fn=...)``.
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)


# ──────────────────────────────────────────────────────────────────────
# Main DataModule
# ──────────────────────────────────────────────────────────────────────
class StanfordBcsDataModule(LightningDataModule):
    """LightningDataModule for the Stanford Dogs dataset with **stratified** splits.

    Parameters
    ----------
    data_dir:
        Root directory containing (or that will contain) ``Images/``.
    batch_size:
        Mini-batch size for all DataLoaders.
    num_workers:
        Number of parallel data-loading workers.
    image_size:
        Spatial resolution fed to the model.
    val_split:
        Fraction of data reserved for **validation**.
    test_split:
        Fraction of data reserved for **test**.
    seed:
        Random seed for the stratified split.
    split_dir:
        Directory where the split manifest JSON is saved / loaded.
        Defaults to ``<data_dir>/splits/``.
    """

    def __init__(
        self,
        data_dir: str = "data/stanford_dogs",
        batch_size: int = 64,
        num_workers: int = 4,
        image_size: int = 224,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
        split_dir: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.split_dir = Path(split_dir) if split_dir else Path(data_dir) / "splits"

        # ImageNet normalisation statistics
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            self.normalize,
        ])

        self.val_transforms = transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            self.normalize,
        ])

        self.data_train: Optional[_SubsetWithTransform] = None
        self.data_val: Optional[_SubsetWithTransform] = None
        self.data_test: Optional[_SubsetWithTransform] = None

        self.num_classes: int = 120
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}

        # Populated after setup()
        self.train_indices: List[int] = []
        self.val_indices: List[int] = []
        self.test_indices: List[int] = []
        self.split_manifest_path: Optional[Path] = None

    # ──────────────────────────────────────────────────────────────
    # Download
    # ──────────────────────────────────────────────────────────────
    def prepare_data(self) -> None:
        """Download and extract the dataset if not already present."""
        images_dir = os.path.join(self.data_dir, "Images")
        if os.path.isdir(images_dir):
            return

        os.makedirs(self.data_dir, exist_ok=True)
        tar_path = os.path.join(self.data_dir, "images.tar")

        if not os.path.isfile(tar_path):
            logger.info("Downloading Stanford Dogs → %s …", tar_path)
            urlretrieve(DATASET_URL, tar_path)
            logger.info("Download complete.")

        logger.info("Extracting %s …", tar_path)
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=self.data_dir)
        logger.info("Dataset ready at %s", images_dir)

        if os.path.isfile(tar_path):
            os.remove(tar_path)

    # ──────────────────────────────────────────────────────────────
    # Stratified split + manifest I/O
    # ──────────────────────────────────────────────────────────────
    def _manifest_filename(self) -> str:
        """Deterministic filename encoding split params so different configs don't collide."""
        tag = f"val{self.val_split}_test{self.test_split}_seed{self.seed}"
        short_hash = hashlib.md5(tag.encode()).hexdigest()[:8]
        return f"split_manifest_{short_hash}.json"

    def _save_manifest(
        self,
        full_dataset: ImageFolder,
        train_idx: List[int],
        val_idx: List[int],
        test_idx: List[int],
    ) -> Path:
        """Persist split indices + metadata to a JSON file."""
        self.split_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.split_dir / self._manifest_filename()

        # Build per-sample records (path relative to Images/, label)
        samples = full_dataset.samples  # list of (abs_path, class_idx)
        images_root = os.path.join(self.data_dir, "Images")

        def _rel(abs_path: str) -> str:
            return os.path.relpath(abs_path, images_root)

        manifest: Dict[str, Any] = {
            "meta": {
                "created_at": datetime.now().isoformat(),
                "data_dir": str(self.data_dir),
                "val_split": self.val_split,
                "test_split": self.test_split,
                "seed": self.seed,
                "total_samples": len(full_dataset),
                "num_classes": len(full_dataset.classes),
                "class_names": full_dataset.classes,
            },
            "train": [
                {"index": int(i), "path": _rel(samples[i][0]), "label": int(samples[i][1])}
                for i in train_idx
            ],
            "val": [
                {"index": int(i), "path": _rel(samples[i][0]), "label": int(samples[i][1])}
                for i in val_idx
            ],
            "test": [
                {"index": int(i), "path": _rel(samples[i][0]), "label": int(samples[i][1])}
                for i in test_idx
            ],
        }

        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)

        logger.info("Split manifest saved → %s", manifest_path)
        return manifest_path

    def _load_manifest(self) -> Optional[Dict[str, Any]]:
        """Try to load an existing manifest matching current split params."""
        manifest_path = self.split_dir / self._manifest_filename()
        if not manifest_path.is_file():
            return None

        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)

        logger.info("Reusing existing split manifest ← %s", manifest_path)
        self.split_manifest_path = manifest_path
        return manifest

    def _stratified_split(self, labels: np.ndarray) -> tuple:
        """Perform a two-stage stratified split: (train+val) / test, then train / val.

        Returns
        -------
        train_indices, val_indices, test_indices : np.ndarray
        """
        all_indices = np.arange(len(labels))

        # Stage 1: separate test set
        if self.test_split > 0:
            sss_test = StratifiedShuffleSplit(
                n_splits=1, test_size=self.test_split, random_state=self.seed
            )
            trainval_idx, test_idx = next(sss_test.split(all_indices, labels))
        else:
            trainval_idx = all_indices
            test_idx = np.array([], dtype=int)

        # Stage 2: separate val from train
        trainval_labels = labels[trainval_idx]
        relative_val = self.val_split / (1.0 - self.test_split)  # adjust ratio
        if self.val_split > 0:
            sss_val = StratifiedShuffleSplit(
                n_splits=1, test_size=relative_val, random_state=self.seed
            )
            train_local, val_local = next(sss_val.split(trainval_idx, trainval_labels))
            train_idx = trainval_idx[train_local]
            val_idx = trainval_idx[val_local]
        else:
            train_idx = trainval_idx
            val_idx = np.array([], dtype=int)

        return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()

    # ──────────────────────────────────────────────────────────────
    # setup()
    # ──────────────────────────────────────────────────────────────
    def setup(self, stage: Optional[str] = None) -> None:
        """Load the dataset and create (or reload) the stratified split.

        The method is idempotent — calling it multiple times has no effect
        after the first successful call.
        """
        if self.data_train is not None:
            return

        full_dataset = ImageFolder(root=os.path.join(self.data_dir, "Images"))
        self.classes = full_dataset.classes
        self.class_to_idx = full_dataset.class_to_idx
        self.num_classes = len(self.classes)

        labels = np.array([label for _, label in full_dataset.samples])

        # Try to reload an existing manifest
        manifest = self._load_manifest()
        if manifest is not None:
            self.train_indices = [s["index"] for s in manifest["train"]]
            self.val_indices = [s["index"] for s in manifest["val"]]
            self.test_indices = [s["index"] for s in manifest["test"]]
        else:
            self.train_indices, self.val_indices, self.test_indices = self._stratified_split(labels)
            self.split_manifest_path = self._save_manifest(
                full_dataset, self.train_indices, self.val_indices, self.test_indices
            )

        logger.info(
            "Dataset split: train=%d, val=%d, test=%d (total=%d)",
            len(self.train_indices), len(self.val_indices),
            len(self.test_indices), len(full_dataset),
        )

        self.data_train = _SubsetWithTransform(
            Subset(full_dataset, self.train_indices), transform=self.train_transforms
        )
        self.data_val = _SubsetWithTransform(
            Subset(full_dataset, self.val_indices), transform=self.val_transforms
        )
        self.data_test = _SubsetWithTransform(
            Subset(full_dataset, self.test_indices), transform=self.val_transforms
        )

    # ──────────────────────────────────────────────────────────────
    # DataLoaders (deterministic seeding)
    # ──────────────────────────────────────────────────────────────
    def _make_generator(self) -> torch.Generator:
        """Create a seeded Generator for reproducible shuffling."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        return g

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            worker_init_fn=_seed_worker,
            generator=self._make_generator(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            worker_init_fn=_seed_worker,
            generator=self._make_generator(),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            worker_init_fn=_seed_worker,
            generator=self._make_generator(),
        )

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────
    def get_split_labels(self, split: str = "train") -> np.ndarray:
        """Return the label array for a given split.

        Parameters
        ----------
        split:
            One of ``"train"``, ``"val"``, ``"test"``.

        Returns
        -------
        np.ndarray of int
            Class labels for every sample in the split.
        """
        indices_map = {"train": self.train_indices, "val": self.val_indices, "test": self.test_indices}
        if split not in indices_map:
            raise ValueError(f"Unknown split '{split}'. Use 'train', 'val', or 'test'.")

        full_dataset = self.data_train.subset.dataset  # the original ImageFolder
        labels = np.array([full_dataset.samples[i][1] for i in indices_map[split]])
        return labels
