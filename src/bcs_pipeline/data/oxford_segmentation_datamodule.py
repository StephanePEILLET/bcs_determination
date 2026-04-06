"""
LightningDataModule for Oxford-IIIT Pet Dataset – Segmentation task.

Uses torchvision transforms instead of albumentations to avoid an extra
dependency.  The Oxford trimap annotations contain 3 classes:
    0 = pet (foreground), 1 = background, 2 = border.

The raw trimap pixel values are {1, 2, 3} and are remapped to {0, 1, 2}.
"""

import os
import logging
from typing import Optional, List, Tuple

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger("bcs_pipeline")


class OxfordSegmentationDataset(Dataset):
    """Dataset for Oxford-IIIT Pet segmentation (images + trimap masks)."""

    def __init__(
        self,
        data_dir: str,
        mode: str = "trainval",
        transform=None,
        mask_transform=None,
    ):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        self.mask_transform = mask_transform

        self.images_dir = os.path.join(data_dir, "images")
        self.masks_dir = os.path.join(data_dir, "annotations", "trimaps")

        split_file = os.path.join(data_dir, "annotations", f"{mode}.txt")
        self.samples: List[Tuple[str, str]] = []
        self.labels: List[int] = []

        if not os.path.isfile(split_file):
            split_file = os.path.join(data_dir, "annotations", "list.txt")

        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                parts = line.split(" ")
                image_name = parts[0]
                class_id = int(parts[1]) - 1

                image_path = os.path.join(self.images_dir, f"{image_name}.jpg")
                mask_path = os.path.join(self.masks_dir, f"{image_name}.png")

                if os.path.exists(image_path) and os.path.exists(mask_path):
                    self.samples.append((image_path, mask_path))
                    self.labels.append(class_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        mask = np.array(mask, dtype=np.int64)
        mask = mask - 1  # remap {1,2,3} → {0,1,2}
        mask = Image.fromarray(mask.astype(np.int8), mode="L")

        if self.transform:
            img = self.transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = TF.pil_to_tensor(mask).squeeze(0).long()

        return img, mask


class _SegSubset(Dataset):
    """Subset that applies separate image / mask transforms."""

    def __init__(self, dataset: Dataset, indices: List[int], transform, mask_transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        real_idx = self.indices[i]
        img_path, mask_path = self.dataset.samples[real_idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask_np = np.array(mask, dtype=np.int64) - 1
        mask = Image.fromarray(mask_np.astype(np.int8), mode="L")

        # Synchronized random transforms
        if self.transform and self.mask_transform:
            seed = torch.randint(0, 2**31, (1,)).item()
            torch.manual_seed(seed)
            img = self.transform(img)
            torch.manual_seed(seed)
            mask = self.mask_transform(mask)
        else:
            if self.transform:
                img = self.transform(img)
            mask = TF.pil_to_tensor(mask).squeeze(0).long()

        return img, mask


class OxfordSegmentationDataModule(pl.LightningDataModule):
    """LightningDataModule for Oxford-IIIT Pet Segmentation.

    Parameters
    ----------
    data_dir:
        Path to the Oxford dataset root (containing ``images/`` and
        ``annotations/trimaps/``).
    batch_size:
        Mini-batch size.
    num_workers:
        DataLoader workers.
    image_size:
        Spatial resolution fed to the model.
    val_split:
        Fraction of *trainval* reserved for validation.
    seed:
        Random seed for the stratified split.
    """

    NUM_CLASSES = 3  # foreground / background / border

    def __init__(
        self,
        data_dir: str = "data/Oxford-IIIT_pet_dataset",
        batch_size: int = 16,
        num_workers: int = 4,
        image_size: int = 256,
        val_split: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.train_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=15, translate=(0.0625, 0.0625), scale=(0.9, 1.1)
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.train_mask_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=TF.InterpolationMode.NEAREST),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=15, translate=(0.0625, 0.0625), scale=(0.9, 1.1),
                interpolation=TF.InterpolationMode.NEAREST,
            ),
            transforms.PILToTensor(),
        ])

        self.val_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.val_mask_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=TF.InterpolationMode.NEAREST),
            transforms.PILToTensor(),
        ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if self.data_train is not None:
            return

        hp = self.hparams
        full_trainval = OxfordSegmentationDataset(hp.data_dir, mode="trainval")
        self.data_test = OxfordSegmentationDataset(
            hp.data_dir, mode="test",
            transform=self.val_transforms,
            mask_transform=self.val_mask_transforms,
        )

        labels = np.array(full_trainval.labels)
        all_indices = np.arange(len(full_trainval))

        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=hp.val_split, random_state=hp.seed,
        )
        train_idx, val_idx = next(sss.split(all_indices, labels))

        logger.info(
            "Oxford Seg split: train=%d, val=%d, test=%d",
            len(train_idx), len(val_idx), len(self.data_test),
        )

        self.data_train = _SegSubset(
            full_trainval, train_idx, self.train_transforms, self.train_mask_transforms,
        )
        self.data_val = _SegSubset(
            full_trainval, val_idx, self.val_transforms, self.val_mask_transforms,
        )

    def train_dataloader(self):
        return DataLoader(
            self.data_train, batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers, shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val, batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers, shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test, batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers, shuffle=False,
        )
