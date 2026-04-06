import os
import logging
from typing import Optional, Tuple, Dict, Any

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger("bcs_pipeline")


class OxfordPetDataset(Dataset):
    """
    Dataset wrapper for Oxford-IIIT Pet Dataset parsing trainval.txt / test.txt files.
    """

    def __init__(self, data_dir: str, mode: str = "trainval", transform=None):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        self.images_dir = os.path.join(data_dir, "images")
        
        # Determine split file to read
        split_file = os.path.join(data_dir, "annotations", f"{mode}.txt")
        self.samples = []
        self.labels = []
        
        if not os.path.isfile(split_file):
            logger.warning(f"Could not find {split_file}. Attempting to read list.txt")
            split_file = os.path.join(data_dir, "annotations", "list.txt")

        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                # Format: Image CLASS-ID SPECIES BREED ID
                parts = line.split(" ")
                image_name = parts[0]
                # Oxford class IDs are 1-indexed (1 to 37), convert to 0-indexed (0 to 36)
                class_id = int(parts[1]) - 1
                
                # Check for image extension
                image_path = os.path.join(self.images_dir, f"{image_name}.jpg")
                if not os.path.exists(image_path):
                    # Some files might have different extensions, check manually if needed
                    logger.debug(f"Missing file: {image_path}")
                    continue
                
                self.samples.append(image_path)
                self.labels.append(class_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load image (RGB)
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
            
        return img, label


class OxfordClassificationDataModule(pl.LightningDataModule):
    """
    LightningDataModule for Oxford-IIIT Pet Dataset Classification.
    """

    def __init__(
        self,
        data_dir: str = "data/Oxford-IIIT_pet_dataset",
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        val_split: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed
        
        # ImageNet normalization statistics
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize,
        ])

        self.val_transforms = transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            self.normalize,
        ])
        
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.num_classes = 37

    def prepare_data(self):
        pass # Assuming data is already uncompressed by the user

    def setup(self, stage: Optional[str] = None):
        if self.data_train is not None:
            return
            
        # Parse datasets from txt files
        full_trainval_ds = OxfordPetDataset(self.data_dir, mode="trainval")
        self.data_test = OxfordPetDataset(self.data_dir, mode="test", transform=self.val_transforms)
        
        # Stratified Split for train/val from trainval
        labels = np.array(full_trainval_ds.labels)
        all_indices = np.arange(len(full_trainval_ds))
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.val_split, random_state=self.seed)
        train_idx, val_idx = next(sss.split(all_indices, labels))
        
        logger.info(f"Oxford Split: train={len(train_idx)}, val={len(val_idx)}, test={len(self.data_test)}")

        # Create mapped subsets mapping transform
        class WrappedSubset(Dataset):
            def __init__(self, subset, transform=None):
                self.subset = subset
                self.transform = transform
            def __len__(self): return len(self.subset)
            def __getitem__(self, i):
                img_path = self.subset.dataset.samples[self.subset.indices[i]]
                label = self.subset.dataset.labels[self.subset.indices[i]]
                img = Image.open(img_path).convert("RGB")
                if self.transform: img = self.transform(img)
                return img, label

        self.data_train = WrappedSubset(torch.utils.data.Subset(full_trainval_ds, train_idx), self.train_transforms)
        self.data_val = WrappedSubset(torch.utils.data.Subset(full_trainval_ds, val_idx), self.val_transforms)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
