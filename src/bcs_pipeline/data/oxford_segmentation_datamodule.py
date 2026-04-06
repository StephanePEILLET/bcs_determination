import os
import logging
from typing import Optional

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

logger = logging.getLogger("bcs_pipeline")


class OxfordSegmentationDataset(Dataset):
    """
    Dataset wrapper for Oxford-IIIT Pet Segmentation (Trimaps).
    """
    def __init__(self, data_dir: str, mode: str = "trainval", transform=None):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        
        self.images_dir = os.path.join(data_dir, "images")
        self.masks_dir = os.path.join(data_dir, "annotations", "trimaps")
        
        split_file = os.path.join(data_dir, "annotations", f"{mode}.txt")
        self.samples = []
        self.labels = [] # We keep track of labels for stratified split!
        
        if not os.path.isfile(split_file):
            split_file = os.path.join(data_dir, "annotations", "list.txt")

        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line: continue
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
        
        # Load image via cv2 for Albumentations
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load mask via cv2 (grayscale)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Oxford trimap pixels are 1=foreground, 2=background, 3=border. Convert to 0, 1, 2
        mask = mask.astype(np.int64) - 1
        
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        
        return img, mask


class OxfordSegmentationDataModule(pl.LightningDataModule):
    """LightningDataModule for Oxford-IIIT Pet Segmentation."""
    
    def __init__(
        self,
        data_dir: str = "data/Oxford-IIIT_pet_dataset",
        batch_size: int = 16, # Segmentation batch size is usually smaller than classification
        num_workers: int = 4,
        image_size: int = 256,
        val_split: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Albumentations pipelines
        self.train_transforms = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        self.val_transforms = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        self.data_train = None
        self.data_val = None
        self.data_test = None

    def setup(self, stage: Optional[str] = None):
        if self.data_train is not None:
            return
            
        full_trainval_ds = OxfordSegmentationDataset(self.hparams.data_dir, mode="trainval")
        self.data_test = OxfordSegmentationDataset(
            self.hparams.data_dir, mode="test", transform=self.val_transforms
        )
        
        labels = np.array(full_trainval_ds.labels)
        all_indices = np.arange(len(full_trainval_ds))
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.hparams.val_split, random_state=self.hparams.seed)
        train_idx, val_idx = next(sss.split(all_indices, labels))
        
        # Wrapper class to apply transforms to subsets dynamically
        class AlbumentationSubset(Dataset):
            def __init__(self, dataset, indices, transform):
                self.dataset = dataset
                self.indices = indices
                self.transform = transform
            def __len__(self): return len(self.indices)
            def __getitem__(self, i):
                real_idx = self.indices[i]
                img_path, mask_path = self.dataset.samples[real_idx]
                
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                # Oxford Trimap to 0,1,2
                mask = mask.astype(np.int64) - 1
                
                if self.transform:
                    aug = self.transform(image=img, mask=mask)
                    return aug['image'], aug['mask']
                return img, mask

        self.data_train = AlbumentationSubset(full_trainval_ds, train_idx, self.train_transforms)
        self.data_val = AlbumentationSubset(full_trainval_ds, val_idx, self.val_transforms)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False)
