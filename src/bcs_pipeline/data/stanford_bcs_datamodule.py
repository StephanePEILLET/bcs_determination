import os
import tarfile
import logging
from typing import Optional, Tuple
from urllib.request import urlretrieve

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

logger = logging.getLogger("bcs_pipeline")

DATASET_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"

class StanfordBcsDataModule(LightningDataModule):
    """
    LightningDataModule for Stanford Bcs Dataset.
    Expects data in `data_dir/Images/` with 120 class subdirectories.
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
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

        # ImageNet normalization statistics
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

        self.val_transforms = transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.14)), # Typical ratio for 224 is 256
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

        self.data_train: Optional[torch.utils.data.Subset] = None
        self.data_val: Optional[torch.utils.data.Subset] = None
        self.data_test: Optional[torch.utils.data.Subset] = None
        
        self.num_classes = 120

    def prepare_data(self):
        """Download and extract dataset if not already present."""
        images_dir = os.path.join(self.data_dir, "Images")
        if os.path.exists(images_dir):
            return

        os.makedirs(self.data_dir, exist_ok=True)
        tar_path = os.path.join(self.data_dir, "images.tar")

        if not os.path.exists(tar_path):
            logger.info(f"Downloading Stanford Dogs dataset to {tar_path}...")
            urlretrieve(DATASET_URL, tar_path)
            logger.info("Download complete.")

        logger.info(f"Extracting {tar_path}...")
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=self.data_dir)
        logger.info(f"Dataset ready at {images_dir}")

        # Cleanup tar file
        if os.path.exists(tar_path):
            os.remove(tar_path)

    def setup(self, stage: Optional[str] = None):
        """Load data and apply splits."""
        if not self.data_train and not self.data_val and not self.data_test:
            # Load full dataset
            full_dataset = ImageFolder(root=os.path.join(self.data_dir, "Images"))
            
            # Save classes
            self.classes = full_dataset.classes
            self.class_to_idx = full_dataset.class_to_idx

            dataset_size = len(full_dataset)
            val_size = int(self.val_split * dataset_size)
            test_size = int(self.test_split * dataset_size)
            train_size = dataset_size - val_size - test_size

            # Split dataset
            train_dataset, val_dataset, test_dataset = random_split(
                dataset=full_dataset,
                lengths=[train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

            # Define dataset wrapper to apply different transforms to subsets
            class SubsetWithTransform(torch.utils.data.Dataset):
                def __init__(self, subset, transform=None):
                    self.subset = subset
                    self.transform = transform
                    
                def __getitem__(self, index):
                    x, y = self.subset[index]
                    if self.transform:
                        x = self.transform(x)
                    return x, y
                    
                def __len__(self):
                    return len(self.subset)

            self.data_train = SubsetWithTransform(train_dataset, transform=self.train_transforms)
            self.data_val = SubsetWithTransform(val_dataset, transform=self.val_transforms)
            self.data_test = SubsetWithTransform(test_dataset, transform=self.val_transforms)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )
