import logging
from typing import Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger("bcs_pipeline")


class DummySegmentationDataset(Dataset):
    """
    Placeholder dataset for Stanford Dogs Segmentation.
    In the future, this will load image paths and mask paths, 
    and apply segmentation-specific transforms (e.g. from albumentations).
    """
    def __init__(self, size: int = 100, image_size: int = 224):
        self.size = size
        self.image_size = image_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Returns a dummy image (C, H, W) and dummy mask (1, H, W) or (H, W)
        img = torch.randn(3, self.image_size, self.image_size)
        mask = torch.randint(0, 2, (self.image_size, self.image_size), dtype=torch.long)
        return img, mask


class StanfordSegmentationDataModule(pl.LightningDataModule):
    """
    Skeleton DataModule for Segmentation tasks on Stanford Dogs or similar datasets.
    """
    def __init__(
        self,
        data_dir: str = "data/stanford_dogs",
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        # TODO: Implement download logic for masks or alternative dataset format
        pass

    def setup(self, stage: Optional[str] = None):
        if self.data_train is not None:
            return

        # TODO: Implement train/val/test splits with actual mask image paths
        # Using dummy dataset for the skeleton
        logger.warning("StanfordSegmentationDataModule is using a Dummy Dataset.")
        self.data_train = DummySegmentationDataset(size=800, image_size=self.hparams.image_size)
        self.data_val = DummySegmentationDataset(size=100, image_size=self.hparams.image_size)
        self.data_test = DummySegmentationDataset(size=100, image_size=self.hparams.image_size)

    def train_dataloader(self):
        return DataLoader(
            self.data_train, batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val, batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test, batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, shuffle=False
        )
