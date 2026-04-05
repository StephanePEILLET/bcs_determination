"""
bcs_pipeline – BCS Determination deep-learning pipeline.

Subpackages
-----------
callbacks
    Training callback factories (checkpoint, early stopping, LR monitor).
loggers
    PyTorch Lightning logger factories (TensorBoard, W&B).
trainer_factory
    High-level ``pl.Trainer`` builder.
inference
    Model loading, transforms, and prediction helpers.
data
    ``LightningDataModule`` for the Stanford Dogs dataset.
lightning_module
    ``LightningModule`` implementing training / validation / test logic.
models
    Backbone architectures (ResNet-50 transfer, ViT transfer).
utils
    Configuration loading / validation and logging utilities.
"""
