"""BCS Regression Lightning Module.

Uses a frozen ViT backbone (pre-trained on breed classification) as feature
extractor, and trains a lightweight MLP head to regress the Body Condition
Score (continuous, 1–9 scale).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import MeanAbsoluteError, MeanSquaredError


class BCSRegressionHead(nn.Module):
    """Small MLP: embedding_dim → hidden → 1 (BCS score)."""

    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        target_mean: float = 5.0,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # Warm-start the output layer: bias = target mean, weights = 0.
        # The head therefore starts by predicting the dataset mean and only
        # needs to learn residuals (avoids the LR collapsing before the bias
        # has time to drift up from 0 to the BCS range).
        final_linear = self.head[-1]
        nn.init.zeros_(final_linear.weight)
        nn.init.constant_(final_linear.bias, float(target_mean))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)  # (B,)


class LitBCSRegression(LightningModule):
    """Frozen ViT backbone + trainable regression head for BCS prediction."""

    def __init__(
        self,
        backbone_ckpt: Optional[str] = None,
        model_name: str = "vit",
        num_classes: int = 132,
        embedding_dim: int = 768,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        target_mean: float = 5.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Build and freeze ViT backbone
        self.backbone = self._build_backbone(backbone_ckpt, model_name, num_classes)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        # Trainable regression head
        self.head = BCSRegressionHead(
            embedding_dim, hidden_dim, dropout, target_mean=target_mean
        )

        # Loss and metrics
        self.loss_fn = nn.MSELoss()
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.val_mse = MeanSquaredError()

    def _build_backbone(self, ckpt_path, model_name, num_classes):
        """Load classification checkpoint and extract ViT backbone."""
        from bcs_pipeline.lightning_module.classification_module import LitClassificationModule

        lit_model = LitClassificationModule(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=False,
        )
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
            missing, unexpected = lit_model.load_state_dict(state_dict, strict=False)
            # Surface mismatches so a silently-random ViT can't go unnoticed.
            backbone_missing = [k for k in missing if k.startswith("net.")]
            backbone_unexpected = [k for k in unexpected if k.startswith("net.")]
            if backbone_missing or backbone_unexpected:
                print(
                    f"[backbone] missing={len(backbone_missing)} "
                    f"unexpected={len(backbone_unexpected)} keys on backbone load"
                )
                if backbone_missing[:3]:
                    print(f"  e.g. missing: {backbone_missing[:3]}")
                if backbone_unexpected[:3]:
                    print(f"  e.g. unexpected: {backbone_unexpected[:3]}")

        # Return just the ViT wrapper (ViTTransfer)
        return lit_model.net

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CLS token embedding from frozen ViT backbone."""
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=False), torch.amp.autocast(device_type="cpu", enabled=False):
            # Ensure float32 and same device as backbone
            x_f32 = x.float().to(next(self.backbone.parameters()).device)
            outputs = self.backbone.vit(x_f32, output_hidden_states=True)
            # CLS token = first token of last hidden state
            cls_embedding = outputs.hidden_states[-1][:, 0]  # (B, 768)
        return cls_embedding.to(x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        return self.head(features.float())

    def training_step(self, batch, batch_idx):
        x, y = batch  # y: float BCS scores
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.train_mae(y_hat, y)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/mae", self.train_mae, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.val_mae(y_hat, y)
        self.val_mse(y_hat, y)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/mae", self.val_mae, prog_bar=True)
        self.log("val/mse", self.val_mse)
        return loss

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.head.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"},
        }

    def on_train_epoch_start(self):
        # Ensure backbone stays in eval mode (no dropout etc.)
        self.backbone.eval()
