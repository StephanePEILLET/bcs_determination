"""BCS Classification Lightning Module.

Reframes Body Condition Score prediction as an **ordinal classification** over the
discrete BCS scores actually present in the training data (e.g. ``[4, 5, 6]`` for the
OGR cats), rather than continuous regression. It shares the same frozen ViT breed
backbone as :class:`bcs_pipeline.lightning_module.bcs_regression_module.LitBCSRegression`
and trains a lightweight MLP classification head.

Why classification: the OGR dataset is tiny and narrow (11 cats, only 3 distinct
scores), on which the regression head collapsed to predicting the dataset mean and
scored *worse* than a constant baseline. Classification yields honest, interpretable
outputs (per-class probabilities + confidence) and a guaranteed floor (the majority
class). At inference the ordinal nature is recovered via the probability-weighted
**expected score** ``Σ pᵢ·scoreᵢ``, which keeps the downstream ``bcs`` float schema.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, MeanAbsoluteError

from bcs_pipeline.lightning_module.bcs_regression_module import _BCSBackboneMixin

# ── Canonical BCS covariate (metadata) encoding ──────────────────────────────
# Ordered names of the encoded metadata dimensions. Sex and coat length are each
# a 2-dim one-hot whose all-zeros state means "unknown" — so a missing value at
# inference is representable and matches the metadata-dropout used at training.
BCS_COVARIATE_NAMES: List[str] = ["sex_M", "sex_F", "coat_short", "coat_long"]


def encode_bcs_covariates(
    sex: Optional[str] = None,
    long_coat: Optional[object] = None,
) -> Dict[str, float]:
    """Encode per-animal metadata into the canonical {name: 0/1} covariate map.

    ``sex`` is ``"M"``/``"F"`` (case-insensitive) or ``None`` (unknown).
    ``long_coat`` is truthy/falsy (1/0, True/False, "Y"/"N") or ``None`` (unknown).
    Unknown fields encode as all-zeros for their one-hot block.
    """
    d = {n: 0.0 for n in BCS_COVARIATE_NAMES}
    if sex is not None:
        s = str(sex).strip().upper()
        if s in ("M", "MALE"):
            d["sex_M"] = 1.0
        elif s in ("F", "FEMALE"):
            d["sex_F"] = 1.0
    if long_coat is not None:
        if isinstance(long_coat, str):
            lc = long_coat.strip().upper() in ("1", "Y", "YES", "TRUE", "LONG")
        else:
            lc = bool(long_coat)
        d["coat_long" if lc else "coat_short"] = 1.0
    return d


def covariates_to_vector(names: Sequence[str], encoded: Dict[str, float]) -> List[float]:
    """Order an encoded covariate map into a vector following *names*."""
    return [float(encoded.get(n, 0.0)) for n in names]


class BCSClassificationHead(nn.Module):
    """MLP head: (LayerNorm(ViT feats) ⊕ covariates) → hidden → num_bcs_classes.

    The optional covariate vector (encoded per-animal metadata such as sex / coat
    length) is concatenated to the layer-normalised ViT features before the MLP,
    so the head can condition its BCS prediction on that metadata. ``covariate_dim
    == 0`` recovers a plain image-only head.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        num_bcs_classes: int = 3,
        covariate_dim: int = 0,
        class_log_prior: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.covariate_dim = covariate_dim
        # Normalise only the ViT features; covariates are already in a small,
        # bounded 0/1 encoding and are concatenated raw.
        self.norm = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim + covariate_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_bcs_classes),
        )
        # Warm-start the output layer: weights = 0, bias = log class prior. The
        # head therefore starts by predicting the class base rates and only needs
        # to learn the image-dependent residual (mirrors the regression module's
        # mean warm-start, but for classification).
        final_linear = self.mlp[-1]
        nn.init.zeros_(final_linear.weight)
        if class_log_prior is not None:
            with torch.no_grad():
                final_linear.bias.copy_(torch.tensor(class_log_prior, dtype=final_linear.bias.dtype))
        else:
            nn.init.zeros_(final_linear.bias)

    def forward(self, feats: torch.Tensor, cov: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm(feats)
        if self.covariate_dim > 0:
            if cov is None:
                cov = feats.new_zeros(feats.shape[0], self.covariate_dim)
            h = torch.cat([h, cov], dim=-1)
        return self.mlp(h)  # (B, num_bcs_classes) logits


class LitBCSClassification(_BCSBackboneMixin, LightningModule):
    """Frozen ViT backbone + trainable classification head for BCS prediction."""

    def __init__(
        self,
        bcs_classes: Sequence[float],
        backbone_ckpt: Optional[str] = None,
        model_name: str = "vit",
        num_classes: int = 132,
        embedding_dim: int = 768,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: Optional[Sequence[float]] = None,
        class_log_prior: Optional[Sequence[float]] = None,
        covariate_names: Optional[Sequence[str]] = None,
        covariate_dropout: float = 0.3,
    ):
        super().__init__()
        # ``bcs_classes`` (the ordered BCS scores this head predicts), the
        # covariate spec and the optional class weights/priors are saved as
        # hyper-parameters so inference can reconstruct the score mapping and the
        # metadata layout from the checkpoint alone.
        self.save_hyperparameters()

        self.bcs_classes: List[float] = [float(c) for c in bcs_classes]
        n_cls = len(self.bcs_classes)
        # Registered buffer: the score value for each class index, used to turn
        # probabilities into an expected BCS score.
        self.register_buffer(
            "class_scores", torch.tensor(self.bcs_classes, dtype=torch.float32)
        )

        # Covariate (metadata) conditioning. ``covariate_names`` are the ordered
        # names of the encoded metadata dims (e.g. ["sex_M","sex_F","coat_short",
        # "coat_long"]); an empty list = image-only head.
        self.covariate_names: List[str] = list(covariate_names or [])
        self.covariate_dim = len(self.covariate_names)
        self.covariate_dropout = float(covariate_dropout)

        # Build and freeze ViT backbone (shared logic with regression module).
        self.backbone = self._build_backbone(backbone_ckpt, model_name, num_classes)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        # Trainable classification head
        self.head = BCSClassificationHead(
            embedding_dim,
            hidden_dim,
            dropout,
            num_bcs_classes=n_cls,
            covariate_dim=self.covariate_dim,
            class_log_prior=class_log_prior,
        )

        # Loss (optionally class-weighted to counter imbalance) and metrics.
        weight = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None
            else None
        )
        self.loss_fn = nn.CrossEntropyLoss(weight=weight)
        self.train_acc = Accuracy(task="multiclass", num_classes=n_cls)
        self.val_acc = Accuracy(task="multiclass", num_classes=n_cls)
        # Expected-score MAE lets us compare against the old regression baseline.
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()

    # ── Score ↔ class-index helpers ──────────────────────────────────────
    def _scores_to_indices(self, y: torch.Tensor) -> torch.Tensor:
        """Map float BCS scores to class indices via nearest ``class_scores``."""
        # y: (B,) float; class_scores: (C,). Nearest-match is exact here since
        # training labels are drawn from ``bcs_classes``.
        diffs = (y.view(-1, 1) - self.class_scores.view(1, -1)).abs()
        return diffs.argmin(dim=1)

    def expected_score(self, logits: torch.Tensor) -> torch.Tensor:
        """Probability-weighted expected BCS score ``Σ pᵢ·scoreᵢ`` per sample."""
        probs = F.softmax(logits, dim=-1)
        return (probs * self.class_scores.view(1, -1)).sum(dim=-1)

    def _prep_cov(self, cov: Optional[torch.Tensor], batch_size: int, device) -> Optional[torch.Tensor]:
        """Return a (B, covariate_dim) tensor, or None when no covariates."""
        if self.covariate_dim == 0:
            return None
        if cov is None or cov.numel() == 0:
            return torch.zeros(batch_size, self.covariate_dim, device=device)
        return cov.to(device).float()

    # ── Forward / steps ──────────────────────────────────────────────────
    def logits(self, x: torch.Tensor, cov: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = self.extract_features(x).float()
        cov = self._prep_cov(cov, features.shape[0], features.device)
        return self.head(features, cov)

    def forward(self, x: torch.Tensor, cov: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.logits(x, cov)

    def predict_proba(self, x: torch.Tensor, cov: Optional[torch.Tensor] = None) -> torch.Tensor:
        return F.softmax(self.logits(x, cov), dim=-1)

    def _shared_step(self, batch, training: bool = False):
        x, cov, y = batch  # cov: (B, K) or (B, 0); y: float BCS scores
        cov = self._prep_cov(cov, x.shape[0], x.device)
        if training and cov is not None and self.covariate_dropout > 0:
            # Metadata dropout: randomly blank the whole covariate vector per
            # sample so the head stays usable when metadata is missing at
            # inference (sex/coat are user-supplied and may be unknown).
            keep = (torch.rand(x.shape[0], 1, device=cov.device) >= self.covariate_dropout).float()
            cov = cov * keep
        logits = self(x, cov)
        target = self._scores_to_indices(y.to(logits.device))
        loss = self.loss_fn(logits, target)
        exp_score = self.expected_score(logits)
        return loss, logits, target, exp_score, y

    def training_step(self, batch, batch_idx):
        loss, logits, target, exp_score, y = self._shared_step(batch, training=True)
        self.train_acc(logits, target)
        self.train_mae(exp_score, y.to(exp_score.device))
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", self.train_acc, prog_bar=True)
        self.log("train/mae", self.train_mae, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, target, exp_score, y = self._shared_step(batch, training=False)
        self.val_acc(logits, target)
        self.val_mae(exp_score, y.to(exp_score.device))
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True)
        self.log("val/mae", self.val_mae)
        return loss

    def predict_step(self, batch, batch_idx):
        x, cov, _ = batch
        return self.predict_proba(x, cov)

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
        # Keep the backbone frozen/eval (no dropout drift).
        self.backbone.eval()
