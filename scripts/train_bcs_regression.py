#!/usr/bin/env python
"""Train the BCS Regression model using Leave-One-Cat-Out cross-validation.

Runs the full LOCO-CV pipeline (SAM3 segmentation → ViT feature extraction →
MLP regression head) and saves:
  - Per-fold best checkpoints in ``checkpoints/bcs_regression/fold_<cat_id>/``
  - A global ``predictions.json`` with all fold predictions + metrics

Usage
-----
.. code-block:: bash

    # Default: 80 epochs, batch_size=4
    python scripts/train_bcs_regression.py

    # Override parameters
    python scripts/train_bcs_regression.py --max-epochs 100 --batch-size 8

    # Quick test run
    python scripts/train_bcs_regression.py --max-epochs 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from pillow_heif import register_heif_opener
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from tqdm.auto import tqdm

# ── Make the ``src/`` package importable ─────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from bcs_pipeline.app_checkpoints import SAM3_BPE, SAM3_CKPT, resolve_classifier_ckpt
from bcs_pipeline.data.bcs_datamodule import BCSDataModule
from bcs_pipeline.inference import load_segmentation_backend, predict_segmentation_with
from bcs_pipeline.lightning_module.bcs_classification_module import (
    BCS_COVARIATE_NAMES,
    LitBCSClassification,
    covariates_to_vector,
    encode_bcs_covariates,
)
from bcs_pipeline.lightning_module.bcs_regression_module import LitBCSRegression

register_heif_opener()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BCS regression (LOCO-CV)")
    parser.add_argument("--max-epochs", type=int, default=80, help="Max training epochs per fold")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=128, help="MLP hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate in head")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (0=disabled)")
    parser.add_argument("--task", type=str, default="classification",
                        choices=["classification", "regression"],
                        help="Prediction head: 'classification' (ordinal over the discrete "
                             "BCS scores present, default) or legacy 'regression'.")
    parser.add_argument("--covariates", action=argparse.BooleanOptionalAction, default=True,
                        help="Condition the classification head on per-animal metadata "
                             "(sex + long-coat). Use --no-covariates to disable. "
                             "Ignored for --task regression.")
    parser.add_argument("--species", type=str, default="cat", choices=["cat", "dog"],
                        help="Which species BCS model to train. 'dog' is a placeholder "
                             "until a dog BCS dataset is provided.")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Dataset directory. Default: data/Cats_OGR_dataset for cat.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: checkpoints/bcs_regression/<species>/)")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader num_workers")
    parser.add_argument("--sam3-mode", type=str, default="prompted",
                        choices=["prompted", "concept", "automatic"],
                        help="SAM3 segmentation mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_dataset(data_dir: Path) -> pd.DataFrame:
    """Load the Cats OGR dataset and resolve image paths."""
    excel_path = data_dir / "BCS Excel Pra Ana Eiras - Master Thesis LC 2.xlsx"
    df_data = pd.read_excel(excel_path, sheet_name="Data", header=2)
    df_data = df_data.loc[:, ~df_data.columns.astype(str).str.startswith("Unnamed")]
    df_data = df_data.rename(columns={"BCS_Score_1_9_AE": "BCS"})

    images_dir = data_dir / "images"
    rows = []
    for _, r in df_data.iterrows():
        for view, col in [("LV", "Photo_LV_ID"), ("DV", "Photo_DV_ID")]:
            photo_id = r.get(col)
            if pd.isna(photo_id):
                continue
            candidates = list(images_dir.glob(f"{photo_id}.*"))
            if not candidates:
                continue
            rows.append({
                "Animal_ID": r["Animal_ID"],
                "view": view,
                "path": str(candidates[0]),
                "BCS": float(r["BCS"]),
                # Per-animal metadata (covariates). Kept raw here; encoded into
                # the numeric ``cov`` vector later when covariate conditioning is on.
                "Sex": (None if pd.isna(r.get("Sex")) else str(r.get("Sex"))),
                "Long_Coat_YN": (None if pd.isna(r.get("Long_Coat_YN")) else r.get("Long_Coat_YN")),
            })
    return pd.DataFrame(rows)


def compute_masks(records: list, device: torch.device, sam3_mode: str) -> list:
    """Pre-compute SAM3 segmentation masks for all images."""
    print("\n── Chargement SAM3 et pré-calcul des masques ──")
    sam3_handle = load_segmentation_backend(
        "sam3", str(SAM3_CKPT),
        sam3_bpe_path=str(SAM3_BPE) if SAM3_BPE.is_file() else None,
        device=device,
    )
    print("✓ SAM3 chargé")

    masks = []
    for rec in tqdm(records, desc="Masques SAM3", unit="img"):
        img = Image.open(rec["path"]).convert("RGB")
        seg = predict_segmentation_with("sam3", sam3_handle, img, sam3_mode=sam3_mode)
        mask = seg["binary_mask"]
        masks.append(mask)
        coverage = mask.sum() / mask.size * 100
        tqdm.write(f"  {rec['Animal_ID']}-{rec['view']}: {coverage:.1f}% animal")

    # Free SAM3 GPU memory
    del sam3_handle
    torch.cuda.empty_cache()

    return masks


def _class_weights_and_prior(train_bcs: list, bcs_classes: list):
    """Inverse-frequency class weights + log-prior for the head warm-start."""
    counts = np.array([sum(1 for b in train_bcs if b == c) for c in bcs_classes], dtype=float)
    # Avoid div-by-zero if a class is absent from this fold's train split.
    freq = np.where(counts > 0, counts, 1.0) / max(len(train_bcs), 1)
    weights = (1.0 / freq)
    weights = weights / weights.mean()  # normalise around 1
    prior = counts / counts.sum() if counts.sum() > 0 else np.ones_like(counts) / len(counts)
    log_prior = np.log(np.clip(prior, 1e-6, None))
    return weights.tolist(), log_prior.tolist()


def train_fold(
    fold_cat: str,
    records: list,
    vit_ckpt: str,
    output_dir: Path,
    args: argparse.Namespace,
    bcs_classes: list,
    covariate_names: list,
) -> dict:
    """Train a single LOCO fold and return predictions."""
    fold_dir = output_dir / f"fold_{fold_cat}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # DataModule
    dm = BCSDataModule(
        records=records,
        fold_cat=fold_cat,
        batch_size=args.batch_size,
        mask_background=True,
        num_workers=args.num_workers,
    )

    train_bcs = [r["BCS"] for r in records if r["Animal_ID"] != fold_cat]

    # Model — classification (default) or legacy regression.
    if args.task == "classification":
        class_weights, log_prior = _class_weights_and_prior(train_bcs, bcs_classes)
        model = LitBCSClassification(
            bcs_classes=bcs_classes,
            backbone_ckpt=vit_ckpt,
            model_name="vit",
            num_classes=132,
            embedding_dim=768,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=1e-4,
            class_weights=class_weights,
            class_log_prior=log_prior,
            covariate_names=covariate_names,
        )
    else:
        # Warm-start the head's output bias at the training-set BCS mean so the
        # model starts by predicting the average and only learns residuals.
        target_mean = float(np.mean(train_bcs))
        model = LitBCSRegression(
            backbone_ckpt=vit_ckpt,
            model_name="vit",
            num_classes=132,
            embedding_dim=768,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=1e-4,
            target_mean=target_mean,
        )

    # Callbacks
    # Note: val set is only 2 images of the SAME cat (same BCS), so val metrics
    # are a poor signal for model selection. Monitor train/loss instead.
    callbacks = [
        ModelCheckpoint(
            dirpath=str(fold_dir),
            filename="best",
            monitor="train/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
    ]
    if args.patience > 0:
        callbacks.append(
            EarlyStopping(monitor="train/loss", mode="min", patience=args.patience)
        )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        precision=32,
        callbacks=callbacks,
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(model, dm)

    # Load best checkpoint for inference
    ModelCls = LitBCSClassification if args.task == "classification" else LitBCSRegression
    best_ckpt = fold_dir / "best.ckpt"
    if best_ckpt.exists():
        extra = (
            {"bcs_classes": bcs_classes, "covariate_names": covariate_names}
            if args.task == "classification"
            else {}
        )
        model = ModelCls.load_from_checkpoint(
            str(best_ckpt),
            backbone_ckpt=vit_ckpt,
            model_name="vit",
            num_classes=132,
            **extra,
        )

    # Predict on held-out cat
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.float().to(device)
    model.eval()
    dm.setup()

    val_recs = [r for r in records if r["Animal_ID"] == fold_cat]
    fold_results = []
    with torch.no_grad():
        idx = 0
        for x, cov, y in dm.val_dataloader():
            x = x.float().to(device)
            cov = cov.float().to(device)
            actuals = y.numpy().tolist()
            if args.task == "classification":
                logits = model(x, cov)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                exp_scores = model.expected_score(logits).cpu().numpy()
                for j in range(len(actuals)):
                    top = int(probs[j].argmax())
                    rec = val_recs[idx]
                    fold_results.append({
                        "Animal_ID": rec["Animal_ID"],
                        "view": rec["view"],
                        "BCS_actual": actuals[j],
                        "BCS_predicted": float(exp_scores[j]),  # expected score
                        "predicted_class": float(bcs_classes[top]),
                        "confidence": float(probs[j][top]),
                    })
                    idx += 1
            else:
                preds = model(x).float().cpu().numpy().tolist()
                for j in range(len(actuals)):
                    rec = val_recs[idx]
                    fold_results.append({
                        "Animal_ID": rec["Animal_ID"],
                        "view": rec["view"],
                        "BCS_actual": actuals[j],
                        "BCS_predicted": float(preds[j]),
                    })
                    idx += 1

    return {
        "fold_cat": fold_cat,
        "best_ckpt": str(best_ckpt) if best_ckpt.exists() else str(fold_dir / "last.ckpt"),
        "predictions": fold_results,
        "epochs_trained": trainer.current_epoch,
    }


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    # Output directory — per-species so cat and dog models live side by side
    # under checkpoints/bcs_regression/{cat,dog}/.
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else REPO_ROOT / "checkpoints" / "bcs_regression" / args.species
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Species: {args.species}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    # ── 1. Load dataset ──────────────────────────────────────────────
    if args.data_dir:
        data_dir = Path(args.data_dir)
    elif args.species == "cat":
        data_dir = REPO_ROOT / "data" / "Cats_OGR_dataset"
    else:
        data_dir = REPO_ROOT / "data" / "Dogs_BCS_dataset"

    if not data_dir.is_dir() or not any(data_dir.glob("*.xlsx")):
        print(
            f"\n[PLACEHOLDER] No BCS dataset found for species '{args.species}' at "
            f"{data_dir}.\nThe {args.species} BCS model cannot be trained yet. "
            f"Provide a dataset (Excel + images/ matching load_dataset()) and rerun:\n"
            f"    python scripts/train_bcs_regression.py --species {args.species} "
            f"--data-dir <path>\n"
        )
        # Keep the (empty) output dir so the cascade can detect 'no model yet'.
        sys.exit(0)

    df_obs = load_dataset(data_dir)
    print(f"\n{len(df_obs)} images, {df_obs['Animal_ID'].nunique()} animals, "
          f"BCS ∈ [{df_obs['BCS'].min()}, {df_obs['BCS'].max()}]")

    # Discrete BCS scores present in the dataset — the classes the classification
    # head predicts (ordered). Regression ignores this.
    bcs_classes = sorted(float(v) for v in df_obs["BCS"].unique())
    print(f"Task: {args.task} | BCS classes présentes: {bcs_classes}")

    # Covariate (metadata) conditioning: only for classification, when enabled,
    # and only if the metadata actually varies (a constant covariate carries no
    # signal, so we drop it and report why).
    covariate_names: list = []
    if args.task == "classification" and args.covariates:
        varies = {
            "sex": df_obs["Sex"].nunique(dropna=True) > 1,
            "long_coat": df_obs["Long_Coat_YN"].nunique(dropna=True) > 1,
        }
        if varies["sex"] or varies["long_coat"]:
            covariate_names = list(BCS_COVARIATE_NAMES)
            print(f"Covariables (metadata) activées: {covariate_names} | varie: {varies}")
            print("  ⚠ 11 chats : tout gain mesuré est probablement de l'overfitting.")
        else:
            print("Covariables demandées mais toutes constantes → désactivées (aucun signal).")

    # ── 2. Pre-compute SAM3 masks ────────────────────────────────────
    records = df_obs.to_dict("records")
    masks = compute_masks(records, device, args.sam3_mode)
    for rec, mask in zip(records, masks):
        rec["mask"] = mask
    # Encode per-animal metadata into the numeric ``cov`` vector each record
    # carries into the datamodule (empty when covariates are off).
    for rec in records:
        if covariate_names:
            enc = encode_bcs_covariates(sex=rec.get("Sex"), long_coat=rec.get("Long_Coat_YN"))
            rec["cov"] = covariates_to_vector(covariate_names, enc)
        else:
            rec["cov"] = []

    # ── 3. LOCO Cross-Validation ─────────────────────────────────────
    vit_ckpt = str(resolve_classifier_ckpt())
    cat_ids = sorted(df_obs["Animal_ID"].unique())
    print(f"\n── Entraînement LOCO ({len(cat_ids)} folds) ──")
    print(f"Backbone ViT: {vit_ckpt}")
    print(f"Hyperparams: epochs={args.max_epochs}, lr={args.lr}, "
          f"hidden={args.hidden_dim}, dropout={args.dropout}, bs={args.batch_size}")

    all_fold_results = []
    all_predictions = []
    t0 = time.time()

    fold_bar = tqdm(
        list(enumerate(cat_ids)), desc="LOCO folds", unit="fold", total=len(cat_ids),
    )
    for fold_idx, fold_cat in fold_bar:
        fold_bar.set_postfix_str(f"hold-out={fold_cat}")
        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"  Fold {fold_idx+1}/{len(cat_ids)} — Hold-out: {fold_cat}")
        tqdm.write(f"{'='*60}")

        fold_result = train_fold(fold_cat, records, vit_ckpt, output_dir, args, bcs_classes, covariate_names)
        all_fold_results.append(fold_result)
        all_predictions.extend(fold_result["predictions"])

        # Print fold predictions
        for pred in fold_result["predictions"]:
            tqdm.write(f"  {pred['Animal_ID']}-{pred['view']}: "
                  f"actual={pred['BCS_actual']:.1f}, predicted={pred['BCS_predicted']:.2f}")

    elapsed = time.time() - t0

    # ── 4. Global metrics ─────────────────────────────────────────────
    # On the expected/continuous score (comparable across tasks).
    actuals = np.array([p["BCS_actual"] for p in all_predictions])
    preds = np.array([p["BCS_predicted"] for p in all_predictions])
    mae = float(np.mean(np.abs(actuals - preds)))
    mse = float(np.mean((actuals - preds) ** 2))
    rmse = float(np.sqrt(mse))

    metrics = {"mae": mae, "rmse": rmse, "mse": mse}

    print(f"\n{'='*60}")
    print(f"  RÉSULTATS GLOBAUX (LOCO-CV) — task={args.task}")
    print(f"{'='*60}")

    if args.task == "classification":
        # Honest evaluation: accuracy + confusion matrix + reference baselines.
        pred_cls = np.array([p["predicted_class"] for p in all_predictions])
        accuracy = float(np.mean(pred_cls == actuals))
        cls_arr = np.array(bcs_classes)
        confusion = np.zeros((len(bcs_classes), len(bcs_classes)), dtype=int)
        for a, pcl in zip(actuals, pred_cls):
            confusion[int(np.argmin(np.abs(cls_arr - a)))][int(np.argmin(np.abs(cls_arr - pcl)))] += 1

        # Baselines: majority class accuracy, and mean-predictor MAE.
        vals, counts = np.unique(actuals, return_counts=True)
        baseline_majority_accuracy = float(counts.max() / counts.sum())
        baseline_mean_mae = float(np.mean(np.abs(actuals - actuals.mean())))

        metrics.update({
            "accuracy": accuracy,
            "mae_expected": mae,
            "confusion_matrix": confusion.tolist(),
            "confusion_labels": bcs_classes,
            "baseline_majority_accuracy": baseline_majority_accuracy,
            "baseline_mean_mae": baseline_mean_mae,
        })
        print(f"  Accuracy         = {accuracy:.3f}  (baseline majorité = {baseline_majority_accuracy:.3f})")
        print(f"  MAE (attendu)    = {mae:.3f}  (baseline moyenne = {baseline_mean_mae:.3f})")
        print(f"  RMSE             = {rmse:.3f}")
        print(f"  Classes          = {bcs_classes}")
        print(f"  Matrice confusion (lignes=réel, cols=prédit):\n{confusion}")
        beats = "✓" if accuracy >= baseline_majority_accuracy else "✗"
        print(f"  {beats} bat la baseline majorité : {accuracy:.3f} vs {baseline_majority_accuracy:.3f}")
    else:
        print(f"  MAE  = {mae:.3f}")
        print(f"  RMSE = {rmse:.3f}")
        print(f"  MSE  = {mse:.3f}")

    print(f"  Temps total : {elapsed:.1f}s")

    # ── 5. Save results ──────────────────────────────────────────────
    results = {
        "config": {
            "species": args.species,
            "task": args.task,
            "bcs_classes": bcs_classes,
            "covariate_names": covariate_names,
            "max_epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "patience": args.patience,
            "sam3_mode": args.sam3_mode,
            "seed": args.seed,
            "backbone_ckpt": vit_ckpt,
        },
        "metrics": metrics,
        "folds": all_fold_results,
        "predictions": all_predictions,
        "elapsed_seconds": elapsed,
    }

    results_path = output_dir / "predictions.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Résultats sauvegardés → {results_path}")
    print(f"✓ Checkpoints sauvegardés → {output_dir}/fold_*/best.ckpt")


if __name__ == "__main__":
    main()
