# Entraînement & modèles

> Tout ce qui concerne l'entraînement : point d'entrée `train.py` (Hydra/Optuna),
> les LightningModules, les backbones, la fabrique d'entraînement, les DataModules,
> les datasets et les configs. Lire ce fichier avant de lancer ou modifier un
> entraînement, un datamodule, un modèle ou une config.

## Point d'entrée — `train.py`

Point d'entrée léger décoré Hydra. Étapes :
1. Validation de la config → `utils.config_utils.validate_config`
   (schéma : `utils.config_validation`).
2. Setup des dossiers d'expérience → `utils.config_utils.setup_experiment_dirs`.
3. Construction du **DataModule** selon `dataset` (voir table plus bas).
4. Construction du **modèle** : `LitClassificationModule` ou `LitSegmentationModule`.
5. Construction du **Trainer** (callbacks + loggers) → `trainer_factory.build_trainer`.
6. `trainer.fit()` puis `trainer.test()`.

Le mapping `dataset → DataModule` :

| `dataset` | DataModule | Classes |
|---|---|---|
| `stanford` | `StanfordClassificationDataModule` | 120 races chiens |
| `oxford` | `OxfordClassificationDataModule` | Oxford-IIIT |
| `combined` | `CombinedDogsCatsDataModule` | 132 (chiens + chats) |
| `species` | `SpeciesClassificationDataModule` | 2 (0=chien, 1=chat) |
| `cat_breed` | `CatBreedClassificationDataModule` | 12 chats |

> `combined`, `species`, `cat_breed` partagent le constructeur du datamodule
> combiné ; seul le mapping de labels diffère.

## Configs Hydra — `configs/`

| Fichier | Usage |
|---|---|
| `config.yaml` | Classification Stanford Dogs (120), défaut ViT |
| `config_dogs_cats.yaml` | Classification combinée (132) |
| `config_dogs_cats_vit.yaml` | Classification combinée ViT (132) |
| `config_species.yaml` | Espèce binaire (2), ViT — étage 1 |
| `config_dog_breed.yaml` | Race chien (Stanford 120) |
| `config_cat_breed.yaml` | Race chat (Oxford 12) |
| `config_segmentation.yaml` | Segmentation DeepLabV3 |

Clés usuelles : `model_name`, `num_classes`, `dataset`, `lr`, `optimizer_name`,
`scheduler_config`, `regularization.*` (dropout, label_smoothing, mixup/cutmix,
stochastic_depth), `stanford_data_dir`, `oxford_data_dir`.

⚠️ **Toute nouvelle valeur de `dataset` ou `model_name` doit être ajoutée au
schéma** `utils/config_validation.py` (`Literal[...]`) sinon la validation échoue.

Lancement / sweep / TensorBoard : voir section *Training* du [../README.md](../README.md).

## LightningModules — `lightning_module/`

| Classe (fichier) | Rôle |
|---|---|
| `LitClassificationModule` (`classification_module.py`) | Classif race/espèce : mixup/cutmix, label smoothing, stochastic depth, logs TB (images, matrice de confusion, courbes PR, histos). `top_k=min(5, num_classes)` |
| `LitSegmentationModule` (`segmentation_module.py`) | Segmentation DeepLabV3 : loss CE+Dice, métriques IoU/Dice/PixelAcc par classe, overlays TB |
| `LitBCSRegression` (`bcs_regression_module.py`) | BCS **régression** : backbone ViT gelé + `BCSRegressionHead` (MLP 768→hidden→1), tête warm-startée sur la moyenne cible |
| `LitBCSClassification` (`bcs_classification_module.py`) | BCS **classification ordinale** : backbone ViT gelé + `BCSClassificationHead`, class-weights inverse-fréquence ; backbone partagé via `_BCSBackboneMixin` |

Notes BCS :
- Le backbone est **gelé** (`requires_grad=False`, `.eval()`) ; seule la tête entraîne.
- `forward` = `head(extract_features(x))` ; `extract_features` prend le token CLS
  du dernier hidden state du ViT. Bien garder modèle **et** entrée sur le même device.
- Le task (régression / classif) est détecté à l'inférence via l'hyperparamètre
  `bcs_classes` (voir [architecture.md](architecture.md)).

## Backbones — `models/`

| Classe | Rôle |
|---|---|
| `ResNetTransfer` (`resnet_transfer.py`) | ResNet-50 (poids ImageNet) + dropout + stochastic depth optionnel |
| `ViTTransfer` (`vit_transfer.py`) | Wrapper fine-tuning `google/vit-base-patch16-224-in21k` |

## Fabrique d'entraînement

| Module | Fonctions clés |
|---|---|
| `callbacks.py` | `build_checkpoint_callback` (ModelCheckpoint top-k `val/acc`), `build_early_stopping_callback`, `build_lr_monitor`, `build_callbacks` (entrée principale). **`auto_insert_metric_name=False`** pour éviter `epoch=epoch=` et le `/` de `val/acc` interprété comme sous-dossier |
| `loggers.py` | `build_tensorboard_logger`, `build_wandb_logger` (dégradation gracieuse), `build_loggers` |
| `trainer_factory.py` | `build_trainer(cfg, dirs)`, `get_checkpoint_callback(trainer)` |

## DataModules & datasets — `data/` + `datasets.py`

DataModules (`data/`) : `stanford_classification_datamodule`,
`stanford_segmentation_datamodule`, `oxford_classification_datamodule`,
`oxford_segmentation_datamodule`, `combined_classification_datamodule`,
`species_classification_datamodule`, `cat_breed_classification_datamodule`,
`bcs_datamodule`.

- **`bcs_datamodule.py`** : `BCSDataset` (retourne `(image_masquée, bcs)`),
  `BCSDataModule` (Leave-One-Cat-Out). `BCSDataset._apply_mask(img, mask)` met le
  fond en gris neutre (ImageNet mean) ; constantes `IMAGENET_MEAN/STD` partagées
  avec `inference/bcs.py`. Le prétraitement (Resize 256 → CenterCrop 224 → ToTensor
  → Normalize) doit rester **identique** entre entraînement et inférence.
- ⚠️ `src/bcs_pipeline/data/` est masqué aux outils de recherche par le motif
  `data/` du `.gitignore` (les fichiers sont pourtant suivis) — lire directement.

`datasets.py` (racine du package) centralise la **découverte des images** des
datasets (partagé `app.py` ↔ `scripts/preload_db.py`) :
`collect_all_images`, `get_datasets`, `resolve_image_path`, `ground_truth`,
`list_image_files`.

Layout attendu des données (`data/`) : voir *Préparation des données* du
[../README.md](../README.md).

## Entraînement BCS — `scripts/train_bcs_regression.py`

- LOCO-CV (une tête par chat exclu) sur le dataset OGR (`data/Cats_OGR_dataset`,
  annotations `.xlsx` → nécessite `openpyxl`).
- Args clés : `--species {cat,dog}`, `--data-dir`, `--task {regression,classification}`
  (défaut `classification`), `--max-epochs`, `--sam3-mode`, `--output-dir`.
- Sortie **par espèce** : `checkpoints/bcs_regression/<species>/fold_*/` +
  `predictions.json` (config, métriques, folds, prédictions, baselines).
- `dog` est un **placeholder** : sans dataset, le script sort proprement (dossier
  vide → la cascade détecte « pas de modèle »).
- `scripts/run_cascade_trainings.sh` enchaîne species → dog_breed → cat_breed → bcs.

## Utilitaires — `utils/`

| Module | Rôle |
|---|---|
| `config_utils` | Setup dossiers d'expérience, validation, snapshot de config |
| `config_validation` | Schéma Pydantic des configs Hydra (allowlist `dataset`/`model_name`) |
| `dataset_stats` | Statistiques par classe (compute/display/log/save JSON) |
| `device` | `get_best_device` (CUDA/MPS/CPU) |
| `logging_utils` | Setup logging, affichage Rich, infos d'expérience |
