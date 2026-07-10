# Architecture & inférence

> Vue d'ensemble du projet, de la **cascade** de modèles, de la carte du package
> `src/bcs_pipeline`, du flux de données et du **pipeline d'inférence** (package
> `inference/`, orchestration partagée, CLI). Lire ce fichier avant toute action
> touchant l'inférence, le pipeline ou la structure du package.

## Vue d'ensemble

Le projet estime le **Body Condition Score** (BCS, échelle clinique 1–9) d'un
animal à partir d'une image, en combinant plusieurs modèles de vision en
**cascade**. Chaque étage conditionne le suivant.

```
Image ─▶ [1] Espèce (chien/chat) ─▶ [2] Race (routée par espèce)
                                     │
                                     ▼
        [4] BCS (routé par espèce) ◀── [3] Segmentation (silhouette)
                                            + Pose (keypoints, optionnelle)
```

- **[1] Espèce** — classifieur binaire ViT (chien/chat). Route la race et le BCS.
- **[2] Race** — classifieur dédié selon l'espèce (chien : Stanford 120 ; chat :
  Oxford 12) ; repli sur le classifieur combiné (132) si pas de modèle routé.
- **[3] Segmentation** — DeepLabV3 / SAM 2 / SAM 3 ; produit le masque silhouette.
- **Pose** — YOLO (keypoints + bbox), aussi réutilisée comme invites visuelles
  SAM 2/3. Tourne **avant** la segmentation quand le backend consomme la pose.
- **[4] BCS** — backbone ViT gelé + tête (régression **ou** classification
  ordinale) ; consomme le masque pour isoler la silhouette (fond gris neutre),
  route vers le modèle chat/chien selon l'espèce, ensemble les têtes par fold.

> Détail « produit » et métriques : voir [../README.md](../README.md).
> Statut d'entraînement de chaque étage : [BCS_CASCADE_PROGRESS.md](BCS_CASCADE_PROGRESS.md).

## Carte du package `src/bcs_pipeline`

| Module / dossier | Rôle |
|---|---|
| `app_checkpoints.py` | **Source unique** des chemins de checkpoints (`resolve_*`, `*_available`) |
| `datasets.py` | Découverte/résolution des images des datasets (partagé app ↔ preload) |
| `inference_format.py` | Orchestration cascade + formatage du résultat (partagé app ↔ preload) |
| `db.py` | Modèles SQLAlchemy + persistance — voir [webapp.md](webapp.md) |
| `callbacks.py`, `loggers.py`, `trainer_factory.py` | Fabrique d'entraînement — voir [training.md](training.md) |
| `inference/` | Utilitaires d'inférence partagés (détaillés ci-dessous) |
| `data/` | DataModules Lightning — voir [training.md](training.md) |
| `lightning_module/` | LightningModules — voir [training.md](training.md) |
| `models/` | Backbones transfer (ResNet, ViT) — voir [training.md](training.md) |
| `utils/` | Config, device, logging, stats |

## Package `inference/`

Toutes les briques d'inférence, réutilisées par `inference.py` (CLI), `app.py`
et les notebooks. L'API publique est réexportée par `inference/__init__.py`.

| Fichier | Fonctions clés | Rôle |
|---|---|---|
| `classification.py` | `load_classification_model`, `predict_single`, `predict_batch`, `load_class_names`, `load_combined_class_names`, `load_dog_class_names`, `load_cat_class_names`, `get_inference_transform` | Prédiction de race (ResNet/ViT) + noms de classes |
| `species.py` | `load_species_model`, `predict_species`, `SPECIES_CLASS_NAMES` | Étage 1 — espèce (binaire, `num_classes=2`, ordre `[dog, cat]`) |
| `segmentation.py` | `load_segmentation_backend`, `predict_segmentation_with` | DeepLabV3 (trimap 3 classes) + dispatch de backend |
| `segmentation_sam2.py` | `load_sam2_model`, `predict_segmentation_sam2` | SAM 2 zero-shot (modes `prompted`/`automatic`/`pose_prompted`) |
| `segmentation_sam3.py` | (chargé via dispatch) | SAM 3 zero-shot race-aware (4 modes, prompt textuel + pose) |
| `pose.py` | `load_pose_model`, `predict_pose` | Keypoints + bbox YOLO |
| `bcs.py` | `load_bcs_model`, `load_bcs_models`, `predict_bcs`, `get_bcs_transform`, `bcs_category` | Étage 4 — BCS (ensemble des folds, régression ou classif) |
| `visualization.py` | `render_combined`, `save_visualization`, `draw_label_banner`, `draw_pose`, `overlay_segmentation` | Rendu PIL des overlays |
| `coco_export.py` | `build_coco`, `save_coco` | Export COCO JSON + polygones |
| `pipeline.py` | `run_full_inference` | Orchestrateur combiné (tous les étages) |

### Détails cascade importants
- **`predict_species`** renvoie `{"species": "dog"|"cat", "confidence", "top_k"}`.
  L'ordre des labels doit rester `[dog, cat]` (= `SpeciesClassificationDataModule`).
- **`bcs.py`** attend la **silhouette seule** : les pixels de fond sont mis en gris
  neutre (ImageNet mean) via le masque de segmentation, cohérent avec le prétraitement
  de `data/bcs_datamodule.py::BCSDataset`. Le prétraitement d'inférence
  (`get_bcs_transform`) doit **rester identique** à la transform de validation.
- **Ensemble par folds** : la LOCO-CV produit une tête par chat exclu ; l'inférence
  moyenne toutes les têtes `fold_*/best.ckpt` sur un backbone gelé partagé.
- **Routage BCS** : `bcs_ckpt` peut être un dossier avec sous-dossiers `cat/`/`dog/`,
  un dossier par espèce, ou un seul `.ckpt`. Une espèce reconnue **sans** modèle
  propre (ex. chien) tombe dans la branche « indisponible », **pas** sur le modèle chat.
- **Régression vs classification ordinale** : `inference/bcs.py` détecte la tâche via
  l'hyperparamètre `bcs_classes` du checkpoint (régression = rétrocompatible). La
  sortie aval reste `{"bcs", "category", ...}` quelle que soit la tâche.

## Orchestration partagée — `inference_format.py`

- `run_core_inference(cls_model, class_names, seg_handle, seg_backend, pose_model,
  img, ..., species_model=None, dog_breed=None, cat_breed=None)` exécute la cascade
  **espèce → race (routée) → pose? → segmentation** et renvoie
  `(cls, seg, pose, species)`.
- `format_inference_result(cls, seg, pose, image_name, size, seg_backend, bcs=None,
  species=None)` normalise les résultats en dict prêt pour l'API/DB.
- Ce module est **partagé** entre `app.py` et `scripts/preload_db.py` : toute
  évolution de la signature doit être répercutée dans les deux appelants.

## Orchestrateur combiné — `pipeline.py::run_full_inference`

- Exécute les étages dont le checkpoint est fourni, puis compose une PNG combinée
  et un sidecar COCO JSON.
- Renvoie un dict `{"classification", "segmentation", "pose", "bcs", "species",
  "output_path", "coco", "coco_path"}` (branches absentes = `None`).
- Params espèce/race/BCS : `species_ckpt`, `dog_breed_ckpt`, `cat_breed_ckpt`,
  `bcs_ckpt`, `stanford_data_dir`, `oxford_data_dir`.

## CLI — `inference.py`

Deux modes :
- `--mode classify` (défaut, legacy) : top-k races d'un checkpoint de classification.
- `--mode full` : lance les étages dont le checkpoint est fourni, compose une PNG.

Options cascade principales : `--checkpoint_path`, `--seg_checkpoint`,
`--pose_checkpoint`, `--bcs_checkpoint`, `--species_checkpoint`,
`--dog_breed_checkpoint`, `--cat_breed_checkpoint`, `--stanford_data_dir`,
`--oxford_data_dir`, `--output`. La logique lourde vit dans `bcs_pipeline.inference`
pour être partagée avec `app.py` et les notebooks.

Exemple complet : voir la section *Inference* du [../README.md](../README.md).
