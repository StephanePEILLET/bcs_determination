# Body Condition Score (BCS) Determination

> **Estimation automatique du Body Condition Score (BCS) d'un chien à partir d'une image, en combinant classification de race, segmentation sémantique et détection de pose.**

## Contexte

Le **Body Condition Score** est un indicateur clinique (échelle 1–9) utilisé par les vétérinaires pour évaluer l'état corporel d'un animal (embonpoint, idéal, maigreur). Cette estimation repose today'hui sur une palpation manuelle et une évaluation visuelle subjective. Ce projet explore une approche **entièrement visuelle et automatisée** basée sur le deep learning.

## Démarrage rapide

```bash
# 1. Installer les dépendances
uv sync --extra dev                   # inclut SAM 3 par défaut

# 2. Lancer l'application web
uv run python app.py                  # http://localhost:8000

# 3. Ou lancer une inférence en CLI
uv run python inference.py --mode full --image_path data/Reddit_example/dog.jpg
```

<details>
<summary><strong>Bootstrap complet (données + checkpoints + app)</strong></summary>

```bash
chmod +x scripts/setup_and_run.sh
./scripts/setup_and_run.sh            # setup complet + lancement
./scripts/setup_and_run.sh --preload  # idem + pré-chargement DB
./scripts/setup_and_run.sh --cpu      # forcer CPU (pas de GPU)
```

</details>

<details>
<summary><strong>Conda (alternative)</strong></summary>

```bash
conda env create -f environment.yaml
conda activate bcs_analysis
pip install -e ".[dev]"
python app.py                         # http://localhost:8000
```

</details>

<details>
<summary><strong>Docker</strong></summary>

```bash
docker build -t bcs_determination .
docker run -p 8000:8000 bcs_determination
```

</details>

---

## Approche

Le BCS d'un chien dépend fortement de sa **race** (un Greyhound et un Bulldog n'ont pas la même morphologie), de la **forme de son silhouette** (répartition graisse/muscle visible depuis le dessus et le côté), et de sa **posture**. Le pipeline combine trois piliers complémentaires :

### 1. Classification de race (jusqu'à 132 races — chiens + chats)

Identifier la race permet d'ajuster les attentes morphologiques : un Whippet naturellement mince ne doit pas être jugé comme un Labrador en sous-poids.

Deux classifieurs sont fournis et **commutables depuis l'UI** :

- **Chiens uniquement** — Stanford Dogs (120 races)
- **Chiens + Chats** — Stanford Dogs (120 races) + Oxford-IIIT Pet cats only (12 races) = **132 classes** (config `configs/config_dogs_cats.yaml`)

Modèles entraînés :

- **ResNet-50** (ImageNet pretrained, fine-tuned) — val_acc = 0.79
- **ViT-B/16** (google/vit-base-patch16-224-in21k, fine-tuned) — val_acc = 0.87

| Modèle | Backbone | Top-1 | Top-5 | Paramètres |
|---|---|---|---|---|
| ResNet-50 | CNN (23M) | 79% | — | 23,7M |
| ViT-B/16 | Transformer (86M) | 87% | — | 86M |

### 2. Segmentation sémantique (Oxford-IIIT Pet)

Isoler le contour du chien dans l'image permet d'analyser sa **silhouette** : un chien en surpoids présentera un contour plus large au niveau des côtes et de la taille, sans taille marquée vue de dessus.

Trois backends interchangeables :

- **DeepLabV3-ResNet50** (COCO pretrained, fine-tuned) — val_IoU = 0.82, 3 classes trimap (*foreground*, *background*, *border*).
  Métriques : Pixel Accuracy = 93.7%, mIoU = 0.82, mDice = 0.89.
- **SAM 2** (zero-shot, 3 modes : `prompted`, `automatic`, `pose_prompted`) — masque à **2 classes** (*foreground*, *background*) directement issu du masque binaire SAM, sans bordure dérivée (silhouette déjà nette).
- **SAM 3** (zero-shot, race-aware, 4 modes : `prompted`, `pose_prompted`, `concept_prompted`, `pose_concept_prompted`) — masque 2 classes guidé par un *prompt textuel* (la race prédite par le classifier) combiné aux invites visuelles YOLO (bbox + keypoints). Voir [§ Setup SAM 3](#sam-3-setup-optionnel) pour l'installation (modèle gated HuggingFace).

Le masque est **éditable côté navigateur** (pinceau pour ajouter au foreground, gomme pour repasser au background) et le résultat est persisté en base avec les autres annotations.

### 3. Détection de pose (Ultralytics YOLO)

Repérer les **points clés anatomiques** (colonne vertébrale, hanches, côtes, queue) permet de mesurer des rapports de proportions corporelles utilisés dans les grilles BCS vétérinaires (ex. visibilité des côtes, présence d'une taille vue de dessus, épaisseur de la base de la queue).

- Modèle **Ultralytics YOLO pose** fine-tuné sur chiens/chats (`checkpoints/pose/yolo_best.pt`)
- Extraction de keypoints + bounding boxes, réutilisés comme invites visuelles pour SAM 2 / SAM 3

### 4. Vision classique (baseline sans deep learning)

Des méthodes de **détection de contours** traditionnelles (Canny, Sobel, Laplacien, Prewitt) servent de baseline pour comparer l'apport du deep learning sur l'extraction de silhouette :

- OpenCV : Canny multi-seuils, Sobel, Laplacien
- scikit-image : Canny multi-scale, Prewitt, Roberts, Scharr
- Kornia (GPU, différentiable) : Sobel, Canny, Laplacian

### 5. Prédiction du BCS (régression)

Pilier final qui consomme les précédents : la silhouette segmentée (fond remis en gris neutre) est encodée par le **backbone ViT** de la classification (figé), puis une petite **tête MLP** régresse le score corporel continu (échelle 1–9).

- Backbone ViT figé (`vit_dogs_cats`) → embedding CLS (768) → MLP (768→128→1)
- Entraînement **Leave-One-Cat-Out** sur le dataset OGR (`scripts/train_bcs_regression.py`), checkpoints dans `checkpoints/bcs_regression/fold_*/`
- Inférence par **ensemble** : la prédiction moyenne des 11 têtes (une par fold) est servie, l'écart-type inter-fold servant d'indicateur d'incertitude
- Branché dans le pipeline complet (`inference.py --mode full`), l'API/UI web (badge BCS + section dédiée) et l'historique (colonne BCS)

> ⚠️ **Limite actuelle du modèle (qualité, pas intégration).** Sur le jeu OGR (11 chats, 22 vues), la métrique LOCO-CV est MAE = 0.82 / RMSE = 0.92 — mais le modèle **collapse vers la moyenne** : il prédit ~4.8 quasi quel que soit l'entrée (étendue des prédictions 4.69–4.91 contre un vrai BCS de 4 à 6). Le MAE correspond donc essentiellement au baseline « toujours prédire la moyenne ». À améliorer : davantage de données, dégel partiel du backbone, ou features géométriques issues de la pose. L'intégration sert le score tel que produit par le modèle, sans le masquer.

## Stack technique

| Composant | Technologie |
|---|---|
| Framework DL | PyTorch + PyTorch Lightning |
| Hyperparamètres | Hydra + Optuna |
| Logging | TensorBoard (+ W&B optionnel) |
| Modèles | torchvision (ResNet, DeepLabV3), HuggingFace Transformers (ViT), SAM 2, SAM 3 (optionnel), YOLOv8 |
| Serveur d'inférence | FastAPI + Uvicorn (UI interactive, édition d'annotations) |
| Persistance | SQLite via SQLAlchemy (runs + annotations utilisateur) |
| Container | Docker |

## Notebooks d'évaluation

| Notebook | Description |
|---|---|
| `reddit_images_tests_models.ipynb` | **Vue d'ensemble** : les 3 modèles sur images Reddit + contours classiques |
| `evaluate_vit_classification_Stanford_Dogs.ipynb` | Évaluation complète ViT (courbes, matrice de confusion, confiance, Reddit) |
| `visualize_results_Race_Classif_Stanford_Dogs.ipynb` | Évaluation ResNet-50 (t-SNE, calibration, rapports par classe) |
| `evaluate_segmentation_oxford_pet.ipynb` | Évaluation DeepLabV3 (IoU/Dice, overlays, Reddit) |
| `sam2_comparison.ipynb` | Comparaison SAM2 vs DeepLabV3 pour la segmentation |
| `combined_inference_overlay.ipynb` | **Pipeline combiné** : classification + segmentation + pose avec widget interactif |
| `edge_detection.ipynb` | Comparaison exhaustive de contours (skimage, Kornia, OpenCV) |
| `unsupervised_segmentation.ipynb` | Baselines de segmentation non supervisée |
| `pose_detection.ipynb` | Détection de pose avec OpenPose |

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Setup & Installation](#setup--installation)
3. [Training](#training)
4. [Inference](#inference)
5. [Application web](#application-web)
6. [Evaluation Notebooks](#evaluation-notebooks)
7. [Architecture & Module Reference](#architecture--module-reference)
8. [Configuration Reference](#configuration-reference)

## Project Structure

```
bcs_determination/
├── configs/
│   ├── config.yaml                        # Classification Hydra config (Stanford Dogs, 120 races)
│   ├── config_dogs_cats.yaml              # Classification combinée (132 races, chiens + chats)
│   ├── config_dogs_cats_vit.yaml          # Classification ViT (132 races, chiens + chats)
│   └── config_segmentation.yaml           # Segmentation Hydra config
├── src/
│   └── bcs_pipeline/                      # Main Python package
│       ├── __init__.py
│       ├── app_checkpoints.py             # Single source of truth pour les chemins de checkpoints
│       ├── datasets.py                    # Constantes datasets, collecte d'images, résolution de chemins
│       ├── inference_format.py            # Orchestration inference + formatage résultat (partagé app/script)
│       ├── callbacks.py                   # Callback factories
│       ├── loggers.py                     # Logger factories (TensorBoard, W&B)
│       ├── trainer_factory.py             # High-level Trainer builder
│       ├── db.py                          # SQLAlchemy models + session_scope context manager
│       ├── inference/                     # Shared inference utilities
│       │   ├── __init__.py                # Public API re-exports
│       │   ├── classification.py          # ResNet/ViT breed prediction (+ load_combined_class_names)
│       │   ├── segmentation.py            # DeepLabV3 trimap + backend dispatch
│       │   ├── segmentation_sam2.py       # SAM 2 zero-shot segmentation (2 classes)
│       │   ├── pose.py                    # YOLOv8 keypoint detection
│       │   ├── visualization.py           # PIL overlay rendering
│       │   ├── pipeline.py                # Combined orchestrator
│       │   └── coco_export.py             # Export COCO JSON + polygon masks
│       ├── data/
│       │   ├── stanford_classification_datamodule.py
│       │   ├── stanford_segmentation_datamodule.py
│       │   ├── oxford_classification_datamodule.py
│       │   ├── oxford_segmentation_datamodule.py
│       │   └── combined_classification_datamodule.py   # Stanford dogs + Oxford cats (132 cls)
│       ├── lightning_module/
│       │   ├── classification_module.py   # LitClassificationModule
│       │   └── segmentation_module.py     # LitSegmentationModule
│       ├── models/
│       │   ├── resnet_transfer.py         # ResNet-50 transfer backbone
│       │   └── vit_transfer.py            # ViT-B/16 transfer backbone
│       └── utils/
│           ├── config_utils.py
│           ├── config_validation.py
│           ├── dataset_stats.py
│           ├── device.py                  # GPU/CPU/MPS auto-detection
│           └── logging_utils.py
├── templates/
│   └── index.html                         # FastAPI UI (single-page) — édition annotations + masque
├── static/
│   └── images/                            # Logo et favicon
├── notebooks/
│   ├── combined_inference_overlay.ipynb   # Widget interactif (même logique que l'app)
│   ├── evaluate_vit_classification_Stanford_Dogs.ipynb
│   ├── visualize_results_Race_Classif_Stanford_Dogs.ipynb
│   ├── evaluate_segmentation_oxford_pet.ipynb
│   ├── reddit_images_tests_models.ipynb
│   ├── sam2_comparison.ipynb
│   ├── edge_detection.ipynb
│   ├── unsupervised_segmentation.ipynb
│   └── pose_detection.ipynb
├── data/
│   ├── stanford_dogs/                     # 120 breed folders (sous images/Images/)
│   ├── Oxford-IIIT_pet_dataset/           # Pet images + trimaps (chats utilisés pour le combiné)
│   ├── Reddit_example/                    # Out-of-distribution images (téléchargées automatiquement)
│   ├── outputs/                           # JSON + PNG de chaque run (run_<id>.json, run_<id>_mask.png)
│   └── bcs_app.db                         # SQLite — historique des runs et annotations utilisateur
├── checkpoints/                           # Modèles entraînés + SAM 2/3 weights
│   ├── classification/
│   │   └── vit_dogs_cats/last.ckpt        # Classifieur actif (ViT, 132 classes)
│   ├── segmentation/
│   │   ├── deeplabv3_resnet50_last-v1.ckpt
│   │   ├── sam2.1_hiera_large.pt
│   │   └── sam3/                          # SAM 3 (HuggingFace gated)
│   │       ├── sam3.pt
│   │       └── bpe_simple_vocab_16e6.txt.gz
│   └── pose/
│       └── yolo_best.pt
├── scripts/
│   ├── setup_and_run.sh                   # Bootstrap : uv venv + données + checkpoints + lancement
│   └── preload_db.py                      # Pré-chargement DB : inférences sur toutes les images
├── train.py                               # Training entry-point
├── inference.py                           # CLI inference
├── app.py                                 # FastAPI/Uvicorn web app (UI interactive)
├── environment.yaml
├── Dockerfile
└── README.md
```

### Design Principles

| Principle | How it's applied |
|---|---|
| **Modularity** | Each concern (callbacks, loggers, trainer, inference, datasets, DB) is in its own module. |
| **Reusability** | `bcs_pipeline.inference` is shared by `inference.py`, `app.py`, and notebooks. `datasets.py` and `inference_format.py` are shared between `app.py` and `scripts/preload_db.py`. |
| **Lightweight entry-points** | `train.py`, `inference.py`, and `app.py` contain only orchestration — no business logic. |
| **Configuration-driven** | All hyperparameters live in `configs/config.yaml` and can be overridden via CLI. |

---

## Setup & Installation

### Option A — `uv` direct (le plus simple)

Le projet est entièrement géré par [uv](https://docs.astral.sh/uv/) :
[`.python-version`](.python-version) épingle Python 3.12 et
[`pyproject.toml`](pyproject.toml) déclare toutes les dépendances + l'index
PyTorch CUDA 12.4 (Linux) via `[tool.uv.sources]`. Une seule commande
crée le venv `.venv/`, résout torch+CUDA, installe le projet et écrit
`uv.lock` pour la reproductibilité :

```bash
uv sync --extra dev                   # full (DeepLab + SAM 2 + SAM 3, par défaut)
```

Lancement de l'app sans avoir à activer le venv :

```bash
uv run python app.py
# ou :
uv run python inference.py --mode full --image_path data/Reddit_example/dog.jpg
```

### Option B — Bootstrap automatique avec datasets

Le script [`setup_and_run.sh`](scripts/setup_and_run.sh) reprend l'option A
et ajoute : auto-installation d'`uv` si absent, détection GPU (CUDA / MPS /
CPU), téléchargement des datasets (Stanford Dogs, Oxford-IIIT Pet, exemples
Reddit) + des checkpoints SAM 2 / SAM 3, puis lancement de l'app.

```bash
./scripts/setup_and_run.sh                  # tout faire
./scripts/setup_and_run.sh --skip-data      # passer le téléchargement des données
./scripts/setup_and_run.sh --skip-env       # ne pas re-synchroniser le venv
./scripts/setup_and_run.sh --no-launch      # arrêter avant de lancer l'app
./scripts/setup_and_run.sh --preload        # pré-charger la DB (inférences pré-calculées)
./scripts/setup_and_run.sh --cpu            # forcer BCS_DEVICE=cpu (pour les modèles)
```

### Option C — Conda manuel

```bash
conda env create -f environment.yaml
conda activate bcs_analysis
pip install -e ".[dev]"          # DeepLab + SAM 2 + SAM 3 (SAM 3 dans les deps principales)
```

> `environment.yaml` ne provisionne plus que Python 3.12 + PyTorch CUDA 12.4 ;
> les dépendances projet sont résolues par `pyproject.toml` via le
> `pip install -e`. À la différence d'`uv sync`, `pip` n'utilise pas les
> directives `[tool.uv.sources]` — le PyTorch CUDA arrive ici via la
> directive `pytorch-cuda` dans `environment.yaml`.

### SAM 3 — setup

Le backend SAM 3 (Meta, zero-shot avec prompts textuels) est inclus dans
les dépendances par défaut. Le checkpoint est *gated* sur HuggingFace et
requiert un environnement plus récent que les autres backends.

**Prérequis :**

- Python ≥ 3.12, PyTorch ≥ 2.7, CUDA ≥ 12.4 (12.6 recommandé pour
  flash-attn-3)
- Compte HuggingFace avec **accès accordé** au dépôt
  [`facebook/sam3`](https://huggingface.co/facebook/sam3) (formulaire
  Meta à valider, comptez ~1h pour l'approbation)
- ~3,4 Go libres pour le checkpoint

**Récupération du checkpoint :**

```bash
# 1. S'authentifier auprès de HuggingFace (token avec read access)
hf auth login

# 2. Télécharger le checkpoint dans checkpoints/segmentation/sam3/
hf download facebook/sam3 --local-dir checkpoints/segmentation/sam3
# (./scripts/setup_and_run.sh automatise les deux étapes)
```

**Modes disponibles** (sélecteur `sam3_mode` dans l'UI ou
`--sam3_mode` en CLI) :

| Mode | Description |
| --- | --- |
| `prompted` | Point positif au centre de l'image (fallback minimal). |
| `pose_prompted` | bbox YOLO + tous les keypoints visibles → invites visuelles SAM 3. |
| `concept_prompted` | Race prédite par le classifier injectée comme *text prompt* (`"golden retriever"`). |
| `pose_concept_prompted` *(défaut)* | Lance les deux branches et garde le masque le mieux scoré ; fallback gracieux quand un signal est absent. |

> Le BPE vocab (`bpe_simple_vocab_16e6.txt.gz`) est bundlé dans le package
> `sam3` — pas de download supplémentaire nécessaire.
>
> **Pin `setuptools<81`.** Le code amont `sam3/model_builder.py` fait
> `import pkg_resources` au chargement du module, et setuptools 81+ ne
> livre plus `pkg_resources` par défaut. La contrainte est portée par les
> dépendances principales du `pyproject.toml`, mais si vous voyez
> `ModuleNotFoundError: No module named 'pkg_resources'` au démarrage
> de l'app, exécutez `uv pip install 'setuptools<81'`.

### Vérification

```bash
python -c "import pytorch_lightning; print(pytorch_lightning.__version__)"
python -c "from torch.utils.tensorboard import SummaryWriter; print('TensorBoard OK')"
```

### Préparation des données (si Option B)

**Classification — Stanford Dogs**

Le `setup_and_run.sh` télécharge l'archive et l'extrait dans
`data/stanford_dogs/images/Images/`. Si vous installez à la main,
respectez cette structure :

```
data/stanford_dogs/images/
└── Images/
    ├── n02085620-Chihuahua/
    ├── n02085782-Japanese_spaniel/
    └── ...  (120 breed folders)
```

**Segmentation / Cats — Oxford-IIIT Pet**

```
data/Oxford-IIIT_pet_dataset/
├── images/
│   ├── Abyssinian_1.jpg
│   └── ...
└── annotations/
    ├── list.txt          # utilisé pour filtrer SPECIES=1 (chats)
    ├── trainval.txt
    ├── test.txt
    └── trimaps/
        ├── Abyssinian_1.png
        └── ...
```

**Images Reddit (exemples)**

Le script télécharge automatiquement deux images d'exemple depuis Reddit dans `data/Reddit_example/` :

```
data/Reddit_example/
├── reddit_dog_1.jpg
└── reddit_dog_2.jpg
```

---

## Training

Training is managed via [Hydra](https://hydra.cc/).  You can modify
`configs/config.yaml` **or** override values from the CLI.

### Classification (Stanford Dogs)

```bash
# Default config (ViT)
python train.py

# ResNet-50
python train.py model_name=resnet50

# Override hyperparameters
python train.py data_dir=data/Stanford_dogs batch_size=64 max_epochs=50

# Resume from a checkpoint
python train.py trainer.resume_from_checkpoint=experiments/.../checkpoints/last.ckpt
```

### Segmentation (Oxford-IIIT Pet)

```bash
python train.py --config-name config_segmentation
```

### Hyperparameter sweep (Optuna)

```bash
python train.py --multirun \
    lr=0.0001,0.001,0.005 \
    optimizer_name=adam,sgd \
    batch_size=16,32,64
```

### Monitoring with TensorBoard

```bash
tensorboard --logdir experiments/
```

---

## Inference

### CLI

```bash
python inference.py \
    --image_path sample_dog.jpg \
    --checkpoint_path experiments/.../checkpoints/best.ckpt \
    --data_dir data/stanford_dogs \
    --top_k 5
```

Pipeline complet avec score BCS (la régression consomme le masque de segmentation) :

```bash
python inference.py --mode full \
    --image_path data/Reddit_example/dog.jpg \
    --checkpoint_path checkpoints/classification/vit_dogs_cats/last.ckpt \
    --model_name vit --num_classes 132 \
    --seg_checkpoint checkpoints/segmentation/deeplabv3_resnet50_last-v1.ckpt \
    --pose_checkpoint checkpoints/pose/yolo_best.pt \
    --bcs_checkpoint checkpoints/bcs_regression \
    --output outputs/dog_full.png
```

### From Python

```python
from bcs_pipeline.inference import (
    load_classification_model,
    load_combined_class_names,
    predict_single,
)
from PIL import Image

model = load_classification_model(
    "checkpoints/classification/resnet50_dogs_cats/last.ckpt",
    num_classes=132,
)
class_names = load_combined_class_names(
    "data/stanford_dogs/images",
    "data/Oxford-IIIT_pet_dataset",
)
image = Image.open("dog.jpg").convert("RGB")

result = predict_single(model, image, class_names=class_names, top_k=5)
print(result)
# {"class_id": 42, "class_name": "Golden_retriever", "confidence": 0.97, "top_k": [...]}
```

Régression BCS seule (ensemble des folds, masque optionnel) :

```python
from bcs_pipeline.inference import load_bcs_model, predict_bcs

bcs = load_bcs_model("checkpoints/bcs_regression")
print(predict_bcs(bcs, "dog.jpg", mask=None))
# {"bcs": 4.82, "category": "Idéal", "std": 0.09, "num_folds": 11, "scale": [1, 9], ...}
```

---

## Evaluation Notebooks

All notebooks are in the `notebooks/` directory.  Use the `bcs_analysis` kernel.

*(See the table in the [intro](#notebooks-dévaluation) for a full description.)*

```bash
jupyter notebook notebooks/
```

---

## Architecture & Module Reference

### `train.py`

Lightweight Hydra-decorated entry-point.  Steps:
1. Validate config → `bcs_pipeline.utils.config_utils.validate_config`
2. Setup experiment dirs → `bcs_pipeline.utils.config_utils.setup_experiment_dirs`
3. Build data module → `bcs_pipeline.data.*DataModule` (Stanford/Oxford × Classification/Segmentation)
4. Build model → `bcs_pipeline.lightning_module.LitClassificationModule` or `LitSegmentationModule`
5. Build trainer (callbacks + loggers) → `bcs_pipeline.trainer_factory.build_trainer`
6. `trainer.fit()` then `trainer.test()`

### `bcs_pipeline.callbacks`

| Function | Purpose |
|---|---|
| `build_checkpoint_callback()` | `ModelCheckpoint` – saves top-k by `val/acc` |
| `build_early_stopping_callback()` | `EarlyStopping` – monitors `val/acc` |
| `build_lr_monitor()` | `LearningRateMonitor` – logs LR at every step |
| `build_callbacks(cfg, dir)` | **Main entry-point** composing the above |

### `bcs_pipeline.loggers`

| Function | Purpose |
|---|---|
| `build_tensorboard_logger()` | TensorBoard event writer |
| `build_wandb_logger()` | Weights & Biases logger (graceful degradation) |
| `build_loggers(cfg, dirs)` | **Main entry-point** |

### `bcs_pipeline.trainer_factory`

| Function | Purpose |
|---|---|
| `build_trainer(cfg, dirs)` | Assembles `pl.Trainer` from config |
| `get_checkpoint_callback(trainer)` | Retrieves the checkpoint callback post-training |

### `bcs_pipeline.inference`

| Function | Purpose |
|---|---|
| `load_model(ckpt, ...)` | Load classification checkpoint → eval mode |
| `load_class_names(data_dir)` | Parse breed names from dataset folders |
| `get_inference_transform(size)` | Deterministic val/inference transforms |
| `predict_single(model, image)` | Predict on one PIL image (top-k) |
| `predict_batch(model, batch)` | Predict on a pre-processed tensor batch |
| `load_segmentation_backend(backend, ckpt)` | Load DeepLabV3 or SAM 2 model |
| `predict_segmentation_with(backend, handle, image, ...)` | Dispatch segmentation to chosen backend |
| `load_sam2_model(ckpt, config)` | Load SAM 2 predictor + auto mask generator |
| `predict_segmentation_sam2(handle, image, mode, ...)` | SAM 2 segmentation (prompted/automatic/pose_prompted) |
| `load_pose_model(ckpt)` | Load YOLOv8 pose model |
| `predict_pose(model, image, ...)` | Run keypoint detection on a PIL image |
| `render_combined(image, classification, segmentation, pose)` | Compose all overlays into one PIL image |
| `save_visualization(image, path)` | Save visualization PNG to disk |
| `run_full_inference(image, ...)` | Combined orchestrator (all 3 pipelines) |

### `bcs_pipeline.datasets`

Centralise les chemins et la logique de découverte des datasets, partagée entre `app.py` et `scripts/preload_db.py`.

| Function | Purpose |
|---|---|
| `collect_all_images()` | Liste toutes les images `(path, dataset, group, ground_truth)` |
| `get_datasets()` | Retourne les groupes et fichiers par dataset (pour l'UI) |
| `resolve_image_path(dataset, group, filename)` | Résout le chemin complet d'une image |
| `ground_truth(dataset, group)` | Retourne le label de vérité terrain |
| `list_image_files(folder)` | Liste les fichiers image dans un dossier |

### `bcs_pipeline.inference_format`

Orchestration partagée du pipeline d'inférence et formatage du résultat.

| Function | Purpose |
|---|---|
| `run_core_inference(cls_model, ..., img)` | Exécute classification + segmentation + pose |
| `format_inference_result(cls, seg, pose, ...)` | Formate les résultats en dict standardisé |

### `bcs_pipeline.db`

Modèles SQLAlchemy + helpers de persistance.

| Function | Purpose |
|---|---|
| `init_db(db_path)` | Initialise la DB (create_all + migrations) |
| `session_scope(session_factory)` | Context manager : rollback on error, close toujours |
| `save_run(session, ...)` | Persiste un run (idempotent via contrainte unique) |
| `save_annotations(session, run_id, ...)` | Sauvegarde les annotations utilisateur |
| `load_run(session, run_id)` | Charge un run complet (JSON + annotations) |
| `list_runs(session, limit, offset, sort_by, sort_order)` | Liste résumé des runs (paginé, triable). `sort_by` est validé contre une allowlist (`id`, `created_at`, `last_inferred_at`, `image_name`, `predicted_class`, `predicted_confidence`, `seg_backend`, `has_annotations`) — défaut `last_inferred_at desc` |
| `delete_run(session, run_id)` | Supprime un run + fichiers associés |

### `bcs_pipeline.lightning_module.LitClassificationModule`

Full-featured `LightningModule` with:
- **Mixup / CutMix** augmentation (configurable via `regularization.*`)
- **Label smoothing** cross-entropy
- **Stochastic depth** (drop-path) in the backbone
- Comprehensive TensorBoard logging: images, confusion matrix, PR curves,
  weight histograms

### `bcs_pipeline.lightning_module.LitSegmentationModule`

Segmentation `LightningModule` (DeepLabV3-ResNet50) with:
- Combined **Cross-Entropy + Dice** loss
- Per-class **IoU, Dice, Pixel Accuracy** metrics
- TensorBoard overlay visualisations

### `bcs_pipeline.data`

| Class | Description |
|---|---|
| `StanfordClassificationDataModule` | Stanford Dogs classification with stratified splits, RandAugment |
| `StanfordSegmentationDataModule` | Stanford Dogs segmentation |
| `OxfordClassificationDataModule` | Oxford-IIIT Pet classification |
| `OxfordSegmentationDataModule` | Oxford-IIIT Pet segmentation with trimap masks |

### `bcs_pipeline.models`

| Class | Description |
|---|---|
| `ResNetTransfer` | ResNet-50 (ImageNet weights) + dropout + optional stochastic depth |
| `ViTTransfer` | HuggingFace `vit-base-patch16-224-in21k` fine-tuning wrapper |

### `bcs_pipeline.utils`

| Module | Description |
|---|---|
| `config_utils` | Experiment dir setup, config validation, config snapshot save |
| `config_validation` | Hydra config schema validation |
| `dataset_stats` | Per-class dataset statistics (compute, display, log, save as JSON) |
| `logging_utils` | Logging setup, Rich config printing, experiment info logging |

---

## Application web

`app.py` fournit une **interface web interactive** (FastAPI + Uvicorn) qui réplique le widget du notebook `combined_inference_overlay.ipynb` directement dans le navigateur. Elle permet d'explorer visuellement les résultats des trois pipelines sur toutes les images des datasets locaux ou sur des images uploadées.

### Lancement rapide

```bash
# Option 1 : via le script de setup (recommandé, gère tout automatiquement)
chmod +x scripts/setup_and_run.sh
./scripts/setup_and_run.sh                    # setup complet + lancement
./scripts/setup_and_run.sh --preload          # idem + pré-chargement de la DB
./scripts/setup_and_run.sh --skip-env         # si .venv/ existe déjà
./scripts/setup_and_run.sh --skip-data        # si les données sont déjà téléchargées
./scripts/setup_and_run.sh --cpu              # forcer CPU (pas de GPU)
./scripts/setup_and_run.sh --no-launch        # setup sans lancer l'app

# Option 2 : manuellement (venv existant)
source .venv/bin/activate
python app.py
# → Ouvrir http://localhost:8000
```

> Les modèles sont chargés en mémoire au premier appel d'inférence (lazy loading). Le premier run est plus lent, les suivants sont instantanés.

### Fonctionnalités

- **Sélection de dataset** : Reddit (out-of-distribution), Stanford Dogs (120 races), Oxford-IIIT Pet (chiens + chats)
- **Upload d'image** : glisser-déposer ou sélection de fichier (JPG, PNG, WebP)
- **Choix du backend de segmentation** : DeepLabV3 (fine-tuned) ou SAM 2 (zero-shot)
- **3 modes SAM 2** : `prompted` (point central), `automatic` (grille dense), `pose_prompted` (bbox + keypoints YOLO)
- **Affichage côte à côte** : image source / overlay interactif avec segmentation + pose + label de classification
- **Édition d'annotations** :
  - Déplacer les keypoints et les coins des bounding boxes par glisser-déposer
  - Éditer le masque de segmentation avec un pinceau (ajouter au foreground) et une gomme (repasser en background)
  - Ajouter des commentaires textuels sur chaque inférence
- **Historique des inférences** : tableau paginé avec chargement, suppression (unitaire ou par lot), et indicateur d'annotations.
  - **Tri par colonne** : cliquer sur l'en-tête d'une colonne (ID, Date, Image, Race prédite, Confiance, Backend, Annotations) pour trier ascendant/descendant. Le tri est résolu côté serveur via une allowlist de colonnes (résistant à l'injection).
  - **Promotion en tête de liste** : la colonne *Date* affiche `last_inferred_at` (et non plus `created_at`), un timestamp re-bumpé à chaque appel idempotent de `save_run`. Re-lancer une inférence sur une image déjà en base la fait remonter en tête sans créer de doublon ni perdre ses annotations.
- **Export** : PNG de l'overlay, JSON des annotations (avec masque édité encodé en base64)
- **Import** : charger un fichier JSON d'annotations pour appliquer les corrections
- **Pré-chargement de la DB** : bouton dans l'interface pour lancer/arrêter le pré-chargement de toutes les images

### Pré-chargement de la base de données

Le pré-chargement peut être fait **depuis l'interface web** (bouton "Pré-charger la DB" avec bouton "Arrêter" pour stopper en cours) ou **en ligne de commande** via `scripts/preload_db.py`.

La DB est stockée dans `data/bcs_app.db` (SQLite). Une contrainte d'unicité sur `(image_name, dataset, group_name, seg_backend)` garantit qu'il n'y a pas de doublons — le pré-chargement est idempotent et peut être arrêté/repris sans perte de données. Re-lancer une inférence sur une image déjà connue ne crée pas de nouvelle ligne : la ligne existante est conservée (avec ses annotations utilisateur), son timestamp `last_inferred_at` est re-bumpé et son JSON sidecar rafraîchi. L'API `/api/preload/status` accepte un paramètre `seg_backend` pour ne compter que les lignes du backend ciblé (utile lorsque plusieurs backends ont été pré-chargés sur le même dataset).

```bash
# CLI : pré-charger avec les paramètres par défaut (DeepLabV3, top-5)
python scripts/preload_db.py

# Forcer le re-traitement d'images déjà en base
python scripts/preload_db.py --force

# Utiliser SAM 2 comme backend de segmentation
python scripts/preload_db.py --seg-backend sam2 --sam2-mode automatic

# Pré-charger un seul dataset
python scripts/preload_db.py --datasets Reddit

# Via setup_and_run.sh (après le setup)
./scripts/setup_and_run.sh --preload
```

| Option | Défaut | Description |
|---|---|---|
| `--seg-backend` | `deeplab` | Backend de segmentation (`deeplab` ou `sam2`) |
| `--sam2-mode` | `prompted` | Mode SAM 2 (`prompted`, `automatic`, `pose_prompted`) |
| `--top-k` | `5` | Nombre de prédictions de classification |
| `--conf-threshold` | `0.25` | Seuil de confiance YOLO |
| `--db-path` | `data/bcs_app.db` | Chemin vers la base SQLite |
| `--force` | — | Re-traiter les images déjà en base |
| `--datasets` | tous | Limiter à certains datasets |

### Points d'accès API

| Route | Méthode | Description |
|---|---|---|
| `GET /` | — | Interface web |
| `GET /api/datasets` | — | Liste les datasets, groupes et nombre d'images |
| `GET /api/images?dataset=&group=` | — | Liste les fichiers d'un groupe |
| `GET /api/thumbnail/<dataset>/<group>/<file>` | — | Sert un thumbnail JPEG (256 px) |
| `POST /api/inference` | JSON body | Lance les 3 pipelines sur une image de dataset |
| `POST /api/inference/upload` | multipart | Lance les 3 pipelines sur une image uploadée |
| `GET /api/history?limit=&offset=&sort=&order=` | — | Liste paginée des runs. `sort` ∈ {`id`, `created_at`, `last_inferred_at`, `image_name`, `predicted_class`, `predicted_confidence`, `seg_backend`, `has_annotations`} (défaut `last_inferred_at`), `order` ∈ {`asc`, `desc`} (défaut `desc`) |
| `GET /api/history/<id>` | — | Détail d'un run (images base64 + annotations) |
| `POST /api/history/<id>/annotations` | JSON body | Sauvegarde les annotations éditées (boxes, keypoints, masque, commentaires) |
| `DELETE /api/history/<id>` | — | Supprime un run et ses fichiers associés |
| `GET /api/preload/status?seg_backend=` | — | État du pré-chargement (progression, compteur DB). Le paramètre optionnel `seg_backend` restreint le compteur au backend donné, pour éviter un faux *complete* lorsque plusieurs backends sont mélangés en base |
| `POST /api/preload/start` | JSON body | Lance le pré-chargement en arrière-plan |
| `POST /api/preload/stop` | — | Arrête le pré-chargement en cours |

Exemple d'appel API programmatique :

```python
import requests

resp = requests.post("http://localhost:8000/api/inference", json={
    "dataset": "Reddit",
    "group": "all",
    "filename": "reddit_dog_1.jpg",
    "seg_backend": "sam2",
    "sam2_mode": "pose_prompted",
})
result = resp.json()
print(result["classification"]["class_name"])   # ex: "golden_retriever"
print(result["pose"]["num_detections"])           # ex: 1
```

### Docker

```bash
docker build -t bcs_determination .
docker run -p 8000:8000 bcs_determination
```

---

## Configuration Reference

All values below can be overridden from the CLI
(`python train.py key=value`).

### Classification (`configs/config.yaml`)

| Key | Type | Default | Description |
|---|---|---|---|
| `seed` | int | 42 | Global random seed |
| `model_name` | str | `vit` | `resnet50` or `vit` |
| `num_classes` | int | 120 | Number of output classes |
| `lr` | float | 0.001 | Learning rate |
| `optimizer_name` | str | `adam` | `adam` or `sgd` |
| `weight_decay` | float | 1e-4 | Weight decay |
| `batch_size` | int | 32 | Mini-batch size |
| `max_epochs` | int | 100 | Max training epochs |
| `patience` | int | 15 | Early-stopping patience |
| `precision` | str | `32` | Training precision (`16-mixed`, `32`, `64`) |
| `image_size` | int | 224 | Input image size |
| `dataset` | str | `stanford` | `stanford` or `oxford` |
| `task` | str | `classification` | `classification` or `segmentation` |
| `val_split` | float | 0.1 | Validation split ratio |
| `test_split` | float | 0.1 | Test split ratio |
| `regularization.dropout` | float | 0.3 | Dropout rate |
| `regularization.label_smoothing` | float | 0.1 | Label smoothing ε |
| `regularization.mixup_alpha` | float | 0.2 | Mixup alpha (0 to disable) |
| `regularization.cutmix_alpha` | float | 1.0 | CutMix alpha (0 to disable) |
| `regularization.stochastic_depth` | float | 0.1 | Drop-path rate |

### Segmentation (`configs/config_segmentation.yaml`)

| Key | Type | Default | Description |
|---|---|---|---|
| `model_name` | str | `deeplabv3_resnet50` | Segmentation backbone |
| `task` | str | `segmentation` | Task type |
| `dataset` | str | `oxford` | Dataset name |
| `seg_num_classes` | int | 3 | Trimap classes (foreground/background/border) |
| `image_size` | int | 256 | Input resolution |
| `batch_size` | int | 16 | Mini-batch size |
| `max_epochs` | int | 50 | Max training epochs |
| `patience` | int | 10 | Early-stopping patience |
