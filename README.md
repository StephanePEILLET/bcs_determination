# Body Condition Score (BCS) Determination

> **Estimation automatique du Body Condition Score (BCS) d'un chien à partir d'une image, en combinant classification de race, segmentation sémantique et détection de pose.**

## Contexte

Le **Body Condition Score** est un indicateur clinique (échelle 1–9) utilisé par les vétérinaires pour évaluer l'état corporel d'un animal (embonpoint, idéal, maigreur). Cette estimation repose today'hui sur une palpation manuelle et une évaluation visuelle subjective. Ce projet explore une approche **entièrement visuelle et automatisée** basée sur le deep learning.

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

Deux backends interchangeables :

- **DeepLabV3-ResNet50** (COCO pretrained, fine-tuned) — val_IoU = 0.82, 3 classes trimap (*foreground*, *background*, *border*).
  Métriques : Pixel Accuracy = 93.7%, mIoU = 0.82, mDice = 0.89.
- **SAM 2** (zero-shot, 3 modes : `prompted`, `automatic`, `pose_prompted`) — masque à **2 classes** (*foreground*, *background*) directement issu du masque binaire SAM, sans bordure dérivée (silhouette déjà nette).

Le masque est **éditable côté navigateur** (pinceau pour ajouter au foreground, gomme pour repasser au background) et le résultat est persisté en base avec les autres annotations.

### 3. Détection de pose (OpenPose)

Repérer les **points clés anatomiques** (colonne vertébrale, hanches, côtes, queue) permet de mesurer des rapports de proportions corporelles utilisés dans les grilles BCS vétérinaires (ex. visibilité des côtes, présence d'une taille vue de dessus, épaisseur de la base de la queue).

- Protocole OpenPose (COCO + MPII) via les modèles pré-entraînés
- Extraction de keypoints et calcul de features géométriques

### 4. Vision classique (baseline sans deep learning)

Des méthodes de **détection de contours** traditionnelles (Canny, Sobel, Laplacien, Prewitt) servent de baseline pour comparer l'apport du deep learning sur l'extraction de silhouette :

- OpenCV : Canny multi-seuils, Sobel, Laplacien
- scikit-image : Canny multi-scale, Prewitt, Roberts, Scharr
- Kornia (GPU, différentiable) : Sobel, Canny, Laplacian

## Stack technique

| Composant | Technologie |
|---|---|
| Framework DL | PyTorch + PyTorch Lightning |
| Hyperparamètres | Hydra + Optuna |
| Logging | TensorBoard (+ W&B optionnel) |
| Modèles | torchvision (ResNet, DeepLabV3), HuggingFace Transformers (ViT), SAM 2, YOLOv8 |
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
5. [Application web Flask](#application-web-flask)
6. [Evaluation Notebooks](#evaluation-notebooks)
7. [Architecture & Module Reference](#architecture--module-reference)
8. [Configuration Reference](#configuration-reference)

## Project Structure

```
bcs_determination/
├── configs/
│   ├── config.yaml                        # Classification Hydra config (Stanford Dogs, 120 races)
│   ├── config_dogs_cats.yaml              # Classification combinée (132 races, chiens + chats)
│   └── config_segmentation.yaml           # Segmentation Hydra config
├── src/
│   └── bcs_pipeline/                      # Main Python package
│       ├── __init__.py
│       ├── callbacks.py                   # Callback factories
│       ├── loggers.py                     # Logger factories (TensorBoard, W&B)
│       ├── trainer_factory.py             # High-level Trainer builder
│       ├── db.py                          # SQLAlchemy models (runs + annotations + masques édités)
│       ├── inference/                     # Shared inference utilities
│       │   ├── __init__.py                # Public API re-exports
│       │   ├── classification.py          # ResNet/ViT breed prediction (+ load_combined_class_names)
│       │   ├── segmentation.py            # DeepLabV3 trimap + backend dispatch
│       │   ├── segmentation_sam2.py       # SAM 2 zero-shot segmentation (2 classes)
│       │   ├── pose.py                    # YOLOv8 keypoint detection
│       │   ├── visualization.py           # PIL overlay rendering
│       │   └── pipeline.py                # Combined orchestrator
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
│           └── logging_utils.py
├── templates/
│   └── index.html                         # FastAPI UI (single-page) — édition annotations + masque
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
│   ├── Reddit_example/                    # Out-of-distribution webp images (téléchargées automatiquement)
│   │   ├── reddit_dog_1.jpg               #   Image Reddit #1
│   │   └── reddit_dog_2.jpg               #   Image Reddit #2
│   ├── Dog_Pose_Estimations/              # YOLOv8 pose training data
│   ├── outputs/                           # JSON + PNG de chaque run sauvé (run_<id>.json, run_<id>_mask.png)
│   └── bcs_app.db                         # SQLite — historique des runs et annotations utilisateur
├── experiments/                           # Auto-generated (checkpoints, TB logs, splits)
│   ├── resnet50_dogs_cats/                # Classifieur 132 races (chiens + chats) — modèle utilisé par l'app
│   ├── resnet50_adam_cosine_annealing/    # Legacy — ancien classifieur 120 races (chiens seuls)
│   ├── vit_adam_cosine_annealing/
│   └── deeplabv3_resnet50_adam_cosine_annealing/
├── checkpoints/                           # SAM 2 foundation model weights
├── runs/pose/                             # YOLOv8 pose training outputs
├── train.py                               # Training entry-point
├── inference.py                           # CLI inference
├── app.py                                 # FastAPI/Uvicorn web app (UI interactive)
├── setup_and_run.sh                       # Bootstrap : uv venv + données + checkpoints + lancement
├── scripts/
│   ├── setup_and_run.sh                   # (idem, dans scripts/)
│   └── preload_db.py                      # Pré-chargement DB : inférences sur toutes les images
├── environment.yaml
├── Dockerfile
└── README.md
```

### Design Principles

| Principle | How it's applied |
|---|---|
| **Modularity** | Each concern (callbacks, loggers, trainer, inference) is in its own module. |
| **Reusability** | `bcs_pipeline.inference` is shared by `inference.py`, `app.py`, and notebooks. |
| **Lightweight entry-points** | `train.py`, `inference.py`, and `app.py` contain only orchestration — no business logic. |
| **Configuration-driven** | All hyperparameters live in `configs/config.yaml` and can be overridden via CLI. |

---

## Setup & Installation

### Option A — Bootstrap automatique (recommandé)

Le script `setup_and_run.sh` installe `uv` (gestionnaire d'environnement
Python rapide), crée le venv `.venv/`, télécharge les datasets (Stanford
Dogs, Oxford-IIIT Pet, images Reddit d'exemple) et le checkpoint SAM 2, puis
lance l'application :

```bash
chmod +x setup_and_run.sh
./setup_and_run.sh                  # tout faire
./setup_and_run.sh --skip-data      # passer le téléchargement des données
./setup_and_run.sh --skip-env       # réutiliser .venv/ existant
./setup_and_run.sh --no-launch      # arrêter avant de lancer l'app
./setup_and_run.sh --preload        # pré-charger la DB (inférences pré-calculées)
./setup_and_run.sh --cpu            # forcer l'exécution sur CPU
```

### Option B — Conda manuel

```bash
conda env create -f environment.yaml
conda activate bcs_analysis
```

> **Note:** L'`environment.yaml` inclut de lourdes dépendances `pip` que
> Conda installe silencieusement — l'installation peut sembler figée
> 5–10 min. Ne pas interrompre (`Ctrl+C`).

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

## Application web Flask

`app.py` fournit une **interface web interactive** qui réplique le widget du notebook `combined_inference_overlay.ipynb` directement dans le navigateur. Elle permet d'explorer visuellement les résultats des trois pipelines sur toutes les images des datasets locaux.

### Fonctionnalités

- **Sélection de dataset** : Reddit (out-of-distribution), Stanford Dogs (120 races), Oxford-IIIT Pet (chiens + chats)
- **Sélection d'image** : navigation par race/groupe puis par fichier
- **Choix du backend de segmentation** : DeepLabV3 (fine-tuned) ou SAM 2 (zero-shot)
- **3 modes SAM 2** : `prompted` (point central), `automatic` (grille dense), `pose_prompted` (bbox + keypoints YOLO)
- **Affichage côte à côte** : image source / overlay avec segmentation + pose + label de classification
- **Observations textuelles** : top-5 races, distribution des classes de segmentation, nombre de détections de pose

### Lancement

```bash
conda activate bcs_analysis
python app.py
```

Ouvrir **http://localhost:5000** dans un navigateur.

> Les modèles sont chargés en mémoire au premier appel d'inférence (lazy loading). Le premier run est plus lent, les suivants sont instantanés.

### Pré-chargement de la base de données

Le script `scripts/preload_db.py` permet de **pré-calculer toutes les inférences** pour les images des datasets (Stanford Dogs, Oxford-IIIT Pet, Reddit) et de les stocker en base SQLite. Ainsi, l'interface web n'a pas à recalculer les résultats à la demande.

```bash
# Pré-charger avec les paramètres par défaut (DeepLabV3, top-5)
python scripts/preload_db.py

# Forcer le re-traitement d'images déjà en base
python scripts/preload_db.py --force

# Utiliser SAM 2 comme backend de segmentation
python scripts/preload_db.py --seg-backend sam2 --sam2-mode automatic

# Pré-charger un seul dataset
python scripts/preload_db.py --datasets Reddit

# Via setup_and_run.sh (après le setup)
./setup_and_run.sh --preload
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

Le script est **idempotent** : les images déjà présentes en base sont ignorées (sauf `--force`). Chaque résultat inclut l'origine (`source_type="dataset"`, `dataset`, `group_name`) pour distinguer les images de dataset des uploads utilisateur.

### Points d'accès API

| Route | Méthode | Description |
|---|---|---|
| `GET /` | — | Interface web |
| `GET /api/datasets` | — | Liste les datasets, groupes et nombre d'images |
| `GET /api/images?dataset=&group=` | — | Liste les fichiers d'un groupe |
| `GET /api/thumbnail/<dataset>/<group>/<file>` | — | Sert un thumbnail JPEG (256 px) |
| `POST /api/inference` | JSON body | Lance les 3 pipelines, retourne les images base64 + observations |

Exemple d'appel API programmatique :

```python
import requests

resp = requests.post("http://localhost:5000/api/inference", json={
    "dataset": "Reddit",
    "group": "all",
    "filename": "is-my-dog-overweight-v0-am4q7ltvecng1.webp",
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
docker run -p 5000:5000 bcs_determination
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
