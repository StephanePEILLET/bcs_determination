# Body Condition Score (BCS) Determination

> **Estimation automatique du Body Condition Score (BCS) d'un chien à partir d'une image, en combinant classification de race, segmentation sémantique et détection de pose.**

## Contexte

Le **Body Condition Score** est un indicateur clinique (échelle 1–9) utilisé par les vétérinaires pour évaluer l'état corporel d'un animal (embonpoint, idéal, maigreur). Cette estimation repose today'hui sur une palpation manuelle et une évaluation visuelle subjective. Ce projet explore une approche **entièrement visuelle et automatisée** basée sur le deep learning.

## Approche

Le BCS d'un chien dépend fortement de sa **race** (un Greyhound et un Bulldog n'ont pas la même morphologie), de la **forme de son silhouette** (répartition graisse/muscle visible depuis le dessus et le côté), et de sa **posture**. Le pipeline combine trois piliers complémentaires :

### 1. Classification de race (120 races — Stanford Dogs)

Identifier la race permet d'ajuster les attentes morphologiques : un Whippet naturellement mince ne doit pas être jugé comme un Labrador en sous-poids.

- **ResNet-50** (ImageNet pretrained, fine-tuned) — val_acc = 0.79
- **ViT-B/16** (google/vit-base-patch16-224-in21k, fine-tuned) — val_acc = 0.87

| Modèle | Backbone | Top-1 | Top-5 | Paramètres |
|---|---|---|---|---|
| ResNet-50 | CNN (23M) | 79% | — | 23,7M |
| ViT-B/16 | Transformer (86M) | 87% | — | 86M |

### 2. Segmentation sémantique (Oxford-IIIT Pet)

Isoler le contour du chien dans l'image permet d'analyser sa **silhouette** : un chien en surpoids présentera un contour plus large au niveau des côtes et de la taille, sans taille marquée vue de dessus.

- **DeepLabV3-ResNet50** (COCO pretrained, fine-tuned) — val_IoU = 0.82
- 3 classes trimap : *foreground (animal)*, *background*, *border (contour)*
- Métriques : Pixel Accuracy = 93.7%, mIoU = 0.82, mDice = 0.89

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
| Modèles | torchvision (ResNet, DeepLabV3), HuggingFace Transformers (ViT) |
| Serveur d'inférence | FastAPI |
| Container | Docker |

## Notebooks d'évaluation

| Notebook | Description |
|---|---|
| `reddit_images_tests_models.ipynb` | **Vue d'ensemble** : les 3 modèles sur images Reddit + contours classiques |
| `evaluate_vit_classification_Stanford_Dogs.ipynb` | Évaluation complète ViT (courbes, matrice de confusion, confiance, Reddit) |
| `visualize_results_Race_Classif_Stanford_Dogs.ipynb` | Évaluation ResNet-50 (t-SNE, calibration, rapports par classe) |
| `evaluate_segmentation_oxford_pet.ipynb` | Évaluation DeepLabV3 (IoU/Dice, overlays, Reddit) |
| `sam2_comparison.ipynb` | Comparaison SAM2 vs DeepLabV3 pour la segmentation |
| `edge_detection.ipynb` | Comparaison exhaustive de contours (skimage, Kornia, OpenCV) |
| `unsupervised_segmentation.ipynb` | Baselines de segmentation non supervisée |
| `pose_detection.ipynb` | Détection de pose avec OpenPose |

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Setup & Installation](#setup--installation)
3. [Training](#training)
4. [Inference](#inference)
5. [Evaluation Notebooks](#evaluation-notebooks)
6. [Architecture & Module Reference](#architecture--module-reference)
7. [Deployment](#deployment)
8. [Configuration Reference](#configuration-reference)

## Project Structure

```
bcs_determination/
├── configs/
│   ├── config.yaml                        # Classification Hydra config
│   └── config_segmentation.yaml           # Segmentation Hydra config
├── src/
│   └── bcs_pipeline/                      # Main Python package
│       ├── __init__.py
│       ├── callbacks.py                   # Callback factories
│       ├── loggers.py                     # Logger factories (TensorBoard, W&B)
│       ├── trainer_factory.py             # High-level Trainer builder
│       ├── inference/                     # Shared inference utilities
│       │   └── __init__.py
│       ├── data/
│       │   ├── stanford_classification_datamodule.py
│       │   ├── stanford_segmentation_datamodule.py
│       │   ├── oxford_classification_datamodule.py
│       │   └── oxford_segmentation_datamodule.py
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
├── notebooks/
│   ├── evaluate_vit_classification_Stanford_Dogs.ipynb
│   ├── visualize_results_Race_Classif_Stanford_Dogs.ipynb
│   ├── evaluate_segmentation_oxford_pet.ipynb
│   ├── reddit_images_tests_models.ipynb
│   ├── sam2_comparison.ipynb
│   ├── edge_detection.ipynb
│   ├── unsupervised_segmentation.ipynb
│   └── pose_detection.ipynb
├── data/
│   ├── Stanford_dogs/                     # 120 breed folders
│   ├── Oxford-IIIT_pet_dataset/           # Pet images + trimaps
│   └── Reddit_example/                    # 2 out-of-distribution webp images
├── experiments/                           # Auto-generated (checkpoints, TB logs, splits)
│   ├── resnet50_adam_cosine_annealing/
│   ├── vit_adam_cosine_annealing/
│   └── deeplabv3_resnet50_adam_cosine_annealing/
├── models/pose/                           # OpenPose prototxt
├── train.py                               # Training entry-point
├── inference.py                           # CLI inference
├── app.py                                 # FastAPI server
├── environment.yaml
├── Dockerfile
└── README.md
```

### Design Principles

| Principle | How it's applied |
|---|---|
| **Modularity** | Each concern (callbacks, loggers, trainer, inference) is in its own module. |
| **Reusability** | `bcs_pipeline.inference` is shared by `inference.py`, `app.py`, and notebooks. |
| **Lightweight entry-points** | `train.py` and `inference.py` contain only orchestration — no business logic. |
| **Configuration-driven** | All hyperparameters live in `configs/config.yaml` and can be overridden via CLI. |

---

## Setup & Installation

### 1. Create the Conda environment

```bash
conda env create -f environment.yaml
conda activate bcs_analysis
```

> **Note:** The `environment.yaml` includes heavy `pip` dependencies. Conda
> installs these silently, so the installation may appear frozen for 5-10
> minutes. Please be patient and **do not** interrupt (`Ctrl+C`) the process.

### 2. Verify the install

```bash
python -c "import pytorch_lightning; print(pytorch_lightning.__version__)"
python -c "from torch.utils.tensorboard import SummaryWriter; print('TensorBoard OK')"
```

### 3. Data preparation

**Classification (Stanford Dogs)**

Download the Stanford Dogs Dataset and extract it so the directory tree looks
like:

```
data/Stanford_dogs/
└── Images/
    ├── n02085620-Chihuahua/
    ├── n02085782-Japanese_spaniel/
    └── ...  (120 breed folders)
```

> **Tip:** If you point `data_dir` to the correct location the
> `StanfordClassificationDataModule` will auto-download and extract the dataset.

**Segmentation (Oxford-IIIT Pet)**

```
data/Oxford-IIIT_pet_dataset/
├── images/
│   ├── Abyssinian_1.jpg
│   └── ...
└── annotations/
    └── trimaps/
        ├── Abyssinian_1.png
        └── ...
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
from bcs_pipeline.inference import load_model, load_class_names, predict_single
from PIL import Image

model = load_model("checkpoints/best.ckpt", model_name="resnet50")
class_names = load_class_names("data/stanford_dogs")
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
| `load_model(ckpt, ...)` | Load checkpoint → eval mode |
| `load_class_names(data_dir)` | Parse breed names from dataset folders |
| `get_inference_transform(size)` | Deterministic val/inference transforms |
| `predict_single(model, image)` | Predict on one PIL image (top-k) |
| `predict_batch(model, batch)` | Predict on a pre-processed tensor batch |

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

## Deployment

### Local API (FastAPI)

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Interactive docs at `http://localhost:8000/docs`.

### Docker

```bash
docker build -t bcs_determination_api .
docker run -p 8000:8000 bcs_determination_api
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
