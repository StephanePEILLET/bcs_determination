# BCS Determination Classification

Deep-learning pipeline for classifying **120 dog breeds** from the
[Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
using **PyTorch Lightning**, **Hydra**, **TensorBoard** and optionally
**Weights & Biases**.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Setup & Installation](#setup--installation)
3. [Training](#training)
4. [Inference](#inference)
5. [Visualization](#visualization)
6. [Architecture & Module Reference](#architecture--module-reference)
7. [Deployment](#deployment)
8. [Configuration Reference](#configuration-reference)

---

## Project Structure

```
bcs_determination/
├── configs/
│   └── config.yaml               # Hydra configuration (hyperparams, paths, …)
├── src/
│   └── bcs_pipeline/             # Main Python package
│       ├── __init__.py
│       ├── callbacks.py          # Callback factories (checkpoint, early stop, LR)
│       ├── loggers.py            # Logger factories (TensorBoard, W&B)
│       ├── trainer_factory.py    # High-level Trainer builder
│       ├── inference/            # Shared inference utilities
│       │   └── __init__.py       #   load_model, predict_single, predict_batch
│       ├── data/
│       │   └── stanford_bcs_datamodule.py   # LightningDataModule
│       ├── lightning_module/
│       │   └── bcs_determination_module.py  # LightningModule (training logic)
│       ├── models/
│       │   ├── resnet_transfer.py           # ResNet-50 transfer backbone
│       │   └── vit_transfer.py              # ViT transfer backbone
│       └── utils/
│           ├── config_utils.py              # Experiment dirs, config helpers
│           ├── config_validation.py         # Pydantic-based validation
│           └── logging_utils.py             # Rich logging setup
├── experiments/              # Auto-generated experiment outputs
├── train.py                  # Training entry-point (lightweight orchestrator)
├── inference.py              # CLI for single-image prediction
├── app.py                    # FastAPI server
├── environment.yaml          # Conda environment spec
├── Dockerfile                # Production container
├── visualize_results.ipynb   # Visual inference notebook
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

Download the Stanford Dogs Dataset and extract it so the directory tree looks
like:

```
data/stanford_dogs/
└── Images/
    ├── n02085620-Chihuahua/
    ├── n02085782-Japanese_spaniel/
    └── ...  (120 breed folders)
```

> **Tip:** If you point `data_dir` to the correct location the
> `StanfordBcsDataModule` will auto-download and extract the dataset for you.

---

## Training

Training is managed via [Hydra](https://hydra.cc/).  You can modify
`configs/config.yaml` **or** override values from the CLI.

```bash
# Default config
python train.py

# Override hyperparameters
python train.py data_dir=/data/stanford_dogs batch_size=64 max_epochs=50

# Use GPU with mixed precision
python train.py trainer.accelerator=gpu precision=16-mixed

# Resume from a checkpoint
python train.py trainer.resume_from_checkpoint=experiments/.../checkpoints/last.ckpt
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

## Visualization

A Jupyter Notebook `visualize_results.ipynb` lets you visually inspect
predictions on sample images.

```bash
jupyter notebook visualize_results.ipynb
```

Set `CHECKPOINT_PATH` and `DATA_DIR` at the top of the notebook, then run all
cells to see predicted breeds overlaid on the actual images.

---

## Architecture & Module Reference

### `train.py`

Lightweight Hydra-decorated entry-point.  Steps:
1. Validate config → `bcs_pipeline.utils.config_utils.validate_config`
2. Setup experiment dirs → `bcs_pipeline.utils.config_utils.setup_experiment_dirs`
3. Build data module → `bcs_pipeline.data.StanfordBcsDataModule`
4. Build model → `bcs_pipeline.lightning_module.LitBcsDetermination`
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

### `bcs_pipeline.lightning_module.LitBcsDetermination`

Full-featured `LightningModule` with:
- **Mixup / CutMix** augmentation (configurable via `regularization.*`)
- **Label smoothing** cross-entropy
- **Stochastic depth** (drop-path) in the backbone
- Comprehensive TensorBoard logging: images, confusion matrix, PR curves,
  weight histograms

### `bcs_pipeline.data.StanfordBcsDataModule`

- Auto-downloads the Stanford Dogs tar archive
- Applies `RandAugment` + ImageNet normalisation for training
- Deterministic resize/crop for validation and test splits
- Reproducible `random_split()` seeded by `cfg.seed`

### `bcs_pipeline.models`

| Class | Description |
|---|---|
| `ResNetTransfer` | ResNet-50 (ImageNet weights) + dropout + optional stochastic depth |
| `ViTTransfer` | HuggingFace `vit-base-patch16-224-in21k` fine-tuning wrapper |

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

| Key | Type | Default | Description |
|---|---|---|---|
| `seed` | int | 42 | Global random seed |
| `model_name` | str | `resnet50` | `resnet50` or `vit` |
| `num_classes` | int | 120 | Number of output classes |
| `lr` | float | 0.001 | Learning rate |
| `optimizer_name` | str | `adam` | `adam` or `sgd` |
| `weight_decay` | float | 1e-4 | Weight decay |
| `batch_size` | int | 32 | Mini-batch size |
| `max_epochs` | int | 20 | Max training epochs |
| `patience` | int | 5 | Early-stopping patience |
| `precision` | str | `16-mixed` | Training precision (`16-mixed`, `32`, `64`) |
| `use_tensorboard` | bool | true | Enable TensorBoard |
| `use_wandb` | bool | false | Enable W&B |
| `trainer.accelerator` | str | `auto` | `auto`, `cpu`, `gpu`, `tpu` |
| `trainer.devices` | str/int | `auto` | Number of devices |
| `regularization.dropout` | float | 0.3 | Dropout rate |
| `regularization.label_smoothing` | float | 0.1 | Label smoothing ε |
| `regularization.mixup_alpha` | float | 0.2 | Mixup alpha (0 to disable) |
| `regularization.cutmix_alpha` | float | 1.0 | CutMix alpha (0 to disable) |
| `regularization.stochastic_depth` | float | 0.1 | Drop-path rate |
