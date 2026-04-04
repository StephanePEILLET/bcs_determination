# BCS Determination Classification

This repository contains a PyTorch Lightning based pipeline for classifying the 120 dog breeds from the [Stanford Bcs Dataset](http://vision.stanford.edu/aditya86/ImageNetBcs/).

## Project Structure

```
bcs_analysis/
├── configs/              # Hydra configuration files
├── src/
│   └── bcs_pipeline/    # Python package for models, datamodules, logging, etc.
├── experiments/          # Directory where Hydra saves experiment logs/checkpoints
├── app.py                # FastAPI endpoints
├── environment.yaml      # Conda environment file
├── train.py              # Main training script
├── inference.py          # Script for making predictions on new images
├── optuna_multi_gpu_tutorial.ipynb # Tutorial for multi-GPU hyperparameter sweep
├── visualize_results.ipynb # Jupyter Notebook for visual inference tests
└── README.md             # This file
```

## Setup & Installation

1. Create the Conda environment using the provided YAML file:
   ```bash
   conda env create -f environment.yaml
   conda activate bcs_analysis
   ```
   > **Note:** The `environment.yaml` includes heavy `pip` dependencies. Conda installs these silently, so the installation may appear frozen for 5-10 minutes. Please be patient and do not interrupt (`Ctrl+C`) the process, otherwise it will fail with `critical libmamba pip failed to install packages`.

   *(Ensure you have NVIDIA drivers and CUDA configured correctly if training on a GPU).*

2. Data Preparation:
   Download the Stanford Bcs Dataset and extract it to your system. Note the path containing the `Images/` folder, for example, `/path/to/stanford_bcs/images`.

## Training

Parameters and configurations are managed via [Hydra](https://hydra.cc/). You can modify hyperparameters directly in `configs/config.yaml` or override them from the terminal during execution.

```bash
# Basic run with default config (defined in configs/config.yaml)
python train.py

# Override hyperparameters specific to your environment explicitly
python train.py data_dir=/path/to/stanford_bcs/images batch_size=32 max_epochs=50
```

By default:
- Experiment artifacts (logs, checkpoints, tensorboard metrics) will be exported to `experiments/bcs_determination/<date>_<time>`.
- Models are tracked by their validation accuracy.

## Inference

Once you have trained the model and obtained a PyTorch Lightning checkpoint (`.ckpt` file), you can seamlessly make predictions on new dog images.

```bash
python inference.py \
  --image_path sample_dog.jpg \
  --checkpoint_path experiments/bcs_determination/.../checkpoints/epoch=...ckpt \
  --data_dir /path/to/stanford_bcs/images
```

*(Providing `--data_dir` is completely optional, but recommended as it gives the script access to map numerical class IDs back directly to readable dog breed names).*

## Visualization

A Jupyter Notebook, `visualize_results.ipynb`, is provided to intuitively review the model's performance on sample images interactively.

To interact with it, start the Jupyter server:
```bash
jupyter notebook visualize_results.ipynb
```
From within the notebook, configure `CHECKPOINT_PATH` and `DATA_DIR` near the top, and advance through the cells to instantly obtain dog breed probabilities superimposed on the actual imagery.

## Architecture & Code Documentation

This project isolates configurations, data loading, and model architectures. Below is the technical breakdown of core functions:

### 1. `train.py`
The Hydra-decorated central training loop.
- **`train(cfg: DictConfig) -> float`**: Sets up early stopping, TensorBoard logging, and triggers `trainer.fit()`. For hyperparameter tuning, it resolves and returns `checkpoint_callback.best_model_score` (`val_acc`), feeding the metric back to the Optuna sweeps.

### 2. PyTorch Lightning Backend (`src/bcs_pipeline/`)
- **`LitBcsDetermination` (`lightning_module/bcs_determination_module.py`)**: 
  - Subclasses `LightningModule`. Abstract class interacting with either standard ResNets or advanced ViTs (`model_name="vit"`).
  - *`validation_step`*: Orchestrates custom TensorBoard tracking by rendering a `make_grid` output of the batches for deep visual inspection.
  - *`configure_optimizers`*: Retrieves dictionary configurations from Hydra to load advanced schedulers (like `CosineAnnealingLR`).
- **`StanfordBcsDataModule`**:
  - *`setup`*: Performs programmatic cross-validation memory splitting. Ensures robust State-of-the-Art data augmentation by pipelining images through `RandAugment`.

### 3. `inference.py`
- **`predict(image_path, checkpoint_path, model_name, ...)`**: A functional wrapper prioritizing safe operational environments. It forces `model.eval()`, handles PIL image bytes seamlessly through dataset metric normalization via the stored validation transforms, and provides raw probabilities mapped natively to human-readable strings if `data_dir` is supplied.

## Deployment

The project is structured to be deployed easily as a microservice using FastAPI and Docker.

### Local API
You can run the inference server locally using Uvicorn:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
Navigate to `http://localhost:8000/docs` to view the interactive swagger documentation and test the `/predict/` endpoint natively by uploading an image.

### Docker (Production)
A `Dockerfile` is provided which automatically sets up the complete Conda environment securely and launches the API.
```bash
docker build -t bcs_determination_api .
docker run -p 8000:8000 bcs_determination_api
```
