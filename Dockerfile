# BCS Determination — container image.
#
# Build:
#   docker build -t bcs-pipeline .   # DeepLab + SAM 2 + Meta SAM 3 (SAM 3 in default deps)
#
# Run (with HF cache mount for SAM 3 — gated checkpoint):
#   docker run --gpus all -p 8000:8000 \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     -v $(pwd)/checkpoints:/app/checkpoints \
#     -v $(pwd)/data:/app/data \
#     bcs-pipeline
#
# `hf auth login` must have been run on the host beforehand (the token lives
# in ~/.cache/huggingface/token, mounted into the container).
FROM continuumio/miniconda3:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Provision Python 3.12 + PyTorch with CUDA from the conda env file first
# (cached as long as environment.yaml is unchanged).
COPY environment.yaml .
RUN conda env create -f environment.yaml

SHELL ["conda", "run", "-n", "bcs_analysis", "/bin/bash", "-c"]

# Install the project itself before copying the rest so that pyproject
# changes don't bust the source-code cache layer.
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]"

COPY . /app/

EXPOSE 8000

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "bcs_analysis"]
CMD ["python", "app.py"]
