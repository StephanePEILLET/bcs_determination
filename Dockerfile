# BCS Determination — container image.
#
# Build:
#   docker build -t bcs-pipeline .                          # base (DeepLab + SAM 2)
#   docker build -t bcs-pipeline --build-arg WITH_SAM3=1 .  # base + Meta SAM 3
#
# Run (with HF cache mount for SAM 3 — gated checkpoint):
#   docker run --gpus all -p 5000:5000 \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     -v $(pwd)/checkpoints:/app/checkpoints \
#     -v $(pwd)/data:/app/data \
#     bcs-pipeline
#
# `hf auth login` must have been run on the host beforehand (the token lives
# in ~/.cache/huggingface/token, mounted into the container).
FROM continuumio/miniconda3:latest

ARG WITH_SAM3=0

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

# Install the project itself (and optionally the SAM 3 extra) before copying
# the rest so that pyproject changes don't bust the source-code cache layer.
COPY pyproject.toml ./
RUN if [ "$WITH_SAM3" = "1" ]; then \
        pip install --no-cache-dir -e ".[dev,sam3]"; \
    else \
        pip install --no-cache-dir -e ".[dev]"; \
    fi

COPY . /app/

EXPOSE 5000

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "bcs_analysis"]
CMD ["python", "app.py"]
