# Use Miniconda as it manages binary dependencies smoothly
FROM continuumio/miniconda3:latest

# Basic system upgrades and cleanup
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# First, copy the environment file and create the Conda environment
COPY environment.yaml .
RUN conda env create -f environment.yaml

# Set shell to ensure `run` commands happen inside the environment
SHELL ["conda", "run", "-n", "dogs_analysis", "/bin/bash", "-c"]

# Copy all the project source code
COPY . /app/

# Expose the API port
EXPOSE 8000

# The Entrypoint directs Docker to launch all CMD instructions via the active Conda environment
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "dogs_analysis"]

# Launch the FastAPI web server wrapper using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
