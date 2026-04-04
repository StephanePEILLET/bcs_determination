import io
import logging
import os
import tempfile

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

# Import the inference script logic
from inference import predict

logger = logging.getLogger("bcs_pipeline.api")

app = FastAPI(
    title="Stanford Bcs Inference API",
    description="REST API to predict dog breeds from images using trained PyTorch Lightning models."
)

# Configuration defaults - In real scenarios, these can also be loaded via env vars 
DEFAULT_MODEL_NAME = "resnet50"
DEFAULT_DATA_DIR = "" # Optional mapping dataset directory

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Welcome to the Stanford Bcs Inference API! Send a POST request to /predict/"}

@app.post("/predict/")
async def predict_dog(
    file: UploadFile = File(...),
    checkpoint_path: str = Query(..., description="Absolute path to the model checkpoint.ckpt"),
    model_name: str = Query(DEFAULT_MODEL_NAME, description="Architecture name (e.g. resnet50, vit)"),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="The provided file must be an image.")

    temp_path = None
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Save to a secure temporary file to avoid path traversal and race conditions
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir="/tmp") as tmp:
            image.save(tmp, format="PNG")
            temp_path = tmp.name

        # Run inference wrapper
        result = predict(
            image_path=temp_path,
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            num_classes=120,
            data_dir=DEFAULT_DATA_DIR
        )

        if result is None:
            raise HTTPException(status_code=500, detail="Inference returned None. Ensure the checkpoint path is correct.")

        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal server error during inference.")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
