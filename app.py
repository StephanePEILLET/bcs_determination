import io
import os
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

# Import the inference script logic
from inference import predict

app = FastAPI(
    title="Stanford Dogs Inference API",
    description="REST API to predict dog breeds from images using trained PyTorch Lightning models."
)

# Configuration defaults - In real scenarios, these can also be loaded via env vars 
DEFAULT_MODEL_NAME = "vit" # Defaulting to the SOTA model we just added
DEFAULT_DATA_DIR = "" # Optional mapping dataset directory

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Welcome to the Stanford Dogs Inference API! Send a POST request to /predict/"}

@app.post("/predict/")
async def predict_dog(
    file: UploadFile = File(...),
    checkpoint_path: str = Query(..., description="Absolute path to the model checkpoint.ckpt"),
    model_name: str = Query(DEFAULT_MODEL_NAME, description="Architecture name (e.g. resnet50, vit)"),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="The provided file must be an image.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Save purely temporarily so predict function can read it.
        # This acts as an adapter to our existing command-line tailored inference pipeline.
        temp_path = f"/tmp/{file.filename}"
        image.save(temp_path)

        # Run inference wrapper
        result = predict(
            image_path=temp_path,
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            num_classes=120,
            data_dir=DEFAULT_DATA_DIR
        )
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

        if result is None:
            raise HTTPException(status_code=500, detail="Inference returned None. Ensure the checkpoint path is correct.")

        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
