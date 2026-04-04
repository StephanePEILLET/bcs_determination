#!/usr/bin/env python3
"""
Stanford Dogs Inference Script
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dogs_pipeline.lightning_module.stanford_dogs_module import LitStanfordDogs
from dogs_pipeline.data.stanford_dogs_datamodule import StanfordDogsDataModule

def parse_args():
    parser = argparse.ArgumentParser(description="Predict dog breed from an image.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image to classify")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the trained model checkpoint (.ckpt)")
    parser.add_argument("--model_name", type=str, default="resnet50", help="Model name used during training")
    parser.add_argument("--num_classes", type=int, default=120, help="Number of classes in the dataset")
    parser.add_argument("--data_dir", type=str, default="", help="(Optional) Path to dataset root to parse class names. If not provided, indices are returned.")
    return parser.parse_args()

def load_class_names(data_dir: str):
    images_dir = os.path.join(data_dir, "Images")
    if not os.path.exists(images_dir):
        return None
    
    classes = [d.name for d in os.scandir(images_dir) if d.is_dir()]
    classes.sort()
    # Dog breed names are usually formatted like "n02085620-Chihuahua", clean it to get string
    clean_classes = ["-".join(c.split("-")[1:]) for c in classes]
    return clean_classes

def predict(image_path: str, checkpoint_path: str, model_name: str, num_classes: int, data_dir: str = ""):
    # Set up device for execution
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model from {checkpoint_path}...")
    try:
        model = LitStanfordDogs.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            num_classes=num_classes,
            map_location=device
        )
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    # Load datamodule to get the expected validation transforms
    # We initialize it with dummy paths if a proper data_dir isn't provided just to fetch transforms
    datamodule = StanfordDogsDataModule(data_dir=data_dir if data_dir else "/tmp/dummy", image_size=224)
    transform = datamodule.val_transforms

    # Extract class names if a valid data_dir is provided
    class_names = load_class_names(data_dir) if data_dir else None

    # Load and process image
    print(f"Processing image {image_path}...")
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Failed to load image: {e}")
        return

    # Apply inference transformations and reshape to (1, C, H, W)
    x = transform(image).unsqueeze(0).to(device)

    # Make the Prediction
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        top_prob, top_class = torch.max(probs, dim=1)
        
        top_prob = top_prob.item()
        top_class = top_class.item()

    if class_names and top_class < len(class_names):
        pred_label = class_names[top_class]
        print(f"\nPrediction: {pred_label} (Class ID: {top_class})")
    else:
        print(f"\nPrediction: Class ID {top_class}")
        
    print(f"Confidence: {top_prob*100:.2f}%\n")
    
    return {
        "class_id": top_class,
        "confidence": top_prob,
        "class_name": class_names[top_class] if class_names else None
    }

if __name__ == "__main__":
    args = parse_args()
    predict(
        image_path=args.image_path,
        checkpoint_path=args.checkpoint_path,
        model_name=args.model_name,
        num_classes=args.num_classes,
        data_dir=args.data_dir
    )
