"""
BCS Determination – Inference CLI.

Thin command-line wrapper around :mod:`bcs_pipeline.inference`.  All
heavy-lifting (model loading, transforms, prediction) lives in the shared
module so that ``app.py`` and notebooks can reuse the same logic without
duplicating code.

Usage
-----
.. code-block:: bash

    python inference.py \\
        --image_path sample_dog.jpg \\
        --checkpoint_path experiments/.../checkpoints/best.ckpt \\
        --data_dir data/stanford_dogs
"""

import sys
import argparse
from pathlib import Path

from PIL import Image

# ── Make the ``src/`` package importable ─────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))

from bcs_pipeline.inference import (
    load_model,
    load_class_names,
    predict_single,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Predict dog breed from an image using a trained BCS model.",
    )
    parser.add_argument(
        "--image_path", type=str, required=True,
        help="Path to the image to classify.",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Path to the trained model checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--model_name", type=str, default="resnet50",
        help="Model architecture used during training (default: resnet50).",
    )
    parser.add_argument(
        "--num_classes", type=int, default=120,
        help="Number of classes (default: 120).",
    )
    parser.add_argument(
        "--image_size", type=int, default=224,
        help="Image resolution expected by the model (default: 224).",
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Number of top predictions to display (default: 5).",
    )
    parser.add_argument(
        "--data_dir", type=str, default="",
        help="(Optional) Dataset root to resolve human-readable class names.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry-point for the CLI."""
    args = parse_args()

    # ── Load model ───────────────────────────────────────────────────
    model = load_model(
        checkpoint_path=args.checkpoint_path,
        model_name=args.model_name,
        num_classes=args.num_classes,
    )

    # ── Resolve class names ──────────────────────────────────────────
    class_names = load_class_names(args.data_dir) if args.data_dir else None

    # ── Load image ───────────────────────────────────────────────────
    print(f"Processing image {args.image_path}…")
    image = Image.open(args.image_path).convert("RGB")

    # ── Predict ──────────────────────────────────────────────────────
    result = predict_single(
        model,
        image,
        image_size=args.image_size,
        class_names=class_names,
        top_k=args.top_k,
    )

    # ── Display results ──────────────────────────────────────────────
    label = result["class_name"] or f"Class ID {result['class_id']}"
    print(f"\n{'─' * 40}")
    print(f"  Prediction : {label}")
    print(f"  Confidence : {result['confidence'] * 100:.2f}%")
    print(f"{'─' * 40}")

    if result["top_k"]:
        print(f"\n  Top-{args.top_k} predictions:")
        for i, entry in enumerate(result["top_k"], 1):
            name = entry["class_name"] or f"Class {entry['class_id']}"
            print(f"    {i}. {name:30s} {entry['confidence'] * 100:6.2f}%")
    print()


if __name__ == "__main__":
    main()
