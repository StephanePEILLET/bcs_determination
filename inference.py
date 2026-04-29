"""
BCS Determination – Inference CLI.

Two modes:

* ``--mode classify`` (default, legacy): loads a classification checkpoint and
  prints the top-k predicted breeds.
* ``--mode full``: runs whichever of the three pipelines (classification,
  segmentation, pose) have a checkpoint provided, then composes a single PNG
  with the overlays.

Heavy logic lives in ``bcs_pipeline.inference`` so that ``app.py`` and the
notebooks can reuse the same code.

Usage
-----
.. code-block:: bash

    # Classify only
    python inference.py \\
        --image_path sample_dog.jpg \\
        --checkpoint_path experiments/.../checkpoints/best.ckpt \\
        --data_dir data/stanford_dogs

    # Combined visualization
    python inference.py --mode full \\
        --image_path data/Reddit_example/photo.webp \\
        --checkpoint_path experiments/resnet50.../best.ckpt \\
        --seg_checkpoint experiments/deeplabv3.../last-v1.ckpt \\
        --pose_checkpoint runs/pose/train/weights/best.pt \\
        --output outputs/photo_full.png \\
        --data_dir data/stanford_dogs
"""

import argparse
import sys
from pathlib import Path

from PIL import Image

try:
    from bcs_pipeline.inference import (
        load_class_names,
        load_model,
        predict_single,
        run_full_inference,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from bcs_pipeline.inference import (
        load_class_names,
        load_model,
        predict_single,
        run_full_inference,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict dog breed (and optionally segmentation + pose) from an image.",
    )
    parser.add_argument(
        "--image_path", type=str, required=True,
        help="Path to the image to process.",
    )
    parser.add_argument(
        "--mode", choices=["classify", "full"], default="classify",
        help="'classify' = legacy top-k printout. 'full' = run all 3 pipelines.",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None,
        help="Classification checkpoint (.ckpt). Required for --mode classify.",
    )
    parser.add_argument(
        "--seg_checkpoint", type=str, default=None,
        help="Segmentation (DeepLabV3) checkpoint (.ckpt). Optional.",
    )
    parser.add_argument(
        "--pose_checkpoint", type=str, default=None,
        help="Pose detection (Ultralytics YOLO) weights (.pt). Optional.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path of the combined visualization PNG (full mode).",
    )
    parser.add_argument(
        "--coco_path", type=str, default=None,
        help="Optional path to dump the inference outputs as a COCO-format JSON "
             "(bbox + keypoints + segmentation polygons).",
    )
    parser.add_argument(
        "--polygon_simplify", type=float, default=2.0,
        help="Douglas-Peucker tolerance (px) used when vectorizing the seg mask.",
    )
    parser.add_argument(
        "--model_name", type=str, default="resnet50",
        help="Classification architecture (default: resnet50).",
    )
    parser.add_argument(
        "--num_classes", type=int, default=120,
        help="Number of classification classes (default: 120).",
    )
    parser.add_argument(
        "--image_size", type=int, default=224,
        help="Classification input resolution (default: 224).",
    )
    parser.add_argument(
        "--seg_image_size", type=int, default=256,
        help="Segmentation input resolution (default: 256, DeepLabV3 only).",
    )
    parser.add_argument(
        "--seg_backend", choices=["deeplab", "sam2"], default="deeplab",
        help="Segmentation backend: 'deeplab' (fine-tuned trimap) or 'sam2' (zero-shot foundation).",
    )
    parser.add_argument(
        "--sam2_mode", choices=["prompted", "automatic", "pose_prompted"], default="prompted",
        help="SAM2 prompting mode (only used when --seg_backend=sam2). "
             "'pose_prompted' requires --pose_checkpoint.",
    )
    parser.add_argument(
        "--sam2_config", type=str, default=None,
        help="SAM2 model config YAML (default: configs/sam2.1/sam2.1_hiera_l.yaml).",
    )
    parser.add_argument(
        "--conf_threshold", type=float, default=0.25,
        help="YOLO detection confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Number of top predictions to display (default: 5).",
    )
    parser.add_argument(
        "--data_dir", type=str, default="",
        help="(Optional) Dataset root used to resolve human-readable class names.",
    )
    return parser.parse_args()


def _print_classification(result: dict, top_k: int) -> None:
    label = result["class_name"] or f"Class ID {result['class_id']}"
    print(f"\n{'─' * 40}")
    print(f"  Prediction : {label}")
    print(f"  Confidence : {result['confidence'] * 100:.2f}%")
    print(f"{'─' * 40}")
    if result.get("top_k"):
        print(f"\n  Top-{top_k} predictions:")
        for i, entry in enumerate(result["top_k"], 1):
            name = entry["class_name"] or f"Class {entry['class_id']}"
            print(f"    {i}. {name:30s} {entry['confidence'] * 100:6.2f}%")
    print()


def _print_full_summary(result: dict) -> None:
    cls = result.get("classification")
    seg = result.get("segmentation")
    pose = result.get("pose")
    out = result.get("output_path")

    print(f"\n{'─' * 50}")
    if cls:
        label = cls["class_name"] or f"Class ID {cls['class_id']}"
        print(f"  Breed       : {label} ({cls['confidence'] * 100:.2f}%)")
    else:
        print("  Breed       : (not run)")

    if seg:
        import numpy as np
        mask = seg["mask"]
        unique, counts = np.unique(mask, return_counts=True)
        dist = ", ".join(f"cls{int(u)}={c / mask.size * 100:.1f}%"
                         for u, c in zip(unique, counts))
        print(f"  Segmentation: {dist}")
    else:
        print("  Segmentation: (not run)")

    if pose:
        print(f"  Pose        : {pose['num_detections']} detection(s)")
    else:
        print("  Pose        : (not run)")

    if out:
        print(f"  Output PNG  : {out}")
    if result.get("coco_path"):
        print(f"  COCO JSON   : {result['coco_path']}")
    print(f"{'─' * 50}\n")


def main() -> None:
    args = parse_args()

    if args.mode == "full":
        if not (args.checkpoint_path or args.seg_checkpoint or args.pose_checkpoint):
            sys.exit("Error: --mode full requires at least one of "
                     "--checkpoint_path, --seg_checkpoint, --pose_checkpoint.")
        print(f"Processing image {args.image_path}…")
        result = run_full_inference(
            image=args.image_path,
            classification_ckpt=args.checkpoint_path or None,
            segmentation_ckpt=args.seg_checkpoint or None,
            pose_ckpt=args.pose_checkpoint or None,
            output_path=args.output or None,
            classification_model_name=args.model_name,
            classification_num_classes=args.num_classes,
            classification_image_size=args.image_size,
            segmentation_image_size=args.seg_image_size,
            segmentation_backend=args.seg_backend,
            sam2_mode=args.sam2_mode,
            sam2_config=args.sam2_config,
            data_dir=args.data_dir or None,
            top_k=args.top_k,
            pose_conf_threshold=args.conf_threshold,
            coco_path=args.coco_path,
            polygon_simplify=args.polygon_simplify,
        )
        _print_full_summary(result)
        return

    # Legacy classify path
    if not args.checkpoint_path:
        sys.exit("Error: --checkpoint_path is required for --mode classify.")

    model = load_model(
        checkpoint_path=args.checkpoint_path,
        model_name=args.model_name,
        num_classes=args.num_classes,
    )
    class_names = load_class_names(args.data_dir) if args.data_dir else None

    print(f"Processing image {args.image_path}…")
    image = Image.open(args.image_path).convert("RGB")
    result = predict_single(
        model, image,
        image_size=args.image_size,
        class_names=class_names,
        top_k=args.top_k,
    )
    _print_classification(result, args.top_k)


def predict(
    image_path: str,
    checkpoint_path: str,
    model_name: str = "resnet50",
    num_classes: int = 120,
    image_size: int = 224,
    top_k: int = 5,
    data_dir: str = "",
) -> dict:
    """Programmatic single-image classification entry-point used by ``app.py``."""
    model = load_model(checkpoint_path, model_name=model_name, num_classes=num_classes)
    class_names = load_class_names(data_dir) if data_dir else None
    image = Image.open(image_path).convert("RGB")
    return predict_single(
        model, image,
        image_size=image_size,
        class_names=class_names,
        top_k=top_k,
    )


if __name__ == "__main__":
    main()
