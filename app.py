"""FastAPI web application for Body Pawsitive interactive inference.

Replicates the ``combined_inference_overlay.ipynb`` widget in a browser:
select a dataset, breed group, image, segmentation backend and SAM 2 mode,
then run the three pipelines (classification, segmentation, pose) and
visualise the combined overlay side-by-side with the source image.

Supports uploading a custom image and exporting the overlay as PNG.

Usage
-----
.. code-block:: bash

    python app.py
    # Open http://localhost:5000

Served by **Uvicorn** (ASGI).
"""

from __future__ import annotations

import base64
import io
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from starlette.requests import Request
from starlette.responses import HTMLResponse

sys.path.insert(0, str(Path(__file__).parent / "src"))

from bcs_pipeline.inference import (
    load_classification_model,
    load_class_names,
    load_pose_model,
    load_segmentation_backend,
    predict_pose,
    predict_segmentation_with,
    predict_single,
    render_combined,
)
from bcs_pipeline.inference.visualization import (
    render_segmentation_layer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("bcs_app")

REPO_ROOT = Path(__file__).parent.resolve()

CLASSIFICATION_CKPT = (
    REPO_ROOT
    / "experiments/resnet50_adam_cosine_annealing/checkpoints/epoch=epoch=15-val_acc=val_acc=0.79-step=8240.ckpt"
)
DEEPLAB_CKPT = REPO_ROOT / "experiments/deeplabv3_resnet50_adam_cosine_annealing/checkpoints/last-v1.ckpt"
SAM2_CKPT = REPO_ROOT / "checkpoints/sam2.1_hiera_large.pt"
POSE_CKPT = REPO_ROOT / "runs/pose/train/weights/best.pt"

STANFORD_ROOT = REPO_ROOT / "data/stanford_dogs/images"
STANFORD_IMAGES = STANFORD_ROOT / "Images"
OXFORD_IMAGES = REPO_ROOT / "data/Oxford-IIIT_pet_dataset/images"
REDDIT_DIR = REPO_ROOT / "data/Reddit_example"

app = FastAPI(title="Body Pawsitive", docs_url=None, redoc_url=None)
templates = Jinja2Templates(directory=str(REPO_ROOT / "templates"))

_MODELS: dict = {
    "cls": None,
    "class_names": None,
    "seg": {},
    "pose": None,
}


def _ensure_classifier():
    if _MODELS["cls"] is None:
        _MODELS["cls"] = load_classification_model(str(CLASSIFICATION_CKPT))
        _MODELS["class_names"] = load_class_names(str(STANFORD_ROOT))
    return _MODELS["cls"], _MODELS["class_names"]


def _ensure_segmenter(backend: str):
    if backend not in _MODELS["seg"]:
        ckpt = str(SAM2_CKPT) if backend == "sam2" else str(DEEPLAB_CKPT)
        _MODELS["seg"][backend] = load_segmentation_backend(backend, ckpt)
    return _MODELS["seg"][backend]


def _ensure_pose():
    if _MODELS["pose"] is None:
        _MODELS["pose"] = load_pose_model(str(POSE_CKPT))
    return _MODELS["pose"]


def _list_image_files(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts and p.is_file())


def _stanford_groups() -> Dict[str, List[str]]:
    groups = {}
    if not STANFORD_IMAGES.is_dir():
        return groups
    for breed_dir in sorted(p for p in STANFORD_IMAGES.iterdir() if p.is_dir()):
        breed_name = breed_dir.name.split("-", 1)[1] if "-" in breed_dir.name else breed_dir.name
        groups[breed_name] = [p.name for p in _list_image_files(breed_dir)]
    return groups


def _oxford_groups() -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for img_path in _list_image_files(OXFORD_IMAGES):
        prefix = "_".join(img_path.stem.split("_")[:-1])
        groups.setdefault(prefix, []).append(img_path.name)
    return {k: sorted(v) for k, v in sorted(groups.items())}


def _reddit_groups() -> Dict[str, List[str]]:
    files = _list_image_files(REDDIT_DIR)
    return {"all": [f.name for f in files]} if files else {}


def _get_datasets() -> Dict[str, Dict[str, List[str]]]:
    return {
        "Reddit": _reddit_groups(),
        "Stanford Dogs": _stanford_groups(),
        "Oxford-IIIT Pet": _oxford_groups(),
    }


def _resolve_image_path(dataset: str, group: str, filename: str) -> Optional[Path]:
    if dataset == "Reddit":
        return REDDIT_DIR / filename
    if dataset == "Stanford Dogs":
        for breed_dir in STANFORD_IMAGES.iterdir():
            if not breed_dir.is_dir():
                continue
            breed_name = breed_dir.name.split("-", 1)[1] if "-" in breed_dir.name else breed_dir.name
            if breed_name == group:
                return breed_dir / filename
        return None
    if dataset == "Oxford-IIIT Pet":
        return OXFORD_IMAGES / filename
    return None


def _img_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _ground_truth(dataset: str, group: str) -> Optional[str]:
    if dataset == "Reddit":
        return None
    if dataset == "Stanford Dogs":
        return group if group else None
    if dataset == "Oxford-IIIT Pet":
        return group.replace("_", " ") if group else None
    return None


def _run_inference_on_image(
    img: Image.Image,
    image_name: str,
    seg_backend: str = "deeplab",
    sam2_mode: str = "prompted",
    top_k: int = 5,
    conf_threshold: float = 0.25,
) -> dict:
    img = img.convert("RGB")
    cls_model, class_names = _ensure_classifier()
    seg_handle = _ensure_segmenter(seg_backend)
    pose_model = _ensure_pose()

    cls = predict_single(cls_model, img, class_names=class_names, top_k=top_k)

    needs_pose_first = seg_backend == "sam2" and sam2_mode == "pose_prompted"
    pose = predict_pose(pose_model, img, conf_threshold=conf_threshold) if needs_pose_first else None

    seg = predict_segmentation_with(
        seg_backend, seg_handle, img,
        sam2_mode=sam2_mode, pose_result=pose,
    )

    if pose is None:
        pose = predict_pose(pose_model, img, conf_threshold=conf_threshold)

    canvas = img.convert("RGB")

    seg_layer = render_segmentation_layer(canvas, seg["mask"])

    full = render_combined(
        canvas,
        classification=cls, segmentation=seg, pose=pose,
        show_label=False,
    )

    mask = seg["mask"]
    unique, counts = np.unique(mask, return_counts=True)
    seg_dist = [
        {"class": int(u), "pct": round(float(c) / mask.size * 100, 1)}
        for u, c in zip(unique, counts)
    ]

    return {
        "source_b64": _img_to_b64(img, "JPEG"),
        "seg_b64": _img_to_b64(seg_layer, "PNG"),
        "full_b64": _img_to_b64(full, "PNG"),
        "classification": {
            "class_name": cls.get("class_name"),
            "confidence": round(cls.get("confidence", 0.0) * 100, 2),
            "top_k": [
                {
                    "rank": i + 1,
                    "class_name": e.get("class_name"),
                    "confidence": round(e.get("confidence", 0.0) * 100, 2),
                }
                for i, e in enumerate(cls.get("top_k", []))
            ],
        },
        "segmentation": {
            "backend": seg.get("backend", seg_backend),
            "distribution": seg_dist,
        },
        "pose": {
            "num_detections": pose.get("num_detections", 0),
            "best_conf": round(float(pose["box_confs"][0]), 2) if pose.get("num_detections", 0) > 0 else None,
        },
        "pose_annotations": {
            "boxes": pose["boxes"].tolist(),
            "keypoints": pose["keypoints"].tolist(),
            "kpt_confs": pose["kpt_confs"].tolist(),
            "box_confs": pose["box_confs"].tolist(),
        },
        "image_size": list(img.size),
        "image_name": image_name,
    }


def _run_inference(
    image_path: Path,
    seg_backend: str = "deeplab",
    sam2_mode: str = "prompted",
    top_k: int = 5,
    conf_threshold: float = 0.25,
) -> dict:
    img = Image.open(str(image_path)).convert("RGB")
    return _run_inference_on_image(
        img, image_path.name,
        seg_backend=seg_backend,
        sam2_mode=sam2_mode,
        top_k=top_k,
        conf_threshold=conf_threshold,
    )


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})


@app.get("/api/datasets")
def api_datasets():
    datasets = _get_datasets()
    payload = {}
    for name, groups in datasets.items():
        payload[name] = {
            "groups": {g: len(files) for g, files in groups.items()},
            "total_images": sum(len(f) for f in groups.values()),
        }
    return payload


@app.get("/api/images")
def api_images(dataset: str = Query(""), group: str = Query("")):
    datasets = _get_datasets()
    groups = datasets.get(dataset, {})
    return groups.get(group, [])


@app.get("/api/thumbnail/{dataset}/{group}/{filepath:path}")
def api_thumbnail(filepath: str, dataset: str, group: str):
    filename = filepath
    image_path = _resolve_image_path(dataset, group, filename)
    if image_path is None or not image_path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    img = Image.open(str(image_path)).convert("RGB")
    img.thumbnail((256, 256))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")


@app.post("/api/inference")
async def api_inference(request: Request):
    body = await request.json()
    dataset = body.get("dataset", "")
    group = body.get("group", "")
    filename = body.get("filename", "")
    seg_backend = body.get("seg_backend", "deeplab")
    sam2_mode = body.get("sam2_mode", "prompted")
    top_k = body.get("top_k", 5)
    conf_threshold = body.get("conf_threshold", 0.25)

    image_path = _resolve_image_path(dataset, group, filename)
    if image_path is None or not image_path.exists():
        return JSONResponse({"error": f"Image not found: {filename}"}, status_code=404)

    try:
        result = _run_inference(
            image_path,
            seg_backend=seg_backend,
            sam2_mode=sam2_mode,
            top_k=top_k,
            conf_threshold=conf_threshold,
        )
        result["ground_truth"] = _ground_truth(dataset, group)
        return result
    except Exception as exc:
        logger.exception("Inference failed for %s", image_path)
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/inference/upload")
async def api_inference_upload(
    file: UploadFile = File(...),
    seg_backend: str = Form("deeplab"),
    sam2_mode: str = Form("prompted"),
    top_k: str = Form("5"),
    conf_threshold: str = Form("0.25"),
):
    if not file.filename:
        return JSONResponse({"error": "Empty filename"}, status_code=400)

    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        return JSONResponse({"error": "Invalid image file"}, status_code=400)

    try:
        result = _run_inference_on_image(
            img, file.filename,
            seg_backend=seg_backend,
            sam2_mode=sam2_mode,
            top_k=int(top_k),
            conf_threshold=float(conf_threshold),
        )
        result["ground_truth"] = None
        return result
    except Exception as exc:
        logger.exception("Upload inference failed for %s", file.filename)
        return JSONResponse({"error": str(exc)}, status_code=500)


if __name__ == "__main__":
    logger.info("Starting Body Pawsitive on http://localhost:5000 (Uvicorn ASGI)")
    uvicorn.run(app, host="0.0.0.0", port=5000)
