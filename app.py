"""FastAPI web application for Body Pawsitive interactive inference.

Replicates the ``combined_inference_overlay.ipynb`` widget in a browser:
select a dataset, breed group, image, segmentation backend and SAM 2 mode,
then run the three pipelines (classification, segmentation, pose) and
visualise the combined overlay side-by-side with the source image.

Supports uploading a custom image, exporting the overlay as PNG / JSON,
and persisting results to an SQLite database.

Usage
-----
.. code-block:: bash

    python app.py                  # default port (see DEFAULT_PORT)
    PORT=8080 python app.py        # custom port via env variable

Served by **Uvicorn** (ASGI).
"""

from __future__ import annotations

import base64
import io
import logging
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from starlette.requests import Request
from starlette.responses import HTMLResponse

sys.path.insert(0, str(Path(__file__).parent / "src"))

from bcs_pipeline.app_checkpoints import (
    CLASSIFICATION_MODEL_NAME,
    CLASSIFICATION_NUM_CLASSES,
    CLASSIFICATION_CKPT_DIR,
    DEEPLAB_CKPT,
    SAM2_CKPT,
    SAM3_CKPT,
    SAM3_BPE,
    POSE_CKPT,
    REPO_ROOT,
    describe_active_models,
    resolve_classifier_ckpt,
)
from bcs_pipeline.datasets import (
    collect_all_images,
    get_datasets,
    ground_truth,
    list_image_files,
    resolve_image_path,
    STANFORD_ROOT,
    OXFORD_ROOT,
)
from bcs_pipeline.db import (
    InferenceRun,
    UserAnnotation,
    delete_run as db_delete_run,
    init_db,
    list_runs as db_list_runs,
    load_run as db_load_run,
    save_annotations as db_save_annotations,
    save_run as db_save_run,
    session_scope,
)
from bcs_pipeline.inference import (
    load_classification_model,
    load_combined_class_names,
    load_pose_model,
    load_segmentation_backend,
    render_combined,
)
from bcs_pipeline.inference.visualization import (
    render_segmentation_layer,
)
from bcs_pipeline.inference_format import (
    format_inference_result,
    run_core_inference,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("bcs_app")

OUTPUT_DIR = REPO_ROOT / "data" / "outputs"

_DB_ENGINE = None
_DB_SESSION = None

_MODELS: dict = {
    "cls": None,
    "class_names": None,
    "seg": {},
    "pose": None,
}


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _DB_ENGINE, _DB_SESSION
    db_path = str(REPO_ROOT / "data" / "bcs_app.db")
    _DB_ENGINE, _DB_SESSION = init_db(db_path)
    yield
    if _DB_ENGINE:
        _DB_ENGINE.dispose()


app = FastAPI(title="Body Pawsitive", docs_url=None, redoc_url=None, lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=str(REPO_ROOT / "static")), name="static")
templates = Jinja2Templates(directory=str(REPO_ROOT / "templates"))


_preload_state: dict = {
    "running": False,
    "total": 0,
    "processed": 0,
    "errors": 0,
    "done": False,
    "message": "",
}


def _preload_worker(seg_backend: str, sam2_mode: str, sam3_mode: str = "pose_concept_prompted"):
    global _preload_state
    try:
        _preload_state["running"] = True
        _preload_state["done"] = False
        _preload_state["processed"] = 0
        _preload_state["errors"] = 0
        _preload_state["message"] = "Chargement des mod\u00e8les\u2026"

        cls_model, class_names = _ensure_classifier()
        seg_handle = _ensure_segmenter(seg_backend)
        pose_model = _ensure_pose()

        all_entries = collect_all_images()
        _preload_state["total"] = len(all_entries)

        if not all_entries:
            _preload_state["message"] = "Aucune image trouv\u00e9e"
            _preload_state["done"] = True
            _preload_state["running"] = False
            return

        with session_scope(_DB_SESSION) as session:
            rows = session.query(
                InferenceRun.image_name,
                InferenceRun.dataset,
                InferenceRun.group_name,
            ).filter(InferenceRun.seg_backend == seg_backend).all()
            existing = {(r.image_name, r.dataset, r.group_name) for r in rows}

        to_process = [
            (p, ds, gn, gt) for p, ds, gn, gt in all_entries
            if (p.name, ds, gn) not in existing
        ]
        _preload_state["total"] = len(to_process)

        if not to_process:
            _preload_state["message"] = "Base d\u00e9j\u00e0 compl\u00e8te"
            _preload_state["done"] = True
            _preload_state["running"] = False
            return

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        _preload_state["message"] = f"Traitement de {len(to_process)} images\u2026"

        for idx, (img_path, dataset, group_name, gt) in enumerate(to_process):
            if not _preload_state["running"]:
                _preload_state["message"] = "Pr\u00e9-chargement annul\u00e9"
                break

            try:
                img = Image.open(str(img_path)).convert("RGB")
                result = _run_inference_on_image(
                    img, img_path.name,
                    seg_backend=seg_backend,
                    sam2_mode=sam2_mode,
                    sam3_mode=sam3_mode,
                )

                with session_scope(_DB_SESSION) as session:
                    try:
                        db_save_run(
                            session, OUTPUT_DIR, result,
                            source_type="dataset",
                            dataset=dataset,
                            group_name=group_name,
                            ground_truth=gt,
                            seg_backend=seg_backend,
                            sam2_mode=sam2_mode,
                        )
                    except Exception:
                        logger.exception("Failed to save preload run for %s", img_path.name)

                _preload_state["processed"] += 1
            except Exception:
                logger.exception("Preload failed for %s", img_path)
                _preload_state["errors"] += 1

            _preload_state["message"] = (
                f"{_preload_state['processed']}/{len(to_process)} trait\u00e9es"
                + (f" — {_preload_state['errors']} erreur(s)" if _preload_state["errors"] else "")
            )

        _preload_state["done"] = True
        _preload_state["message"] = (
            f"Termin\u00e9 : {_preload_state['processed']} images, "
            f"{_preload_state['errors']} erreur(s)"
        )
    except Exception:
        logger.exception("Preload worker crashed")
        _preload_state["message"] = "Erreur lors du pr\u00e9-chargement"
        _preload_state["done"] = True
    finally:
        _preload_state["running"] = False


def _ensure_classifier():
    if _MODELS["cls"] is not None:
        return _MODELS["cls"], _MODELS["class_names"]

    ckpt = resolve_classifier_ckpt()
    if ckpt is None:
        raise FileNotFoundError(
            f"No classifier checkpoint found under {CLASSIFICATION_CKPT_DIR}. "
            f"Train one with `python train.py --config-name=config_dogs_cats_vit` "
            "and place the resulting `last.ckpt` there, or edit "
            "src/bcs_pipeline/app_checkpoints.py to point to a different model."
        )
    logger.info("Loading classifier (%s, %d classes) from %s",
                CLASSIFICATION_MODEL_NAME, CLASSIFICATION_NUM_CLASSES,
                ckpt.relative_to(REPO_ROOT))
    _MODELS["cls"] = load_classification_model(
        str(ckpt),
        model_name=CLASSIFICATION_MODEL_NAME,
        num_classes=CLASSIFICATION_NUM_CLASSES,
    )
    _MODELS["class_names"] = load_combined_class_names(str(STANFORD_ROOT), str(OXFORD_ROOT))
    return _MODELS["cls"], _MODELS["class_names"]


def _ensure_segmenter(backend: str):
    if backend not in _MODELS["seg"]:
        if backend == "sam2":
            ckpt = str(SAM2_CKPT)
            _MODELS["seg"][backend] = load_segmentation_backend(backend, ckpt)
        elif backend == "sam3":
            ckpt = str(SAM3_CKPT)
            bpe = str(SAM3_BPE) if SAM3_BPE.is_file() else None
            _MODELS["seg"][backend] = load_segmentation_backend(
                backend, ckpt, sam3_bpe_path=bpe,
            )
        else:
            ckpt = str(DEEPLAB_CKPT)
            _MODELS["seg"][backend] = load_segmentation_backend(backend, ckpt)
    return _MODELS["seg"][backend]


def _ensure_pose():
    if _MODELS["pose"] is None:
        _MODELS["pose"] = load_pose_model(str(POSE_CKPT))
    return _MODELS["pose"]


def _img_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _run_inference_on_image(
    img: Image.Image,
    image_name: str,
    seg_backend: str = "deeplab",
    sam2_mode: str = "prompted",
    sam3_mode: str = "pose_concept_prompted",
    top_k: int = 5,
    conf_threshold: float = 0.25,
) -> dict:
    img = img.convert("RGB")
    cls_model, class_names = _ensure_classifier()
    seg_handle = _ensure_segmenter(seg_backend)
    pose_model = _ensure_pose()

    cls, seg, pose = run_core_inference(
        cls_model, class_names, seg_handle, seg_backend, pose_model, img,
        sam2_mode=sam2_mode, sam3_mode=sam3_mode,
        top_k=top_k, conf_threshold=conf_threshold,
    )

    canvas = img.convert("RGB")
    seg_layer = render_segmentation_layer(canvas, seg["mask"])
    full = render_combined(
        canvas, classification=cls, segmentation=seg, pose=pose,
        show_label=False,
    )

    result = format_inference_result(cls, seg, pose, image_name, img.size, seg_backend)
    result["source_b64"] = _img_to_b64(img, "JPEG")
    result["seg_b64"] = _img_to_b64(seg_layer, "PNG")
    result["full_b64"] = _img_to_b64(full, "PNG")
    return result


def _run_inference(
    image_path: Path,
    seg_backend: str = "deeplab",
    sam2_mode: str = "prompted",
    sam3_mode: str = "pose_concept_prompted",
    top_k: int = 5,
    conf_threshold: float = 0.25,
) -> dict:
    img = Image.open(str(image_path)).convert("RGB")
    return _run_inference_on_image(
        img, image_path.name,
        seg_backend=seg_backend,
        sam2_mode=sam2_mode,
        sam3_mode=sam3_mode,
        top_k=top_k,
        conf_threshold=conf_threshold,
    )


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})


@app.get("/api/datasets")
def api_datasets():
    datasets = get_datasets()
    payload = {}
    for name, groups in datasets.items():
        payload[name] = {
            "groups": {g: len(files) for g, files in groups.items()},
            "total_images": sum(len(f) for f in groups.values()),
        }
    return payload


@app.get("/api/images")
def api_images(dataset: str = Query(""), group: str = Query("")):
    datasets = get_datasets()
    groups = datasets.get(dataset, {})
    files = groups.get(group, [])
    # Find which images already have annotations saved in the DB
    annotated_names: set = set()
    with session_scope(_DB_SESSION) as session:
        rows = (
            session.query(InferenceRun.image_name)
            .join(UserAnnotation, InferenceRun.annotation)
            .filter(
                InferenceRun.dataset == dataset,
                InferenceRun.group_name == group,
            )
            .all()
        )
        annotated_names = {r[0] for r in rows}
    return [{"name": f, "annotated": f in annotated_names} for f in files]


@app.get("/api/thumbnail/{dataset}/{group}/{filepath:path}")
def api_thumbnail(filepath: str, dataset: str, group: str):
    image_path = resolve_image_path(dataset, group, filepath)
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
    sam3_mode = body.get("sam3_mode", "pose_concept_prompted")
    top_k = body.get("top_k", 5)
    conf_threshold = body.get("conf_threshold", 0.25)

    image_path = resolve_image_path(dataset, group, filename)
    if image_path is None or not image_path.exists():
        return JSONResponse({"error": f"Image not found: {filename}"}, status_code=404)

    try:
        result = _run_inference(
            image_path,
            seg_backend=seg_backend,
            sam2_mode=sam2_mode,
            sam3_mode=sam3_mode,
            top_k=top_k,
            conf_threshold=conf_threshold,
        )
        result["ground_truth"] = ground_truth(dataset, group)

        with session_scope(_DB_SESSION) as session:
            try:
                run = db_save_run(
                    session, OUTPUT_DIR, result,
                    source_type="dataset", dataset=dataset, group_name=group,
                    ground_truth=result["ground_truth"],
                    seg_backend=seg_backend, sam2_mode=sam2_mode,
                )
                result["run_id"] = run.id
            except Exception:
                logger.exception("Failed to save run to DB")

        return result
    except Exception as exc:
        logger.exception("Inference failed for %s", image_path)
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/inference/upload")
async def api_inference_upload(
    file: UploadFile = File(...),
    seg_backend: str = Form("deeplab"),
    sam2_mode: str = Form("prompted"),
    sam3_mode: str = Form("pose_concept_prompted"),
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
            sam3_mode=sam3_mode,
            top_k=int(top_k),
            conf_threshold=float(conf_threshold),
        )
        result["ground_truth"] = None

        with session_scope(_DB_SESSION) as session:
            try:
                run = db_save_run(
                    session, OUTPUT_DIR, result,
                    source_type="upload", seg_backend=seg_backend, sam2_mode=sam2_mode,
                )
                result["run_id"] = run.id
            except Exception:
                logger.exception("Failed to save upload run to DB")

        return result
    except Exception as exc:
        logger.exception("Upload inference failed for %s", file.filename)
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/history")
def api_history(
    limit: int = Query(50),
    offset: int = Query(0),
    sort: str = Query("last_inferred_at"),
    order: str = Query("desc"),
):
    with session_scope(_DB_SESSION) as session:
        total = session.query(InferenceRun).count()
        runs = db_list_runs(
            session, limit=limit, offset=offset,
            sort_by=sort, sort_order=order,
        )
        return {
            "total": total,
            "runs": runs,
            "page": offset // limit + 1,
            "page_size": limit,
            "sort": sort,
            "order": order,
        }


@app.get("/api/history/{run_id}")
def api_history_detail(run_id: int):
    with session_scope(_DB_SESSION) as session:
        data = db_load_run(session, run_id)
        if data is None:
            return JSONResponse({"error": "Run not found"}, status_code=404)
        return data


def _decode_and_save_mask(run_id: int, mask_b64: str) -> str:
    if mask_b64.startswith("data:"):
        mask_b64 = mask_b64.split(",", 1)[1]
    raw = base64.b64decode(mask_b64)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mask_path = OUTPUT_DIR / f"run_{run_id}_mask.png"
    img = Image.open(io.BytesIO(raw))
    img.save(mask_path, format="PNG")
    return str(mask_path)


@app.post("/api/history/{run_id}/annotations")
async def api_save_annotations(run_id: int, request: Request):
    body = await request.json()
    with session_scope(_DB_SESSION) as session:
        try:
            mask_b64 = body.get("mask_b64")
            mask_path = _decode_and_save_mask(run_id, mask_b64) if mask_b64 else None
            ann = db_save_annotations(
                session,
                run_id,
                boxes=body.get("boxes", []),
                keypoints=body.get("keypoints", []),
                kpt_confs=body.get("kpt_confs", []),
                box_confs=body.get("box_confs", []),
                comments=body.get("comments", []),
                mask_path=mask_path,
            )
            if ann is None:
                return JSONResponse({"error": "Run not found"}, status_code=404)
            return {"status": "ok", "run_id": run_id}
        except Exception as exc:
            logger.exception("Failed to save annotations for run %d", run_id)
            return JSONResponse({"error": str(exc)}, status_code=500)


@app.delete("/api/history/{run_id}")
def api_delete_history(run_id: int):
    with session_scope(_DB_SESSION) as session:
        ok = db_delete_run(session, run_id)
        if not ok:
            return JSONResponse({"error": "Run not found"}, status_code=404)
        return {"status": "deleted"}


@app.get("/api/preload/status")
def api_preload_status(seg_backend: str = Query("")):
    total_images = len(collect_all_images())
    with session_scope(_DB_SESSION) as session:
        q = session.query(InferenceRun).filter(InferenceRun.source_type == "dataset")
        if seg_backend:
            q = q.filter(InferenceRun.seg_backend == seg_backend)
        db_count = q.count()
    remaining = max(0, total_images - db_count)
    return {
        "total_images": total_images,
        "db_count": db_count,
        "remaining": remaining,
        "complete": remaining == 0,
        "seg_backend": seg_backend or None,
        "running": _preload_state["running"],
        "progress": _preload_state.copy(),
    }


@app.post("/api/preload/start")
async def api_preload_start(request: Request):
    if _preload_state["running"]:
        return JSONResponse({"error": "Pr\u00e9-chargement d\u00e9j\u00e0 en cours"}, status_code=409)

    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    seg_backend = body.get("seg_backend", "deeplab")
    sam2_mode = body.get("sam2_mode", "prompted")
    sam3_mode = body.get("sam3_mode", "pose_concept_prompted")

    ckpt = resolve_classifier_ckpt()
    if ckpt is None:
        return JSONResponse({"error": "Aucun checkpoint de classification trouv\u00e9"}, status_code=400)

    total_images = len(collect_all_images())
    if total_images == 0:
        return JSONResponse({"error": "Aucune image trouv\u00e9e dans les datasets"}, status_code=400)

    thread = threading.Thread(
        target=_preload_worker,
        args=(seg_backend, sam2_mode, sam3_mode),
        daemon=True,
    )
    thread.start()
    return {"status": "started", "total_images": total_images}


@app.post("/api/preload/stop")
def api_preload_stop():
    if not _preload_state["running"]:
        return JSONResponse({"error": "Aucun pr\u00e9-chargement en cours"}, status_code=409)
    _preload_state["running"] = False
    return {"status": "stopping"}


# ── Configuration ───────────────────────────────────────────────────────────
DEFAULT_PORT = 5000
# Override at runtime with:  PORT=8080 python app.py
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", DEFAULT_PORT))
    logger.info("Starting Body Pawsitive on http://localhost:%d (Uvicorn ASGI)", port)
    for line in describe_active_models().splitlines():
        logger.info(line)
    uvicorn.run(app, host="0.0.0.0", port=port)
