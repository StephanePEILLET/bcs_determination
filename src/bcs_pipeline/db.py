"""SQLAlchemy database models and helpers for Body Pawsitive."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import base64

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as SaSession
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

logger = logging.getLogger("bcs_db")

Base = declarative_base()


class InferenceRun(Base):
    __tablename__ = "inference_runs"
    __table_args__ = (
        # One run per (image, dataset, group, segmentation backend). Prevents
        # the preload worker and an interactive `/api/inference` call from
        # racing on the same image and inserting duplicates.
        # NB: NULLs in `dataset` / `group_name` are treated as distinct by
        # SQLite, so user-uploaded images (where both are NULL) are never
        # blocked from re-running by this constraint.
        UniqueConstraint(
            "image_name", "dataset", "group_name", "seg_backend",
            name="uq_run_image_dataset_group_backend",
        ),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_inferred_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    image_name = Column(String, nullable=False)
    source_type = Column(String, default="dataset")
    dataset = Column(String, nullable=True)
    group_name = Column(String, nullable=True)
    ground_truth = Column(String, nullable=True)
    image_width = Column(Integer)
    image_height = Column(Integer)
    seg_backend = Column(String, default="deeplab")
    sam2_mode = Column(String, default="prompted")
    predicted_class = Column(String, nullable=True)
    predicted_confidence = Column(Float, nullable=True)
    predicted_species = Column(String, nullable=True)
    num_pose_detections = Column(Integer, default=0)
    best_pose_conf = Column(Float, nullable=True)
    predicted_bcs = Column(Float, nullable=True)
    bcs_category = Column(String, nullable=True)
    output_path = Column(String, nullable=True)

    annotation = relationship(
        "UserAnnotation",
        back_populates="run",
        uselist=False,
        cascade="all, delete-orphan",
    )

    def to_summary(self) -> Dict[str, Any]:
        has_ann = self.annotation is not None
        num_comments = 0
        if has_ann and self.annotation.comments:
            num_comments = len(json.loads(self.annotation.comments))
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_inferred_at": self.last_inferred_at.isoformat() if self.last_inferred_at else None,
            "image_name": self.image_name,
            "source_type": self.source_type,
            "dataset": self.dataset,
            "predicted_class": self.predicted_class,
            "predicted_confidence": round(self.predicted_confidence, 2)
            if self.predicted_confidence
            else None,
            "predicted_species": self.predicted_species,
            "seg_backend": self.seg_backend,
            "num_pose_detections": self.num_pose_detections,
            "predicted_bcs": round(self.predicted_bcs, 2)
            if self.predicted_bcs is not None
            else None,
            "bcs_category": self.bcs_category,
            "has_annotations": has_ann,
            "num_comments": num_comments,
        }


class UserAnnotation(Base):
    __tablename__ = "user_annotations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("inference_runs.id"), unique=True)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    boxes = Column(Text, nullable=True)
    keypoints = Column(Text, nullable=True)
    kpt_confs = Column(Text, nullable=True)
    box_confs = Column(Text, nullable=True)
    comments = Column(Text, nullable=True)
    mask_path = Column(String, nullable=True)

    run = relationship("InferenceRun", back_populates="annotation")


def init_db(db_path: str):
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    _migrate_schema(engine)
    session_local = sessionmaker(bind=engine)
    logger.info("Database initialized at %s", db_path)
    return engine, session_local


@contextmanager
def session_scope(session_factory) -> Generator[SaSession, None, None]:
    """Yield a fresh ORM session: rollback on exception, always close.

    **Does NOT commit.** Persistence helpers in this module
    (``save_run``, ``save_annotations``, ``delete_run``) commit
    themselves. ``session_scope`` only guards against leaking sessions and
    half-applied state when something raises.
    """
    session = session_factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _migrate_schema(engine) -> None:
    """Apply schema migrations after the initial CREATE TABLE. Idempotent.

    On an existing DB, ``create_all`` is a no-op for already-present tables,
    so any constraint or column added after the initial schema must be
    backfilled here.
    """
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())

    # ── user_annotations.mask_path (added later) ────────────────────────────
    if "user_annotations" in table_names:
        cols = {c["name"] for c in inspector.get_columns("user_annotations")}
        if "mask_path" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE user_annotations ADD COLUMN mask_path VARCHAR"))
            logger.info("Migrated user_annotations: added mask_path column")

    # ── inference_runs.predicted_bcs / bcs_category (added later) ───────────
    if "inference_runs" in table_names:
        run_cols = {c["name"] for c in inspector.get_columns("inference_runs")}
        with engine.begin() as conn:
            if "predicted_bcs" not in run_cols:
                conn.execute(text("ALTER TABLE inference_runs ADD COLUMN predicted_bcs FLOAT"))
                logger.info("Migrated inference_runs: added predicted_bcs column")
            if "bcs_category" not in run_cols:
                conn.execute(text("ALTER TABLE inference_runs ADD COLUMN bcs_category VARCHAR"))
                logger.info("Migrated inference_runs: added bcs_category column")
            if "predicted_species" not in run_cols:
                conn.execute(text("ALTER TABLE inference_runs ADD COLUMN predicted_species VARCHAR"))
                logger.info("Migrated inference_runs: added predicted_species column")

    # ── inference_runs.last_inferred_at (added later) ───────────────────────
    if "inference_runs" in table_names:
        run_cols = {c["name"] for c in inspector.get_columns("inference_runs")}
        if "last_inferred_at" not in run_cols:
            with engine.begin() as conn:
                conn.execute(text(
                    "ALTER TABLE inference_runs ADD COLUMN last_inferred_at DATETIME"
                ))
                # Backfill existing rows: treat first creation as last activity.
                conn.execute(text(
                    "UPDATE inference_runs SET last_inferred_at = created_at "
                    "WHERE last_inferred_at IS NULL"
                ))
                conn.execute(text(
                    "CREATE INDEX IF NOT EXISTS ix_inference_runs_last_inferred_at "
                    "ON inference_runs (last_inferred_at)"
                ))
            logger.info("Migrated inference_runs: added last_inferred_at column (backfilled from created_at)")

    # ── inference_runs UNIQUE(image_name, dataset, group_name, seg_backend) ──
    # On a pre-existing DB, the constraint declared in __table_args__ is not
    # retro-applied by create_all. Backfill via a unique index after pruning
    # any duplicates (keeping the most recent row per group).
    if "inference_runs" in table_names:
        existing_indexes = {idx["name"] for idx in inspector.get_indexes("inference_runs")}
        if "uq_run_image_dataset_group_backend" not in existing_indexes:
            with engine.begin() as conn:
                # Find duplicate groups (>1 row sharing the unique tuple).
                dup_rows = conn.execute(text(
                    "SELECT image_name, dataset, group_name, seg_backend, COUNT(*) AS n "
                    "FROM inference_runs "
                    "GROUP BY image_name, dataset, group_name, seg_backend "
                    "HAVING n > 1"
                )).fetchall()
                if dup_rows:
                    logger.warning(
                        "Found %d duplicate run group(s) — pruning all but the most recent in each.",
                        len(dup_rows),
                    )
                    # For each duplicated tuple, keep MAX(id) and delete the rest.
                    deleted = conn.execute(text(
                        "DELETE FROM inference_runs WHERE id IN ("
                        "  SELECT id FROM inference_runs r1 "
                        "  WHERE id < (SELECT MAX(id) FROM inference_runs r2 "
                        "              WHERE r2.image_name = r1.image_name "
                        "                AND COALESCE(r2.dataset,'')   = COALESCE(r1.dataset,'') "
                        "                AND COALESCE(r2.group_name,'') = COALESCE(r1.group_name,'') "
                        "                AND COALESCE(r2.seg_backend,'') = COALESCE(r1.seg_backend,''))"
                        ")"
                    ))
                    logger.warning("Pruned %d duplicate run(s).", deleted.rowcount or 0)
                conn.execute(text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_run_image_dataset_group_backend "
                    "ON inference_runs (image_name, dataset, group_name, seg_backend)"
                ))
            logger.info("Migrated inference_runs: created unique index on (image_name, dataset, group_name, seg_backend)")


def save_run(
    session: SaSession,
    output_dir: Path,
    result: Dict[str, Any],
    source_type: str = "dataset",
    dataset: Optional[str] = None,
    group_name: Optional[str] = None,
    ground_truth: Optional[str] = None,
    seg_backend: str = "deeplab",
    sam2_mode: str = "prompted",
) -> InferenceRun:
    """Persist an inference result. Idempotent on the unique tuple
    ``(image_name, dataset, group_name, seg_backend)``: if a row already
    exists for that key, the existing row is returned and the on-disk JSON
    is refreshed (so the latest prediction is reflected on subsequent
    GETs)."""
    cls = result.get("classification", {})
    pose = result.get("pose", {})
    bcs = result.get("bcs") or {}
    img_size = result.get("image_size", [0, 0])
    image_name = result.get("image_name", "unknown")

    run = InferenceRun(
        image_name=image_name,
        source_type=source_type,
        dataset=dataset,
        group_name=group_name,
        ground_truth=ground_truth,
        image_width=img_size[0],
        image_height=img_size[1],
        seg_backend=seg_backend,
        sam2_mode=sam2_mode,
        predicted_class=cls.get("class_name"),
        predicted_confidence=cls.get("confidence"),
        predicted_species=result.get("species", {}).get("species")
        if isinstance(result.get("species"), dict)
        else None,
        num_pose_detections=pose.get("num_detections", 0),
        best_pose_conf=pose.get("best_conf"),
        predicted_bcs=bcs.get("bcs"),
        bcs_category=bcs.get("category"),
    )
    session.add(run)
    try:
        session.flush()
    except IntegrityError:
        # Another writer beat us to it; recover the canonical row instead of
        # surfacing a 500. The caller will see a fully-populated InferenceRun.
        session.rollback()
        existing = (
            session.query(InferenceRun)
            .filter(
                InferenceRun.image_name == image_name,
                InferenceRun.dataset == dataset,
                InferenceRun.group_name == group_name,
                InferenceRun.seg_backend == seg_backend,
            )
            .first()
        )
        if existing is None:
            raise  # constraint violated for some other reason — propagate
        logger.info(
            "save_run: run already exists for %s/%s/%s/%s → returning #%d",
            image_name, dataset, group_name, seg_backend, existing.id,
        )
        # Bump last_inferred_at so the row floats to the top of history.
        existing.last_inferred_at = datetime.now(timezone.utc)
        session.commit()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"run_{existing.id}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
        logger.info("Refreshed JSON for existing run #%d", existing.id)
        return existing

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"run_{run.id}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    run.output_path = str(output_file)

    session.commit()
    logger.info("Saved inference run #%d -> %s", run.id, output_file.name)
    return run


def save_annotations(
    session: SaSession,
    run_id: int,
    boxes: List,
    keypoints: List,
    kpt_confs: List,
    box_confs: List,
    comments: List,
    mask_path: Optional[str] = None,
) -> Optional[UserAnnotation]:
    run = session.query(InferenceRun).filter(InferenceRun.id == run_id).first()
    if not run:
        return None

    ann = run.annotation
    if ann is None:
        ann = UserAnnotation(run_id=run_id)
        run.annotation = ann

    ann.boxes = json.dumps(boxes)
    ann.keypoints = json.dumps(keypoints)
    ann.kpt_confs = json.dumps(kpt_confs)
    ann.box_confs = json.dumps(box_confs)
    ann.comments = json.dumps(comments)
    if mask_path is not None:
        ann.mask_path = mask_path
    ann.updated_at = datetime.now(timezone.utc)

    session.commit()

    # Sync the JSON file on disk so it stays consistent with the DB
    if run.output_path:
        output_file = Path(run.output_path)
        if output_file.exists():
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["pose_annotations"] = {
                    "boxes": boxes,
                    "keypoints": keypoints,
                    "kpt_confs": kpt_confs,
                    "box_confs": box_confs,
                }
                data["user_comments"] = comments
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
                logger.info("Updated JSON file %s with annotations", output_file.name)
            except Exception:
                logger.warning("Could not update JSON file %s", output_file.name, exc_info=True)

    logger.info("Saved annotations for run #%d (%d comments)", run_id, len(comments))
    return ann


def load_run(session: SaSession, run_id: int) -> Optional[Dict[str, Any]]:
    run = session.query(InferenceRun).filter(InferenceRun.id == run_id).first()
    if not run or not run.output_path:
        return None

    output_file = Path(run.output_path)
    if not output_file.exists():
        return None

    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if run.annotation:
        ann = run.annotation
        orig = data.get("pose_annotations", {})
        data["pose_annotations"] = {
            "boxes": json.loads(ann.boxes) if ann.boxes else orig.get("boxes", []),
            "keypoints": json.loads(ann.keypoints) if ann.keypoints else orig.get("keypoints", []),
            "kpt_confs": json.loads(ann.kpt_confs) if ann.kpt_confs else orig.get("kpt_confs", []),
            "box_confs": json.loads(ann.box_confs) if ann.box_confs else orig.get("box_confs", []),
        }
        data["user_comments"] = json.loads(ann.comments) if ann.comments else []
        if ann.mask_path and Path(ann.mask_path).exists():
            with open(ann.mask_path, "rb") as mf:
                data["mask_b64"] = base64.b64encode(mf.read()).decode("ascii")

    data["run_id"] = run.id
    data["saved_at"] = (
        run.annotation.updated_at.isoformat()
        if run.annotation and run.annotation.updated_at
        else None
    )
    return data


_SORTABLE_COLUMNS = {
    "id": InferenceRun.id,
    "created_at": InferenceRun.created_at,
    "last_inferred_at": InferenceRun.last_inferred_at,
    "image_name": InferenceRun.image_name,
    "predicted_class": InferenceRun.predicted_class,
    "predicted_species": InferenceRun.predicted_species,
    "predicted_confidence": InferenceRun.predicted_confidence,
    "seg_backend": InferenceRun.seg_backend,
}


def list_runs(
    session: SaSession,
    limit: int = 50,
    offset: int = 0,
    sort_by: str = "last_inferred_at",
    sort_order: str = "desc",
) -> List[Dict[str, Any]]:
    """Return a page of run summaries.

    ``sort_by`` is matched against an allowlist (see ``_SORTABLE_COLUMNS``)
    plus the synthetic ``has_annotations`` column. Anything else falls back
    to ``last_inferred_at``. ``sort_order`` is ``"asc"`` or ``"desc"``
    (default).
    """
    descending = sort_order.lower() != "asc"
    query = session.query(InferenceRun)

    if sort_by == "has_annotations":
        # LEFT JOIN so rows without annotations are still returned. Sort by
        # whether an annotation exists, then by recency as a stable tiebreak.
        query = query.outerjoin(UserAnnotation, InferenceRun.annotation)
        primary = UserAnnotation.id.isnot(None)
        primary = primary.desc() if descending else primary.asc()
        tiebreak = InferenceRun.last_inferred_at.desc()
        query = query.order_by(primary, tiebreak)
    else:
        column = _SORTABLE_COLUMNS.get(sort_by, InferenceRun.last_inferred_at)
        primary = column.desc() if descending else column.asc()
        # Always tiebreak by id desc for deterministic pagination.
        query = query.order_by(primary, InferenceRun.id.desc())

    runs = query.offset(offset).limit(limit).all()
    return [r.to_summary() for r in runs]


def delete_run(session: SaSession, run_id: int) -> bool:
    run = session.query(InferenceRun).filter(InferenceRun.id == run_id).first()
    if not run:
        return False
    if run.output_path:
        Path(run.output_path).unlink(missing_ok=True)
    if run.annotation and run.annotation.mask_path:
        Path(run.annotation.mask_path).unlink(missing_ok=True)
    session.delete(run)
    session.commit()
    return True
