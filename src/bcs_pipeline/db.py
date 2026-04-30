"""SQLAlchemy database models and helpers for Body Pawsitive."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import base64

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.orm import Session as SaSession
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

logger = logging.getLogger("bcs_db")

Base = declarative_base()


class InferenceRun(Base):
    __tablename__ = "inference_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
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
    num_pose_detections = Column(Integer, default=0)
    best_pose_conf = Column(Float, nullable=True)
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
            "image_name": self.image_name,
            "source_type": self.source_type,
            "dataset": self.dataset,
            "predicted_class": self.predicted_class,
            "predicted_confidence": round(self.predicted_confidence, 2)
            if self.predicted_confidence
            else None,
            "seg_backend": self.seg_backend,
            "num_pose_detections": self.num_pose_detections,
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


def _migrate_schema(engine) -> None:
    """Add columns added after the initial schema. Idempotent."""
    inspector = inspect(engine)
    if "user_annotations" not in inspector.get_table_names():
        return
    cols = {c["name"] for c in inspector.get_columns("user_annotations")}
    if "mask_path" not in cols:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE user_annotations ADD COLUMN mask_path VARCHAR"))
        logger.info("Migrated user_annotations: added mask_path column")


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
    cls = result.get("classification", {})
    pose = result.get("pose", {})
    img_size = result.get("image_size", [0, 0])

    run = InferenceRun(
        image_name=result.get("image_name", "unknown"),
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
        num_pose_detections=pose.get("num_detections", 0),
        best_pose_conf=pose.get("best_conf"),
    )
    session.add(run)
    session.flush()

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


def list_runs(session: SaSession, limit: int = 50) -> List[Dict[str, Any]]:
    runs = session.query(InferenceRun).order_by(InferenceRun.id.desc()).limit(limit).all()
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
