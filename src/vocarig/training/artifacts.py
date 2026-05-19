"""Training artifact naming helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re


DEFAULT_ARTIFACT_DIR = Path("models/trained")
DEFAULT_STEM = "vocarig_lipsync_gru"
RUN_ID_STAMP_RE = re.compile(r"run_(\d{8}-\d{6})_")


@dataclass(frozen=True)
class TrainingArtifactPaths:
    checkpoint: Path
    metrics: Path
    onnx: Path
    timestamp: str


def new_artifact_paths(
    root: Path,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    stem: str = DEFAULT_STEM,
    timestamp: str | None = None,
) -> TrainingArtifactPaths:
    stamp = timestamp or utc_timestamp()
    directory = project_path(root, artifact_dir)
    base = f"{stem}_{stamp}"
    return TrainingArtifactPaths(
        checkpoint=directory / f"{base}.pt",
        metrics=directory / f"{base}_training_metrics.json",
        onnx=directory / f"{base}.onnx",
        timestamp=stamp,
    )


def onnx_path_for_checkpoint(checkpoint_path: str | Path) -> Path:
    return Path(checkpoint_path).with_suffix(".onnx")


def metrics_path_for_checkpoint(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path)
    return path.with_name(f"{path.stem}_training_metrics.json")


def project_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def timestamp_from_run_id(run_id: str | None) -> str | None:
    if not run_id:
        return None
    match = RUN_ID_STAMP_RE.search(str(run_id))
    return match.group(1) if match else None
