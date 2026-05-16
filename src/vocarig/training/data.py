"""Dataset loading helpers for VocaRig training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset


@dataclass(frozen=True)
class VocaRigDatasetBundle:
    dataset: TensorDataset
    audio_windows: torch.Tensor
    previous_lip: torch.Tensor
    time_style: torch.Tensor
    y: torch.Tensor
    utterance_ids: np.ndarray
    frame_ids: np.ndarray
    metadata: dict


def load_npz_dataset(
    path: str | Path,
    context_frames: int = 11,
    n_mels: int = 80,
    lip_size: int = 21,
    time_size: int = 3,
    style_size: int = 2,
) -> VocaRigDatasetBundle:
    """Load frame-level arrays from a `.npz` dataset."""

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with np.load(dataset_path, allow_pickle=False) as data:
        required = {"audio_windows", "previous_lip", "time_values", "style_values", "y"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"Dataset missing arrays: {', '.join(missing)}")
        audio = np.asarray(data["audio_windows"], dtype=np.float32)
        previous = np.asarray(data["previous_lip"], dtype=np.float32)
        time_values = np.asarray(data["time_values"], dtype=np.float32)
        style_values = np.asarray(data["style_values"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.float32)
        utterance_ids = (
            np.asarray(data["utterance_ids"], dtype=np.int64)
            if "utterance_ids" in data
            else np.zeros(y.shape[0], dtype=np.int64)
        )
        frame_ids = (
            np.asarray(data["frame_ids"], dtype=np.int64)
            if "frame_ids" in data
            else np.arange(y.shape[0], dtype=np.int64)
        )
        metadata = {"path": str(dataset_path), "keys": list(data.files)}
        if "lip_names" in data:
            metadata["lip_names"] = [str(item) for item in data["lip_names"].tolist()]
        if "arkit_names" in data:
            metadata["arkit_names"] = [str(item) for item in data["arkit_names"].tolist()]

    if audio.ndim != 3 or audio.shape[1:] != (context_frames, n_mels):
        raise ValueError(f"audio_windows must have shape (N, {context_frames}, {n_mels})")
    _validate_2d("previous_lip", previous, lip_size)
    _validate_2d("time_values", time_values, time_size)
    _validate_2d("style_values", style_values, style_size)
    _validate_2d("y", y, lip_size)
    row_count = y.shape[0]
    for name, array in {
        "audio_windows": audio,
        "previous_lip": previous,
        "time_values": time_values,
        "style_values": style_values,
    }.items():
        if array.shape[0] != row_count:
            raise ValueError(f"{name}/y row mismatch")
    time_style = np.concatenate([time_values, style_values], axis=1).astype(np.float32)

    audio_t = torch.from_numpy(audio)
    previous_t = torch.from_numpy(previous)
    time_style_t = torch.from_numpy(time_style)
    y_t = torch.from_numpy(y)
    return VocaRigDatasetBundle(
        dataset=TensorDataset(audio_t, previous_t, time_style_t, y_t),
        audio_windows=audio_t,
        previous_lip=previous_t,
        time_style=time_style_t,
        y=y_t,
        utterance_ids=utterance_ids,
        frame_ids=frame_ids,
        metadata=metadata,
    )


def load_sequence_npz_dataset(
    path: str | Path,
    context_frames: int = 11,
    n_mels: int = 80,
    lip_size: int = 21,
    time_size: int = 3,
    style_size: int = 2,
    sequence_window: int = 64,
    sequence_stride: int = 32,
) -> VocaRigDatasetBundle:
    """Load fixed-length sequence windows from a `.npz` dataset."""

    if sequence_window < 2:
        raise ValueError("sequence_window must be at least 2")
    if sequence_stride < 1:
        raise ValueError("sequence_stride must be at least 1")
    base = load_npz_dataset(path, context_frames, n_mels, lip_size, time_size, style_size)
    audio = base.audio_windows.numpy()
    previous = base.previous_lip.numpy()
    time_style = base.time_style.numpy()
    y = base.y.numpy()
    order = np.lexsort((base.frame_ids, base.utterance_ids))
    utterance_sorted = base.utterance_ids[order]

    audio_windows: list[np.ndarray] = []
    previous_windows: list[np.ndarray] = []
    time_windows: list[np.ndarray] = []
    y_windows: list[np.ndarray] = []
    for utterance_id in np.unique(utterance_sorted):
        indices = order[np.flatnonzero(utterance_sorted == utterance_id)]
        if len(indices) < sequence_window:
            continue
        for start in range(0, len(indices) - sequence_window + 1, sequence_stride):
            window = indices[start : start + sequence_window]
            audio_windows.append(audio[window])
            previous_windows.append(previous[window])
            time_windows.append(time_style[window])
            y_windows.append(y[window])
    if not audio_windows:
        raise ValueError("Dataset does not contain any usable sequence windows")

    audio_t = torch.from_numpy(np.asarray(audio_windows, dtype=np.float32))
    previous_t = torch.from_numpy(np.asarray(previous_windows, dtype=np.float32))
    time_t = torch.from_numpy(np.asarray(time_windows, dtype=np.float32))
    y_t = torch.from_numpy(np.asarray(y_windows, dtype=np.float32))
    metadata = {
        **base.metadata,
        "window_count": int(y_t.shape[0]),
        "sequence_count": int(len(np.unique(utterance_sorted))),
        "sequence_window": sequence_window,
        "sequence_stride": sequence_stride,
    }
    return VocaRigDatasetBundle(
        dataset=TensorDataset(audio_t, previous_t, time_t, y_t),
        audio_windows=audio_t,
        previous_lip=previous_t,
        time_style=time_t,
        y=y_t,
        utterance_ids=base.utterance_ids,
        frame_ids=base.frame_ids,
        metadata=metadata,
    )


def split_indices(size: int, validation_split: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic train/validation window indices."""

    if size < 2:
        raise ValueError("Dataset must contain at least 2 windows")
    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split must be between 0 and 1")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(size, generator=generator)
    validation_size = max(1, int(round(size * validation_split)))
    validation_size = min(validation_size, size - 1)
    return indices[validation_size:], indices[:validation_size]


def _validate_2d(name: str, values: np.ndarray, width: int) -> None:
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(f"{name} must have shape (N, {width}), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinite values")
