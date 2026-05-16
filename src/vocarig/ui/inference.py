"""Inference helpers shared by the FastAPI UI and tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from vocarig.audio.features import AudioFeatureConfig, audio_to_feature_windows, load_wav_bytes
from vocarig.models.blendshapes import ARKIT_BLENDSHAPE_NAMES, LIP_BLENDSHAPE_NAMES
from vocarig.runtime.mixer import lip_to_arkit

if TYPE_CHECKING:
    import torch

    from vocarig.models.lipsync_gru import LipSyncGRU


@dataclass(frozen=True)
class ModelLoadResult:
    model: "LipSyncGRU | None"
    checkpoint: dict | None
    message: str

    @property
    def ok(self) -> bool:
        return self.model is not None


@dataclass(frozen=True)
class InferenceStepResult:
    lip: np.ndarray
    hidden: "torch.Tensor"


def default_audio_window(context_frames: int = 11, n_mels: int = 80) -> np.ndarray:
    """Return a zero audio context window."""

    return np.zeros((context_frames, n_mels), dtype=np.float32)


def default_previous_lip() -> np.ndarray:
    """Return neutral previous lip values."""

    return np.zeros(len(LIP_BLENDSHAPE_NAMES), dtype=np.float32)


def default_style_values() -> np.ndarray:
    """Return default style values."""

    return np.asarray([0.5, 0.5], dtype=np.float32)


def build_time_style(
    delta_time: float,
    time_since_audio_update: float,
    energy: float,
    style_values: np.ndarray,
) -> np.ndarray:
    """Build the model's time+style vector."""

    style = _validate_vector("style_values", style_values, 2)
    time_values = np.asarray(
        [
            np.clip(delta_time, 0.0, 1.0),
            np.clip(time_since_audio_update / 0.5, 0.0, 1.0),
            np.clip(energy, 0.0, 1.0),
        ],
        dtype=np.float32,
    )
    return np.concatenate([time_values, style]).astype(np.float32)


_MODEL_CACHE: dict[tuple[str, str], tuple[float, "LipSyncGRU", dict]] = {}


def load_checkpoint_model(path: str | Path, device: str = "auto") -> ModelLoadResult:
    """Load a VocaRig checkpoint with simple mtime caching."""

    checkpoint_path = Path(path).resolve()
    if not checkpoint_path.exists():
        return ModelLoadResult(None, None, f"Model file not found: {checkpoint_path}")

    try:
        import torch

        from vocarig.models.lipsync_gru import LipSyncGRU
        from vocarig.training.precision import resolve_device

        device_str = _normalize_device(device)
        key = (str(checkpoint_path), device_str)
        mtime = checkpoint_path.stat().st_mtime
        if key in _MODEL_CACHE:
            cached_mtime, model, checkpoint = _MODEL_CACHE[key]
            if cached_mtime == mtime:
                return ModelLoadResult(model, checkpoint, f"Loaded model (cached): {checkpoint_path.name}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_config = checkpoint.get("model_config", {})
        if model_config.get("model_type") != "lipsync_gru":
            return ModelLoadResult(None, checkpoint, "Checkpoint is not a VocaRig LipSyncGRU model")
        model = LipSyncGRU(**{key: value for key, value in model_config.items() if key != "model_type"})
        model.load_state_dict(checkpoint["model_state_dict"])
        target_device = resolve_device(device_str)
        model = model.to(target_device).float()
        model.eval()
        _MODEL_CACHE[key] = (mtime, model, checkpoint)
        return ModelLoadResult(model, checkpoint, f"Loaded model: {checkpoint_path.name}")
    except Exception as exc:
        return ModelLoadResult(None, None, f"Could not load model: {exc}")


def run_inference_step(
    model: "LipSyncGRU",
    audio_window: np.ndarray,
    previous_lip: np.ndarray,
    time_style: np.ndarray,
    hidden: "torch.Tensor | None" = None,
    steps_since_reset: int = 0,
) -> InferenceStepResult:
    """Run one streaming inference step."""

    import torch

    audio = np.asarray(audio_window, dtype=np.float32)
    if audio.shape != (model.audio_context_frames, model.n_mels):
        raise ValueError(
            f"audio_window must have shape ({model.audio_context_frames}, {model.n_mels})"
        )
    previous = _validate_vector("previous_lip", previous_lip, model.lip_size)
    ts = _validate_vector("time_style", time_style, model.time_size + model.style_size)
    device = next(model.parameters()).device
    with torch.no_grad():
        audio_t = torch.from_numpy(audio).unsqueeze(0).to(device)
        previous_t = torch.from_numpy(previous).unsqueeze(0).to(device)
        ts_t = torch.from_numpy(ts).unsqueeze(0).to(device)
        if hidden is not None:
            hidden = hidden.to(device)
        result = model(audio_t, previous_t, ts_t, hidden, steps_since_reset=steps_since_reset)
        lip = result.lip.squeeze(0).cpu().numpy().astype(np.float32)
        return InferenceStepResult(lip=lip, hidden=result.hidden.detach())


def run_audio_bytes(
    model: "LipSyncGRU",
    wav_bytes: bytes,
    audio_config: AudioFeatureConfig,
    style_values: np.ndarray | None = None,
    volume_threshold: float = 0.0,
) -> dict:
    """Run a whole WAV file through the same streaming pipeline."""

    samples, sample_rate = load_wav_bytes(wav_bytes)
    windows, times, energies = audio_to_feature_windows(samples, sample_rate, audio_config, volume_threshold)
    style = default_style_values() if style_values is None else _validate_vector("style_values", style_values, 2)
    previous = default_previous_lip()
    hidden = None
    rows = []
    values = []
    for index in range(windows.shape[0]):
        time_style = build_time_style(
            1.0 / float(audio_config.fps),
            0.0,
            float(energies[index]),
            style,
        )
        result = run_inference_step(
            model,
            windows[index],
            previous,
            time_style,
            hidden,
            index,
        )
        previous = result.lip
        hidden = result.hidden
        values.append(result.lip.tolist())
        rows.append(
            {
                "time": float(times[index]),
                "lip_values": result.lip.tolist(),
                "arkit_values": lip_to_arkit(result.lip),
                "confidence": float(np.clip(0.35 + energies[index], 0.0, 1.0)),
            }
        )
    return {
        "fps": audio_config.fps,
        "frame_count": len(rows),
        "lip_values": values,
        "frames": rows,
    }


def lip_rows(values: np.ndarray) -> list[dict[str, float | str]]:
    """Return UI rows for lip values."""

    lip = _validate_vector("lip", values, len(LIP_BLENDSHAPE_NAMES))
    return [{"control": name, "value": float(lip[index])} for index, name in enumerate(LIP_BLENDSHAPE_NAMES)]


def arkit_rows(values: dict[str, float]) -> list[dict[str, float | str]]:
    """Return UI rows for ARKit values."""

    return [{"control": name, "value": float(values.get(name, 0.0))} for name in ARKIT_BLENDSHAPE_NAMES]


def _normalize_device(device: str) -> str:
    normalized = str(device or "auto").lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    return normalized


def _validate_vector(name: str, values: np.ndarray, expected_size: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (expected_size,):
        raise ValueError(f"{name} must have shape ({expected_size},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return np.clip(array, 0.0, 1.0).astype(np.float32)
