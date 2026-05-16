"""Lightweight runtime model worker for VocaRig UI."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np

from vocarig.models.blendshapes import LIP_BLENDSHAPE_NAMES
from vocarig.runtime.mixer import lip_to_arkit
from vocarig.ui.inference import (
    build_time_style,
    default_audio_window,
    default_previous_lip,
    default_style_values,
    load_checkpoint_model,
    run_inference_step,
)


HIDDEN_STATE_TTL_SECONDS = 10.0


class ModelWorker:
    """Caches model and per-stream hidden state."""

    def __init__(self) -> None:
        self.hidden_states: dict[tuple[str, str, str], dict[str, Any]] = {}

    def device_status(self, device: str) -> dict:
        try:
            import torch

            from vocarig.training.precision import resolve_device

            effective = str(resolve_device(device))
            cuda_available = bool(torch.cuda.is_available())
        except Exception as exc:
            effective = f"error: {exc}"
            cuda_available = False
        return {
            "ok": True,
            "selected": device,
            "effective": effective,
            "cuda_available": cuda_available,
            "modes": ["auto", "cpu", "cuda"],
        }

    def infer(
        self,
        checkpoint_path: str,
        audio_window: Any,
        previous_lip: Any,
        delta_time: float,
        time_since_audio_update: float,
        energy: float,
        style_values: Any,
        device: str,
        stream_id: str,
        reset_state: bool = False,
    ) -> dict:
        start = time.perf_counter()
        load_result = load_checkpoint_model(checkpoint_path, device)
        if not load_result.ok or load_result.model is None:
            return {"ok": False, "error": load_result.message}

        model = load_result.model
        audio = (
            default_audio_window(model.audio_context_frames, model.n_mels)
            if audio_window is None
            else np.asarray(audio_window, dtype=np.float32)
        )
        previous = (
            default_previous_lip()
            if previous_lip is None
            else np.asarray(previous_lip, dtype=np.float32)
        )
        style = default_style_values() if style_values is None else np.asarray(style_values, dtype=np.float32)
        time_style = build_time_style(delta_time, time_since_audio_update, energy, style)

        key = (str(Path(checkpoint_path).resolve()), str(device), str(stream_id))
        now = time.perf_counter()
        state = self.hidden_states.get(key)
        expired = bool(state and now - float(state["last_seen"]) > HIDDEN_STATE_TTL_SECONDS)
        checkpoint_mtime = Path(checkpoint_path).resolve().stat().st_mtime
        if reset_state or expired or (state and state["checkpoint_mtime"] != checkpoint_mtime):
            self.hidden_states.pop(key, None)
            state = None

        result = run_inference_step(
            model,
            audio,
            previous,
            time_style,
            None if state is None else state["hidden"],
            0 if state is None else int(state["steps_since_reset"]),
        )
        self.hidden_states[key] = {
            "hidden": result.hidden,
            "last_seen": now,
            "checkpoint_mtime": checkpoint_mtime,
            "steps_since_reset": 1 if state is None else int(state["steps_since_reset"]) + 1,
        }
        elapsed = (time.perf_counter() - start) * 1000.0
        return {
            "ok": True,
            "lip_values": result.lip.tolist(),
            "arkit_values": lip_to_arkit(result.lip),
            "message": load_result.message,
            "infer_ms": round(elapsed, 3),
            "device": str(next(model.parameters()).device),
            "control_count": len(LIP_BLENDSHAPE_NAMES),
        }

    def benchmark(
        self,
        checkpoint_path: str,
        device: str,
        iterations: int = 1000,
        warmup: int = 50,
    ) -> dict:
        load_result = load_checkpoint_model(checkpoint_path, device)
        if not load_result.ok or load_result.model is None:
            raise RuntimeError(load_result.message)
        model = load_result.model
        audio = default_audio_window(model.audio_context_frames, model.n_mels)
        previous = default_previous_lip()
        style = default_style_values()
        time_style = build_time_style(1.0 / 30.0, 0.0, 0.0, style)
        hidden = None
        for step in range(max(0, warmup)):
            result = run_inference_step(model, audio, previous, time_style, hidden, step)
            hidden = result.hidden
            previous = result.lip
        start = time.perf_counter()
        for step in range(max(1, iterations)):
            result = run_inference_step(model, audio, previous, time_style, hidden, step)
            hidden = result.hidden
            previous = result.lip
        wall_ms = (time.perf_counter() - start) * 1000.0
        avg = wall_ms / max(1, iterations)
        return {
            "ok": True,
            "iterations": iterations,
            "warmup": warmup,
            "wall_ms_total": round(wall_ms, 3),
            "avg_wall_ms": round(avg, 4),
            "throughput_fps": round(1000.0 / max(avg, 1e-9), 1),
            "device": str(next(model.parameters()).device),
        }
