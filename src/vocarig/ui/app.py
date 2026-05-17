"""FastAPI UI server for VocaRig Lab."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import sys
import threading
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import yaml

from vocarig.audio.features import AudioFeatureConfig, audio_to_feature_windows, resample_mono
from vocarig.models.blendshapes import ARKIT_BLENDSHAPE_NAMES, LIP_BLENDSHAPE_NAMES
from vocarig.runtime.mixer import arkit_dict_to_vector
from vocarig.synthetic.dataset import SyntheticGenerationConfig, generate_dataset, save_dataset
from vocarig.training.artifacts import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_STEM,
    metrics_path_for_checkpoint,
    new_artifact_paths,
    onnx_path_for_checkpoint,
)
from vocarig.training.export_onnx import export_onnx
from vocarig.training.train import TrainingRunConfig, list_training_checkpoints, train_model
from vocarig.ui.inference import (
    default_audio_window,
    default_previous_lip,
    lip_rows,
    load_checkpoint_model,
    run_audio_bytes,
)
from vocarig.ui.model_worker import ModelWorker


CONFIG_PATH = ROOT / "configs" / "train_config.yaml"
STATIC_DIR = Path(__file__).resolve().parent / "static"
DATA_DIR = ROOT / "data"
MESH_DIR = DATA_DIR / "mesh"
AUDIO_DIR = DATA_DIR / "audio"


class InferenceRequest(BaseModel):
    stream_id: str = "preview"
    audio_window: list[list[float]] | None = None
    previous_lip: list[float] = Field(default_factory=lambda: default_previous_lip().tolist())
    delta_time: float = 1.0 / 30.0
    time_since_audio_update: float = 0.0
    energy: float = 0.0
    style_values: list[float] = Field(default_factory=lambda: [0.5, 0.5])
    reset_state: bool = False


class AudioInferenceRequest(BaseModel):
    style_values: list[float] = Field(default_factory=lambda: [0.5, 0.5])


class RealtimeAudioRequest(BaseModel):
    stream_id: str = "realtime"
    samples: list[float] = Field(default_factory=list)
    sample_rate: int = Field(default=48_000, ge=8_000, le=192_000)
    previous_lip: list[float] = Field(default_factory=lambda: default_previous_lip().tolist())
    delta_time: float = 1.0 / 30.0
    style_values: list[float] = Field(default_factory=lambda: [0.5, 0.5])
    volume_threshold: float = Field(default=0.015, ge=0.0, le=1.0)
    reset_state: bool = False


class TrainingRequest(BaseModel):
    data_path: str | None = None
    epochs: int | None = Field(default=None, ge=1)
    batch_size: int | None = Field(default=None, ge=1)
    learning_rate: float | None = Field(default=None, gt=0)
    weight_decay: float | None = Field(default=None, ge=0)
    validation_split: float | None = Field(default=None, ge=0, lt=1)
    sequence_window: int | None = Field(default=None, ge=2)
    sequence_stride: int | None = Field(default=None, ge=1)
    checkpoint_interval: int | None = Field(default=None, ge=0)
    metric_eval_interval: int | None = Field(default=None, ge=1)
    precision: str | None = None
    resume_checkpoint: str | None = None
    final_teacher_forcing_ratio: float | None = Field(default=None, ge=0, le=1)
    teacher_decay_start_epoch: int | None = Field(default=None, ge=1)
    teacher_decay_epochs: int | None = Field(default=None, ge=1)
    warmup_loss_steps: int | None = Field(default=None, ge=0)
    early_stopping_patience: int | None = Field(default=None, ge=1)
    early_stopping_min_delta: float | None = Field(default=None, ge=0)
    early_stopping_min_epochs: int | None = Field(default=None, ge=1)
    target_val_loss: float | None = Field(default=None, ge=0)
    target_train_loss: float | None = Field(default=None, ge=0)
    divergence_loss: float | None = Field(default=None, gt=0)
    overfit_gap_ratio: float | None = Field(default=None, gt=1)
    stop_on_target_val_loss: bool | None = None
    stop_on_target_train_loss: bool | None = None
    stop_on_divergence_loss: bool | None = None
    stop_on_plateau: bool | None = None
    stop_on_overfit_gap: bool | None = None
    pose_loss_weight: float | None = Field(default=None, ge=0)
    delta_loss_weight: float | None = Field(default=None, ge=0)
    velocity_loss_weight: float | None = Field(default=None, ge=0)
    jerk_loss_weight: float | None = Field(default=None, ge=0)
    silence_loss_weight: float | None = Field(default=None, ge=0)
    range_loss_weight: float | None = Field(default=None, ge=0)


class SyntheticRequest(BaseModel):
    utterances: int | None = Field(default=None, ge=1)
    seed: int | None = Field(default=None, ge=0)
    min_phonemes: int | None = Field(default=None, ge=1)
    max_phonemes: int | None = Field(default=None, ge=1)
    silence_probability: float | None = Field(default=None, ge=0, le=1)
    tr_probability: float | None = Field(default=None, ge=0, le=1)


class DeviceRequest(BaseModel):
    device: str


class ModelSelectionRequest(BaseModel):
    path: str


class BenchmarkRequest(BaseModel):
    iterations: int = Field(default=1000, ge=1, le=10000)
    warmup: int = Field(default=50, ge=0, le=1000)


class ExportRequest(BaseModel):
    opset_version: int | None = None


class SyntheticJobState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.thread: threading.Thread | None = None
        self.status = "idle"
        self.progress = 0
        self.total = 0
        self.phase = ""
        self.error: str | None = None
        self.frames = 0

    def running(self) -> bool:
        return bool(self.thread and self.thread.is_alive())

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "status": self.status,
                "running": self.running(),
                "progress": self.progress,
                "total": self.total,
                "phase": self.phase,
                "error": self.error,
                "frames": self.frames,
            }

synthetic_job = SyntheticJobState()

class TrainingJobState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.thread: threading.Thread | None = None
        self.stop_event: threading.Event | None = None
        self.events: list[dict] = []
        self.metrics: dict | None = None
        self.error: str | None = None
        self.status = "idle"
        self.config: Any | None = None

    def running(self) -> bool:
        return bool(self.thread and self.thread.is_alive())

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "status": self.status,
                "running": self.running(),
                "events": list(self.events),
                "metrics": self.metrics,
                "error": self.error,
                "config": _training_config_payload(self.config),
            }


training_job = TrainingJobState()
realtime_audio_lock = threading.RLock()
realtime_audio_buffers: dict[str, np.ndarray] = {}
REALTIME_AUDIO_BUFFER_SECONDS = 2.0

app = FastAPI(title="VocaRig Lab", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
if MESH_DIR.exists():
    app.mount("/mesh", StaticFiles(directory=MESH_DIR), name="mesh")
if AUDIO_DIR.exists():
    app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")
app.state.model_worker = ModelWorker()
app.state.device_mode = "auto"
app.state.model_path = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status() -> dict:
    config = _load_config()
    paths = _paths_from_config(config)
    active_checkpoint = selected_model_path()
    paths["checkpoint"] = active_checkpoint
    paths["onnx"] = onnx_path_for_checkpoint(active_checkpoint)
    paths["metrics"] = metrics_path_for_checkpoint(active_checkpoint)
    return {
        "lip_names": LIP_BLENDSHAPE_NAMES,
        "arkit_names": ARKIT_BLENDSHAPE_NAMES,
        "paths": {key: str(value) for key, value in paths.items()},
        "files": {key: value.exists() for key, value in paths.items()},
        "device": _device_payload(),
        "model": _model_payload(),
        "audio": _audio_payload(config),
    }


@app.get("/api/datasets")
def datasets() -> dict:
    return _dataset_payload()


@app.get("/api/device")
def get_device() -> dict:
    return _device_payload()


@app.post("/api/device")
def set_device(request: DeviceRequest) -> dict:
    mode = _normalize_device(request.device)
    if mode == "cuda":
        probe = app.state.model_worker.device_status("cuda")
        if str(probe.get("effective", "")).startswith("error:"):
            raise HTTPException(status_code=400, detail=probe.get("effective", "CUDA not available"))
    app.state.device_mode = mode
    return _device_payload()


@app.get("/api/models")
def get_models() -> dict:
    return _model_payload()


@app.post("/api/models/select")
def select_model(request: ModelSelectionRequest) -> dict:
    path = _project_path(request.path)
    models_root = ROOT / "models"
    try:
        path.resolve().relative_to(models_root.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Model must be inside project models directory") from exc
    if path.suffix.lower() != ".pt":
        raise HTTPException(status_code=400, detail="Model must be a .pt checkpoint")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found: {path}")
    app.state.model_path = str(path)
    return _model_payload()


@app.post("/api/infer")
def infer(request: InferenceRequest) -> dict:
    checkpoint = selected_model_path()
    if not checkpoint.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found: {checkpoint}")
    config = _load_config()
    audio_config = _audio_config(config)
    audio_window = request.audio_window
    if audio_window is None:
        audio_window = default_audio_window(audio_config.context_frames, audio_config.n_mels).tolist()
    result = app.state.model_worker.infer(
        str(checkpoint),
        audio_window,
        request.previous_lip,
        request.delta_time,
        request.time_since_audio_update,
        request.energy,
        request.style_values,
        app.state.device_mode,
        request.stream_id,
        request.reset_state,
    )
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=result.get("error", "Inference failed"))
    lip = np.asarray(result["lip_values"], dtype=np.float32)
    return {
        "lip_values": result["lip_values"],
        "lip_rows": lip_rows(lip),
        "arkit_values": result["arkit_values"],
        "arkit_vector": arkit_dict_to_vector(result["arkit_values"]).tolist(),
        "model_message": result["message"],
        "latency_ms": result["infer_ms"],
        "telemetry": {"device": result.get("device", "cpu"), "infer_ms": result["infer_ms"]},
    }


@app.post("/api/infer/audio")
async def infer_audio(
    file: UploadFile = File(...),
    style_values: str | None = Form(None),
    volume_threshold: float = Form(0.015),
) -> dict:
    checkpoint = selected_model_path()
    if not checkpoint.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found: {checkpoint}")
    load_result = load_checkpoint_model(checkpoint, app.state.device_mode)
    if not load_result.ok or load_result.model is None:
        raise HTTPException(status_code=500, detail=load_result.message)
    try:
        style = np.asarray(json.loads(style_values), dtype=np.float32) if style_values else np.asarray([0.5, 0.5], dtype=np.float32)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid style_values: {exc}") from exc
    content = await file.read()
    try:
        start = time.perf_counter()
        threshold = float(np.clip(volume_threshold, 0.0, 1.0))
        payload = run_audio_bytes(load_result.model, content, _audio_config(_load_config()), style, threshold)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    frame_count = max(1, int(payload.get("frame_count") or 1))
    payload["latency_ms"] = round(elapsed_ms, 3)
    payload["infer_ms"] = round(elapsed_ms / frame_count, 3)
    payload["volume_threshold"] = threshold
    payload["model_message"] = load_result.message
    return payload


@app.post("/api/infer/realtime")
def infer_realtime(request: RealtimeAudioRequest) -> dict:
    checkpoint = selected_model_path()
    if not checkpoint.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found: {checkpoint}")
    config = _load_config()
    audio_config = _audio_config(config)
    audio_window, energy = _realtime_audio_window(request, audio_config)
    result = app.state.model_worker.infer(
        str(checkpoint),
        audio_window.tolist(),
        request.previous_lip,
        request.delta_time,
        0.0 if request.samples else request.delta_time,
        energy,
        request.style_values,
        app.state.device_mode,
        request.stream_id,
        request.reset_state,
    )
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=result.get("error", "Inference failed"))
    lip = np.asarray(result["lip_values"], dtype=np.float32)
    return {
        "lip_values": result["lip_values"],
        "lip_rows": lip_rows(lip),
        "arkit_values": result["arkit_values"],
        "arkit_vector": arkit_dict_to_vector(result["arkit_values"]).tolist(),
        "model_message": result["message"],
        "latency_ms": result["infer_ms"],
        "telemetry": {"device": result.get("device", "cpu"), "infer_ms": result["infer_ms"]},
        "energy": energy,
        "volume_threshold": request.volume_threshold,
        "mode": "realtime",
    }


def _realtime_audio_window(
    request: RealtimeAudioRequest,
    audio_config: AudioFeatureConfig,
) -> tuple[np.ndarray, float]:
    samples = np.asarray(request.samples, dtype=np.float32).reshape(-1)
    if samples.size:
        samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
        samples = np.clip(samples, -1.0, 1.0)
        samples = resample_mono(samples, request.sample_rate, audio_config.sample_rate)

    with realtime_audio_lock:
        if request.reset_state:
            realtime_audio_buffers.pop(request.stream_id, None)
        previous = realtime_audio_buffers.get(request.stream_id)
        if previous is None:
            previous = np.zeros(0, dtype=np.float32)
        if samples.size:
            buffer = np.concatenate([previous, samples.astype(np.float32)])
        else:
            buffer = previous
        if buffer.size == 0:
            buffer = np.zeros(audio_config.window_size, dtype=np.float32)
        keep = max(
            int(audio_config.sample_rate * REALTIME_AUDIO_BUFFER_SECONDS),
            audio_config.window_size + audio_config.context_frames * audio_config.hop_size,
        )
        buffer = buffer[-keep:].astype(np.float32, copy=False)
        realtime_audio_buffers[request.stream_id] = buffer

    windows, _, energies = audio_to_feature_windows(
        buffer,
        audio_config.sample_rate,
        audio_config,
        request.volume_threshold,
    )
    return windows[-1], float(energies[-1]) if energies.size else 0.0


@app.websocket("/ws/stream")
async def stream_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_json()
            try:
                request = InferenceRequest(**payload)
                result = await run_in_threadpool(infer, request)
                await websocket.send_json({"ok": True, **result})
            except Exception as exc:
                await websocket.send_json({"ok": False, "error": str(exc)})
    except WebSocketDisconnect:
        return


@app.websocket("/ws/infer")
async def infer_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_json()
            request_id = payload.get("request_id") if isinstance(payload, dict) else None
            try:
                request = RealtimeAudioRequest(**payload)
                result = await run_in_threadpool(infer_realtime, request)
                response = {"ok": True, **result}
                if request_id is not None:
                    response["request_id"] = request_id
                await websocket.send_json(response)
            except HTTPException as exc:
                response = {"ok": False, "error": str(exc.detail)}
                if request_id is not None:
                    response["request_id"] = request_id
                await websocket.send_json(response)
            except Exception as exc:
                response = {"ok": False, "error": str(exc)}
                if request_id is not None:
                    response["request_id"] = request_id
                await websocket.send_json(response)
    except WebSocketDisconnect:
        return


@app.websocket("/ws/train")
async def train_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(training_job.snapshot())
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return


@app.websocket("/ws/synthetic")
async def synthetic_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(synthetic_job.snapshot())
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return


@app.get("/api/data")
def data_info(path: str | None = None) -> dict:
    paths = _paths_from_config(_load_config())
    data_path = _resolve_dataset_path(path, paths["data"]) if path else paths["data"]
    metadata_path = _metadata_path_for_dataset(data_path, paths["synthetic_metadata"])
    payload: dict[str, Any] = {
        "data_path": str(data_path),
        "metadata_path": str(metadata_path),
        "data_exists": data_path.exists(),
        "metadata_exists": metadata_path.exists(),
    }
    if data_path.exists():
        with np.load(data_path, allow_pickle=False) as data:
            payload["audio_windows_shape"] = list(data["audio_windows"].shape)
            payload["y_shape"] = list(data["y"].shape)
            payload["lip_names"] = [str(item) for item in data["lip_names"].tolist()]
    if metadata_path.exists():
        payload["metadata"] = _read_json(metadata_path)
    return payload


@app.post("/api/synthetic/generate")
def generate_synthetic(request: SyntheticRequest) -> dict:
    with synthetic_job.lock:
        if synthetic_job.running():
            raise HTTPException(status_code=400, detail="Synthetic generation already running")
            
        synthetic_job.status = "starting"
        synthetic_job.progress = 0
        synthetic_job.total = 0
        synthetic_job.phase = "initializing"
        synthetic_job.error = None
        synthetic_job.frames = 0
        
    def _run_synthesis():
        try:
            config = _load_config()
            paths = _paths_from_config(config)
            audio_cfg = _audio_config(config)
            synthetic_cfg = config.get("synthetic", {})
            language_mix = synthetic_cfg.get("language_mix", {}) or {}
            generation_config = SyntheticGenerationConfig(
                utterance_count=int(request.utterances or synthetic_cfg.get("utterance_count", 4000)),
                seed=int(request.seed if request.seed is not None else synthetic_cfg.get("seed", 42)),
                sample_rate=audio_cfg.sample_rate,
                fps=audio_cfg.fps,
                min_phonemes=int(request.min_phonemes or synthetic_cfg.get("min_phonemes", 18)),
                max_phonemes=int(request.max_phonemes or synthetic_cfg.get("max_phonemes", 56)),
                silence_probability=float(
                    request.silence_probability
                    if request.silence_probability is not None
                    else synthetic_cfg.get("silence_probability", 0.18)
                ),
                tr_probability=float(
                    request.tr_probability if request.tr_probability is not None else language_mix.get("tr", 0.55)
                ),
                audio=audio_cfg,
            )
            if generation_config.min_phonemes > generation_config.max_phonemes:
                raise ValueError("min_phonemes must be <= max_phonemes")
                
            def progress_cb(current, total, phase):
                with synthetic_job.lock:
                    synthetic_job.progress = current
                    synthetic_job.total = total
                    synthetic_job.phase = phase
                    synthetic_job.status = "running"
            
            dataset = generate_dataset(generation_config, progress_callback=progress_cb)
            
            with synthetic_job.lock:
                synthetic_job.phase = "saving..."
            
            save_dataset(dataset, paths["data"], paths["synthetic_metadata"], generation_config)
            
            with synthetic_job.lock:
                synthetic_job.status = "completed"
                synthetic_job.phase = "done"
                synthetic_job.frames = int(dataset.y.shape[0])
                
        except Exception as exc:
            traceback.print_exc()
            with synthetic_job.lock:
                synthetic_job.status = "error"
                synthetic_job.error = str(exc)

    synthetic_job.thread = threading.Thread(target=_run_synthesis, daemon=True)
    synthetic_job.thread.start()
    return {"ok": True}


@app.get("/api/metrics")
def metrics(path: str | None = None) -> dict:
    if path:
        metrics_path = _project_path(path)
    else:
        metrics_path = _default_metrics_path()
    if not metrics_path.exists():
        return {"exists": False, "path": str(metrics_path)}
    payload = _read_json(metrics_path)
    payload["exists"] = True
    payload["path"] = str(metrics_path)
    return payload


@app.get("/api/metrics/options")
def metrics_options() -> dict:
    selected = _default_metrics_path()
    root = ROOT / "models"
    options = []
    if root.exists():
        for path in sorted(root.glob("**/*_training_metrics.json")):
            options.append(_metrics_option(path, selected=path.resolve() == selected.resolve()))
    return {"selected": str(selected), "options": options}


@app.get("/api/diagnosis")
def diagnosis(path: str | None = None) -> dict:
    if path:
        metrics_path = _project_path(path)
    else:
        metrics_path = _default_metrics_path()
    if not metrics_path.exists():
        return {"overall_status": "no_data", "summary": "No training data available.", "diagnoses": [], "metrics": {}}
    data = _read_json(metrics_path)
    history = data.get("history", [])
    final_val = data.get("final_val_loss")
    best_val = data.get("best_val_loss")
    status = "healthy" if final_val is not None and best_val is not None else "insufficient_data"
    return {
        "overall_status": status,
        "summary": f"epochs={data.get('completed_epochs', 0)} best_val={best_val}",
        "diagnoses": [],
        "metrics": {
            "epoch_count": len(history),
            "final_val_loss": final_val,
            "best_val_loss": best_val,
            "precision": data.get("precision"),
            "device": data.get("device"),
        },
    }


@app.get("/api/train/status")
def train_status() -> dict:
    return training_job.snapshot()


@app.get("/api/train/checkpoints")
def train_checkpoints() -> dict:
    paths = _paths_from_config(_load_config())
    return list_training_checkpoints(paths["checkpoint_dir"])


@app.post("/api/train/start")
def train_start(request: TrainingRequest) -> dict:
    if training_job.running():
        raise HTTPException(status_code=409, detail="Training is already running")
    config_data = _load_config()
    paths = _paths_from_config(config_data)
    model_config = config_data.get("model", {})
    training_config = config_data.get("training", {})
    export_config = config_data.get("export", {})
    data_path = _resolve_dataset_path(request.data_path, paths["data"])
    artifact_paths = new_artifact_paths(
        ROOT,
        export_config.get("artifact_dir", DEFAULT_ARTIFACT_DIR),
        export_config.get("checkpoint_prefix", DEFAULT_STEM),
    )
    run_config = TrainingRunConfig(
        data_path=data_path,
        checkpoint_path=artifact_paths.checkpoint,
        metrics_path=artifact_paths.metrics,
        checkpoint_dir=paths["checkpoint_dir"],
        resume_checkpoint=Path(request.resume_checkpoint) if request.resume_checkpoint else None,
        checkpoint_interval=_request_config_value(request, "checkpoint_interval", training_config, 25, int),
        audio_context_frames=int(model_config.get("audio_context_frames", 11)),
        n_mels=int(model_config.get("n_mels", 80)),
        lip_size=int(model_config.get("lip_size", 21)),
        time_size=int(model_config.get("time_size", 3)),
        style_size=int(model_config.get("style_size", 2)),
        hidden_size=int(model_config.get("hidden_size", 128)),
        audio_channels=int(model_config.get("audio_channels", 64)),
        max_step=float(model_config.get("max_step", 0.12)),
        reference_dt=float(model_config.get("reference_dt", 1.0 / 30.0)),
        warmup_steps=int(model_config.get("warmup_steps", 3)),
        sequence_window=_request_config_value(request, "sequence_window", training_config, 96, int),
        sequence_stride=_request_config_value(request, "sequence_stride", training_config, 24, int),
        epochs=_request_config_value(request, "epochs", training_config, 250, int),
        batch_size=_request_config_value(request, "batch_size", training_config, 64, int),
        learning_rate=_request_config_value(request, "learning_rate", training_config, 0.00035, float),
        weight_decay=_request_config_value(request, "weight_decay", training_config, 0.0001, float),
        validation_split=_request_config_value(request, "validation_split", training_config, 0.1, float),
        device=app.state.device_mode,
        precision=str(request.precision or training_config.get("precision", "fp32")),
        allow_tf32=bool(training_config.get("allow_tf32", True)),
        amp_dtype=str(training_config.get("amp_dtype", "fp16")),
        num_workers=int(training_config.get("num_workers", 0)),
        log_interval=int(training_config.get("log_interval", 20)),
        metric_eval_interval=_request_config_value(request, "metric_eval_interval", training_config, 1, int),
        seed=int(training_config.get("seed", 42)),
        final_teacher_forcing_ratio=_request_config_value(
            request, "final_teacher_forcing_ratio", training_config, 0.0, float
        ),
        teacher_decay_start_epoch=_request_config_value(request, "teacher_decay_start_epoch", training_config, 25, int),
        teacher_decay_epochs=_request_config_value(request, "teacher_decay_epochs", training_config, 100, int),
        warmup_loss_steps=_request_config_value(request, "warmup_loss_steps", training_config, 6, int),
        early_stopping_patience=_request_config_value(request, "early_stopping_patience", training_config, 40, int),
        early_stopping_min_delta=_request_config_value(
            request, "early_stopping_min_delta", training_config, 0.00001, float
        ),
        early_stopping_min_epochs=_request_config_value(request, "early_stopping_min_epochs", training_config, 100, int),
        target_val_loss=_request_config_value(request, "target_val_loss", training_config, 0.0025, float),
        target_train_loss=_request_config_value(request, "target_train_loss", training_config, 0.0012, float),
        divergence_loss=_request_config_value(request, "divergence_loss", training_config, 0.25, float),
        overfit_gap_ratio=_request_config_value(request, "overfit_gap_ratio", training_config, 4.0, float),
        stop_on_target_val_loss=_request_config_value(
            request, "stop_on_target_val_loss", training_config, False, bool
        ),
        stop_on_target_train_loss=_request_config_value(
            request, "stop_on_target_train_loss", training_config, False, bool
        ),
        stop_on_divergence_loss=_request_config_value(
            request, "stop_on_divergence_loss", training_config, True, bool
        ),
        stop_on_plateau=_request_config_value(request, "stop_on_plateau", training_config, True, bool),
        stop_on_overfit_gap=_request_config_value(request, "stop_on_overfit_gap", training_config, False, bool),
        pose_loss_weight=_request_config_value(request, "pose_loss_weight", training_config, 1.0, float),
        delta_loss_weight=_request_config_value(request, "delta_loss_weight", training_config, 0.0, float),
        velocity_loss_weight=_request_config_value(request, "velocity_loss_weight", training_config, 0.12, float),
        jerk_loss_weight=_request_config_value(request, "jerk_loss_weight", training_config, 0.04, float),
        silence_loss_weight=_request_config_value(request, "silence_loss_weight", training_config, 0.25, float),
        range_loss_weight=_request_config_value(request, "range_loss_weight", training_config, 0.02, float),
    )
    _start_training_job(run_config)
    return {"ok": True}


@app.post("/api/train/stop")
def train_stop() -> dict:
    with training_job.lock:
        if training_job.stop_event is not None:
            training_job.stop_event.set()
        if training_job.running():
            training_job.status = "stopping"
    return {"ok": True}


@app.post("/api/train/abort")
def train_abort() -> dict:
    return train_stop()


@app.post("/api/train/resume")
def train_resume() -> dict:
    with training_job.lock:
        if training_job.running():
            training_job.status = "running"
    return {"ok": True}


@app.post("/api/train/clear")
def train_clear() -> dict:
    with training_job.lock:
        if training_job.running():
            raise HTTPException(status_code=409, detail="Training is running")
        training_job.events = []
        training_job.metrics = None
        training_job.error = None
        training_job.status = "idle"
        training_job.config = None
        training_job.stop_event = None
    return {"ok": True}


@app.post("/api/export")
def export(request: ExportRequest) -> dict:
    checkpoint = selected_model_path()
    if not checkpoint.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found: {checkpoint}")
    opset = int(request.opset_version or _load_config().get("export", {}).get("opset_version", 18))
    try:
        output_path = export_onnx(checkpoint, onnx_path_for_checkpoint(checkpoint), opset)
        import onnx

        model = onnx.load(output_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "ok": True,
        "onnx_path": str(output_path),
        "input": model.graph.input[0].name,
        "output": model.graph.output[0].name,
        "nodes": len(model.graph.node),
    }


@app.post("/api/benchmark")
def benchmark(request: BenchmarkRequest) -> dict:
    checkpoint = selected_model_path()
    if not checkpoint.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found: {checkpoint}")
    try:
        return app.state.model_worker.benchmark(
            str(checkpoint),
            app.state.device_mode,
            request.iterations,
            request.warmup,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def main() -> None:
    import uvicorn

    uvicorn.run("vocarig.ui.app:app", host="127.0.0.1", port=8010, reload=False)


def selected_model_path() -> Path:
    if app.state.model_path:
        return _project_path(str(app.state.model_path))
    configured = _paths_from_config(_load_config())["checkpoint"]
    if configured.exists():
        return configured
    latest = _latest_trained_model()
    return latest or configured


def _latest_trained_model() -> Path | None:
    candidates = [path for path in (ROOT / "models" / "trained").glob("*.pt") if path.is_file()]
    if not candidates:
        candidates = [path for path in (ROOT / "models").glob("*.pt") if path.is_file()]
    return max(candidates, key=lambda item: item.stat().st_mtime) if candidates else None


def _default_metrics_path() -> Path:
    selected_metrics = metrics_path_for_checkpoint(selected_model_path())
    if selected_metrics.exists():
        return selected_metrics
    return _paths_from_config(_load_config())["metrics"]


def _start_training_job(config: TrainingRunConfig) -> None:
    stop_event = threading.Event()
    with training_job.lock:
        training_job.stop_event = stop_event
        training_job.events = []
        training_job.metrics = None
        training_job.error = None
        training_job.status = "running"
        training_job.config = config

    def on_progress(event: dict) -> None:
        with training_job.lock:
            event.setdefault("time", time.time())
            training_job.events.append(event)

    def worker() -> None:
        try:
            result = train_model(config, progress_callback=on_progress, stop_event=stop_event)
            with training_job.lock:
                training_job.metrics = result
                training_job.status = "stopped" if result.get("stopped") else "finished"
            checkpoint_path = result.get("checkpoint_path") or str(config.checkpoint_path)
            if checkpoint_path and Path(checkpoint_path).exists():
                app.state.model_path = checkpoint_path
        except Exception:
            with training_job.lock:
                training_job.error = traceback.format_exc()
                training_job.status = "failed"

    thread = threading.Thread(target=worker, daemon=True)
    with training_job.lock:
        training_job.thread = thread
    thread.start()


def _paths_from_config(config: dict) -> dict[str, Path]:
    synthetic_config = config.get("synthetic", {})
    training_config = config.get("training", {})
    export_config = config.get("export", {})
    artifact_dir = export_config.get("artifact_dir", DEFAULT_ARTIFACT_DIR)
    return {
        "data": _project_path(synthetic_config.get("output_path", "data/synthetic/synthetic_vocarig.npz")),
        "synthetic_metadata": _project_path(
            synthetic_config.get("metadata_path", "data/synthetic/synthetic_vocarig_metadata.json")
        ),
        "checkpoint": _project_path(export_config.get("checkpoint_path", "models/vocarig_lipsync_gru.pt")),
        "onnx": _project_path(export_config.get("onnx_path", "models/vocarig_lipsync_gru.onnx")),
        "metrics": _project_path(
            training_config.get("metrics_path", "models/vocarig_lipsync_gru_training_metrics.json")
        ),
        "checkpoint_dir": _project_path(training_config.get("checkpoint_dir", "models/training_checkpoints")),
        "artifact_dir": _project_path(artifact_dir),
    }


def _dataset_payload(selected_path: str | Path | None = None) -> dict:
    paths = _paths_from_config(_load_config())
    selected = _resolve_dataset_path(selected_path, paths["data"]) if selected_path else paths["data"]
    options = _dataset_options(selected)
    if not selected.exists() and options:
        selected = Path(options[0]["path"])
    return {"selected": str(selected), "options": options}


def _dataset_options(selected: Path) -> list[dict]:
    options: list[dict] = []
    for kind, directory in (("synthetic", ROOT / "data" / "synthetic"), ("processed", ROOT / "data" / "processed")):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.npz")):
            options.append(_dataset_option(path, kind, path.resolve() == selected.resolve()))
    return options


def _dataset_option(path: Path, kind: str, selected: bool) -> dict:
    payload = {
        "path": str(path),
        "name": path.name,
        "kind": kind,
        "selected": selected,
        "size_mb": round(path.stat().st_size / 1048576, 3),
        "frames": None,
        "duration_seconds": None,
    }
    try:
        with np.load(path, allow_pickle=False) as data:
            if "y" in data:
                frames = int(data["y"].shape[0])
                payload["frames"] = frames
                payload["duration_seconds"] = round(frames / 30.0, 3)
    except Exception:
        pass
    return payload


def _resolve_dataset_path(value: str | Path | None, default: Path) -> Path:
    path = _project_path(value) if value else default
    allowed_roots = [ROOT / "data" / "synthetic", ROOT / "data" / "processed"]
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {path}")
    if path.suffix.lower() != ".npz":
        raise HTTPException(status_code=400, detail="Dataset must be a .npz file")
    if path.resolve() == default.resolve():
        return path
    if not any(_is_relative_to(path, root) for root in allowed_roots):
        raise HTTPException(status_code=400, detail="Dataset must be under data/synthetic or data/processed")
    return path


def _metadata_path_for_dataset(data_path: Path, fallback: Path) -> Path:
    if data_path.resolve() == _paths_from_config(_load_config())["data"].resolve():
        return fallback
    return data_path.with_name(f"{data_path.stem}_metadata.json")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _audio_config(config: dict) -> AudioFeatureConfig:
    audio = config.get("audio", {})
    return AudioFeatureConfig(
        sample_rate=int(audio.get("sample_rate", 16000)),
        n_mels=int(audio.get("n_mels", 80)),
        window_ms=float(audio.get("window_ms", 25.0)),
        hop_ms=float(audio.get("hop_ms", 10.0)),
        context_frames=int(audio.get("context_frames", 11)),
        fps=int(audio.get("fps", 30)),
        f_min=float(audio.get("f_min", 50.0)),
        f_max=float(audio.get("f_max", 7600.0)),
    )


def _audio_payload(config: dict) -> dict:
    audio = _audio_config(config)
    return asdict(audio)


def _model_payload() -> dict:
    selected = selected_model_path()
    options = _model_options()
    if selected.exists() and not any(option["path"] == str(selected) for option in options):
        options.insert(0, _model_option(selected, "selected"))
    return {"selected": str(selected), "selected_name": selected.name, "options": options}


def _model_options() -> list[dict]:
    root = ROOT / "models"
    options = []
    if not root.exists():
        return options
    seen: set[Path] = set()
    for path in sorted((root / "trained").glob("*.pt")):
        if path.is_file():
            options.append(_model_option(path, "trained"))
            seen.add(path.resolve())
    for path in sorted(root.glob("*.pt")):
        if path.resolve() not in seen:
            options.append(_model_option(path, "legacy"))
            seen.add(path.resolve())
    for path in sorted((root / "training_checkpoints").glob("**/*.pt")):
        if path.resolve() not in seen:
            options.append(_model_option(path, "checkpoint"))
            seen.add(path.resolve())
    return options


def _model_option(path: Path, group: str) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "label": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "name": path.name,
        "group": group,
        "size_mb": round(stat.st_size / (1024 * 1024), 3),
        "mtime": stat.st_mtime,
    }


def _metrics_option(path: Path, selected: bool = False) -> dict:
    try:
        data = _read_json(path)
    except HTTPException:
        data = {}
    return {
        "path": str(path),
        "label": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "name": path.name,
        "selected": selected,
        "completed_epochs": data.get("completed_epochs"),
        "best_val_loss": data.get("best_val_loss"),
    }


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_config(path: Path | None = None) -> dict:
    config_path = CONFIG_PATH if path is None else path
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read {path}: {exc}") from exc


def _device_payload() -> dict:
    payload = app.state.model_worker.device_status(app.state.device_mode)
    return {
        "selected": payload["selected"],
        "effective": payload["effective"],
        "cuda_available": payload["cuda_available"],
        "modes": payload["modes"],
    }


def _normalize_device(device: str) -> str:
    normalized = str(device or "auto").lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise HTTPException(status_code=400, detail="device must be auto, cpu, or cuda")
    return normalized


def _request_config_value(
    request: BaseModel,
    field_name: str,
    config: dict,
    default: Any,
    cast: type,
) -> Any:
    value = getattr(request, field_name)
    if value is None:
        value = config.get(field_name, default)
    return cast(value)


def _training_config_payload(config: Any | None) -> dict | None:
    if config is None:
        return None
    payload = asdict(config)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


if __name__ == "__main__":
    main()
