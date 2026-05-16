"""Model training entry point for VocaRig."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
import yaml

from vocarig.models.blendshapes import ARKIT_BLENDSHAPE_NAMES, LIP_BLENDSHAPE_NAMES, LIP_INDEX
from vocarig.models.lipsync_gru import LipSyncGRU
from vocarig.training.data import load_sequence_npz_dataset, split_indices
from vocarig.training.precision import (
    PrecisionConfig,
    autocast_context,
    build_precision_runtime,
    configure_device,
    resolve_device,
)


LOSS_METRIC_NAMES = (
    "loss",
    "pose_loss",
    "delta_loss",
    "velocity_loss",
    "jerk_loss",
    "silence_loss",
    "range_loss",
)


@dataclass(frozen=True)
class TrainingRunConfig:
    data_path: Path = Path("data/synthetic/synthetic_vocarig.npz")
    checkpoint_path: Path = Path("models/vocarig_lipsync_gru.pt")
    metrics_path: Path = Path("models/vocarig_lipsync_gru_training_metrics.json")
    checkpoint_dir: Path = Path("models/training_checkpoints")
    resume_checkpoint: Path | None = None
    checkpoint_interval: int = 25
    audio_context_frames: int = 11
    n_mels: int = 80
    lip_size: int = 21
    time_size: int = 3
    style_size: int = 2
    hidden_size: int = 128
    audio_channels: int = 64
    max_step: float = 0.12
    reference_dt: float = 1.0 / 30.0
    warmup_steps: int = 3
    sequence_window: int = 96
    sequence_stride: int = 24
    epochs: int = 1800
    batch_size: int = 64
    learning_rate: float = 0.00035
    weight_decay: float = 0.0001
    validation_split: float = 0.1
    device: str = "auto"
    precision: str = "fp32"
    allow_tf32: bool = True
    amp_dtype: str = "fp16"
    num_workers: int = 0
    log_interval: int = 20
    metric_eval_interval: int = 1
    seed: int = 42
    final_teacher_forcing_ratio: float = 0.0
    teacher_decay_start_epoch: int = 250
    teacher_decay_epochs: int = 700
    warmup_loss_steps: int = 6
    early_stopping_patience: int = 80
    early_stopping_min_delta: float = 0.00001
    early_stopping_min_epochs: int = 120
    target_val_loss: float = 0.0025
    target_train_loss: float = 0.0012
    divergence_loss: float = 0.25
    overfit_gap_ratio: float = 4.0
    stop_on_target_val_loss: bool = False
    stop_on_target_train_loss: bool = False
    stop_on_divergence_loss: bool = True
    stop_on_plateau: bool = True
    stop_on_overfit_gap: bool = False
    pose_loss_weight: float = 1.0
    delta_loss_weight: float = 0.3
    velocity_loss_weight: float = 0.12
    jerk_loss_weight: float = 0.04
    silence_loss_weight: float = 0.45
    range_loss_weight: float = 0.02


ProgressCallback = Callable[[dict], None]


def main() -> None:
    args = _parse_args()
    train_model(build_training_config(args))


def build_training_config(args: argparse.Namespace) -> TrainingRunConfig:
    """Merge YAML config and CLI overrides."""

    data = _load_yaml(args.config)
    model_config = data.get("model", {})
    training_config = data.get("training", {})
    synthetic_config = data.get("synthetic", {})
    export_config = data.get("export", {})

    return TrainingRunConfig(
        data_path=Path(args.data or synthetic_config.get("output_path", "data/synthetic/synthetic_vocarig.npz")),
        checkpoint_path=Path(args.checkpoint or export_config.get("checkpoint_path", "models/vocarig_lipsync_gru.pt")),
        metrics_path=Path(training_config.get("metrics_path", "models/vocarig_lipsync_gru_training_metrics.json")),
        checkpoint_dir=Path(args.checkpoint_dir or training_config.get("checkpoint_dir", "models/training_checkpoints")),
        resume_checkpoint=Path(args.resume) if args.resume else None,
        checkpoint_interval=int(training_config.get("checkpoint_interval", 25)),
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
        sequence_window=int(training_config.get("sequence_window", 96)),
        sequence_stride=int(training_config.get("sequence_stride", 24)),
        epochs=int(args.epochs or training_config.get("epochs", 1800)),
        batch_size=int(args.batch_size or training_config.get("batch_size", 64)),
        learning_rate=float(args.lr or training_config.get("learning_rate", 0.00035)),
        weight_decay=float(training_config.get("weight_decay", 0.0001)),
        validation_split=float(training_config.get("validation_split", 0.1)),
        device=str(args.device or training_config.get("device", "auto")),
        precision=str(args.precision or training_config.get("precision", "fp32")),
        allow_tf32=bool(training_config.get("allow_tf32", True)),
        amp_dtype=str(training_config.get("amp_dtype", "fp16")),
        num_workers=int(training_config.get("num_workers", 0)),
        log_interval=int(training_config.get("log_interval", 20)),
        metric_eval_interval=int(training_config.get("metric_eval_interval", 1)),
        seed=int(training_config.get("seed", 42)),
        final_teacher_forcing_ratio=float(training_config.get("final_teacher_forcing_ratio", 0.0)),
        teacher_decay_start_epoch=int(training_config.get("teacher_decay_start_epoch", 250)),
        teacher_decay_epochs=int(training_config.get("teacher_decay_epochs", 700)),
        warmup_loss_steps=int(training_config.get("warmup_loss_steps", 6)),
        early_stopping_patience=int(training_config.get("early_stopping_patience", 80)),
        early_stopping_min_delta=float(training_config.get("early_stopping_min_delta", 0.00001)),
        early_stopping_min_epochs=int(training_config.get("early_stopping_min_epochs", 120)),
        target_val_loss=float(training_config.get("target_val_loss", 0.0025)),
        target_train_loss=float(training_config.get("target_train_loss", 0.0012)),
        divergence_loss=float(training_config.get("divergence_loss", 0.25)),
        overfit_gap_ratio=float(training_config.get("overfit_gap_ratio", 4.0)),
        stop_on_target_val_loss=bool(training_config.get("stop_on_target_val_loss", False)),
        stop_on_target_train_loss=bool(training_config.get("stop_on_target_train_loss", False)),
        stop_on_divergence_loss=bool(training_config.get("stop_on_divergence_loss", True)),
        stop_on_plateau=bool(training_config.get("stop_on_plateau", True)),
        stop_on_overfit_gap=bool(training_config.get("stop_on_overfit_gap", False)),
        pose_loss_weight=float(training_config.get("pose_loss_weight", 1.0)),
        delta_loss_weight=float(training_config.get("delta_loss_weight", 0.3)),
        velocity_loss_weight=float(training_config.get("velocity_loss_weight", 0.12)),
        jerk_loss_weight=float(training_config.get("jerk_loss_weight", 0.04)),
        silence_loss_weight=float(training_config.get("silence_loss_weight", 0.45)),
        range_loss_weight=float(training_config.get("range_loss_weight", 0.02)),
    )


def train_model(
    config: TrainingRunConfig,
    progress_callback: ProgressCallback | None = None,
    stop_event: threading.Event | None = None,
    pause_event: threading.Event | None = None,
) -> dict:
    """Train a VocaRig GRU and save checkpoint + metrics."""

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = resolve_device(config.device)
    configure_device(device, config.allow_tf32)
    precision_runtime = build_precision_runtime(
        PrecisionConfig(config.precision, config.allow_tf32, config.amp_dtype),
        device,
    )
    bundle = load_sequence_npz_dataset(
        config.data_path,
        context_frames=config.audio_context_frames,
        n_mels=config.n_mels,
        lip_size=config.lip_size,
        time_size=config.time_size,
        style_size=config.style_size,
        sequence_window=config.sequence_window,
        sequence_stride=config.sequence_stride,
    )
    train_indices, val_indices = split_indices(len(bundle.dataset), config.validation_split, config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        Subset(bundle.dataset, train_indices.tolist()),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=generator,
    )
    train_eval_loader = DataLoader(
        Subset(bundle.dataset, train_indices.tolist()),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        Subset(bundle.dataset, val_indices.tolist()),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = _model_from_config(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    run_metadata = _run_metadata(config, bundle.metadata)
    run_signature = _run_signature(run_metadata)
    run_id = _new_run_id(run_signature)
    history: list[dict] = []
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    start_epoch = 1
    stopped = False
    stop_reason: str | None = None
    last_training_checkpoint: str | None = None
    early_stop_best_val_loss = float("inf")
    stale_epochs = 0

    if config.resume_checkpoint is not None:
        resume = _load_resume_checkpoint(config.resume_checkpoint, run_signature, device)
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        history = list(resume["training_state"].get("history", []))
        best_val_loss = float(resume["training_state"].get("best_val_loss", float("inf")))
        best_state = resume["training_state"].get("best_state_dict")
        start_epoch = int(resume["epoch"]) + 1
        run_id = str(resume["run_id"])
        early_stop_best_val_loss = float(resume["training_state"].get("early_stop_best_val_loss", best_val_loss))
        stale_epochs = int(resume["training_state"].get("stale_epochs", 0))

    if progress_callback:
        progress_callback(
            {
                "event": "started",
                "device": str(device),
                "precision": precision_runtime.precision,
                "amp_enabled": precision_runtime.amp_enabled,
                "allow_tf32": config.allow_tf32,
                "epochs": config.epochs,
                "start_epoch": start_epoch,
                "run_id": run_id,
                "train_size": len(train_indices),
                "validation_size": len(val_indices),
            }
        )

    started_at = time.perf_counter()
    for epoch in range(start_epoch, config.epochs + 1):
        if _wait_if_paused(pause_event, stop_event):
            stopped = True
            stop_reason = "aborted"
            break

        teacher_forcing_ratio = _teacher_forcing_ratio(epoch, config)
        train_metrics = _train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            config,
            precision_runtime,
            teacher_forcing_ratio,
            stop_event,
            pause_event,
        )
        train_rollout_metrics = (
            _evaluate(model, train_eval_loader, device, config)
            if _should_evaluate_metrics(epoch, config)
            else train_metrics
        )
        val_metrics = (
            _evaluate(model, val_loader, device, config)
            if _should_evaluate_metrics(epoch, config)
            else train_metrics
        )
        val_loss = val_metrics["loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = _clone_state_dict(model.state_dict())
        if val_loss < early_stop_best_val_loss - config.early_stopping_min_delta:
            early_stop_best_val_loss = val_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
        auto_stop_reason = _automatic_stop_reason(
            epoch,
            train_metrics["loss"],
            train_rollout_metrics["loss"],
            val_loss,
            stale_epochs,
            config,
        )
        if auto_stop_reason:
            stopped = True
            stop_reason = auto_stop_reason

        elapsed = time.perf_counter() - started_at
        row = {
            "epoch": epoch,
            "teacher_forcing_ratio": teacher_forcing_ratio,
            "train_loss": train_metrics["loss"],
            "train_rollout_loss": train_rollout_metrics["loss"],
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "early_stop_best_val_loss": early_stop_best_val_loss,
            "stale_epochs": stale_epochs,
            "early_stopped": bool(auto_stop_reason),
            "stop_reason": auto_stop_reason,
            "elapsed_seconds": elapsed,
            **_prefixed_metrics("train", train_metrics),
            **_prefixed_metrics("train_rollout", train_rollout_metrics),
            **_prefixed_metrics("val", val_metrics),
        }
        history.append(row)
        if progress_callback:
            progress_callback({"event": "epoch", **row})
        if _should_log(epoch, config):
            print(
                f"epoch={epoch} train={row['train_loss']:.6f} "
                f"rollout={row['train_rollout_loss']:.6f} val={row['val_loss']:.6f}"
            )
        if _should_save_training_checkpoint(epoch, config):
            last_training_checkpoint = _save_training_checkpoint(
                config,
                model,
                optimizer,
                epoch,
                run_id,
                run_signature,
                run_metadata,
                history,
                best_val_loss,
                best_state,
                row,
                early_stop_best_val_loss,
                stale_epochs,
            )
        if auto_stop_reason:
            break
        if stop_event is not None and stop_event.is_set():
            stopped = True
            stop_reason = "aborted"
            break

    if history:
        last_row = history[-1]
        last_training_checkpoint = _save_training_checkpoint(
            config,
            model,
            optimizer,
            int(last_row["epoch"]),
            run_id,
            run_signature,
            run_metadata,
            history,
            best_val_loss,
            best_state,
            last_row,
            early_stop_best_val_loss,
            stale_epochs,
        )
    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {
        "data_path": str(config.data_path),
        "device": str(device),
        "precision": precision_runtime.precision,
        "amp_enabled": precision_runtime.amp_enabled,
        "allow_tf32": config.allow_tf32,
        "training_config": _serializable_config(config),
        "run_id": run_id,
        "run_signature": run_signature,
        "checkpoint_dir": str(config.checkpoint_dir),
        "resume_checkpoint": str(config.resume_checkpoint) if config.resume_checkpoint else None,
        "last_training_checkpoint": last_training_checkpoint,
        "epochs": config.epochs,
        "completed_epochs": len(history),
        "train_size": len(train_indices),
        "validation_size": len(val_indices),
        "best_val_loss": None if not history else best_val_loss,
        "final_train_loss": None if not history else history[-1]["train_loss"],
        "final_train_rollout_loss": None if not history else history[-1]["train_rollout_loss"],
        "final_val_loss": None if not history else history[-1]["val_loss"],
        "final_train_components": None if not history else _components_from_row("train", history[-1]),
        "final_val_components": None if not history else _components_from_row("val", history[-1]),
        "stopped": stopped,
        "stop_reason": stop_reason,
        "early_stopped": bool(stop_reason and stop_reason != "aborted"),
        "history": history,
    }
    checkpoint = {
        "run_id": run_id,
        "run_signature": run_signature,
        "model_state_dict": _state_dict_cpu(model.state_dict()),
        "model_config": model.config_dict(),
        "training_config": _serializable_config(config),
        "metrics": metrics,
        "lip_names": LIP_BLENDSHAPE_NAMES,
        "arkit_names": ARKIT_BLENDSHAPE_NAMES,
        "dataset_metadata": bundle.metadata,
    }
    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, config.checkpoint_path)
    config.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if progress_callback:
        progress_callback({"event": "finished", **metrics})
    print(f"Checkpoint written: {config.checkpoint_path}")
    print(f"Metrics written: {config.metrics_path}")
    return metrics


def _model_from_config(config: TrainingRunConfig) -> LipSyncGRU:
    return LipSyncGRU(
        audio_context_frames=config.audio_context_frames,
        n_mels=config.n_mels,
        lip_size=config.lip_size,
        time_size=config.time_size,
        style_size=config.style_size,
        hidden_size=config.hidden_size,
        audio_channels=config.audio_channels,
        max_step=config.max_step,
        reference_dt=config.reference_dt,
        warmup_steps=config.warmup_steps,
    )


def _train_epoch(
    model: LipSyncGRU,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: TrainingRunConfig,
    precision_runtime,
    teacher_forcing_ratio: float,
    stop_event: threading.Event | None = None,
    pause_event: threading.Event | None = None,
) -> dict[str, float]:
    model.train()
    totals = _new_loss_totals(device)
    total_windows = 0
    for audio, previous, time_style, y in loader:
        if _wait_if_paused(pause_event, stop_event):
            break
        audio = audio.to(device, non_blocking=True)
        previous = previous.to(device, non_blocking=True)
        time_style = time_style.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(precision_runtime, device):
            losses = _sequence_loss_components(
                model,
                audio,
                previous,
                time_style,
                y,
                config,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )
            loss = losses["loss"]
        if precision_runtime.scaler is not None and precision_runtime.scaler.is_enabled():
            precision_runtime.scaler.scale(loss).backward()
            precision_runtime.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            precision_runtime.scaler.step(optimizer)
            precision_runtime.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        batch_size = audio.shape[0]
        _accumulate_loss_totals(totals, losses, batch_size)
        total_windows += batch_size
    return _finalize_loss_totals(totals, total_windows)


@torch.inference_mode()
def _evaluate(
    model: LipSyncGRU,
    loader: DataLoader,
    device: torch.device,
    config: TrainingRunConfig,
) -> dict[str, float]:
    model.eval()
    totals = _new_loss_totals(device)
    total_windows = 0
    for audio, previous, time_style, y in loader:
        audio = audio.to(device, non_blocking=True)
        previous = previous.to(device, non_blocking=True)
        time_style = time_style.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        losses = _sequence_loss_components(model, audio, previous, time_style, y, config)
        batch_size = audio.shape[0]
        _accumulate_loss_totals(totals, losses, batch_size)
        total_windows += batch_size
    return _finalize_loss_totals(totals, total_windows)


def _sequence_loss_components(
    model: LipSyncGRU,
    audio: torch.Tensor,
    previous: torch.Tensor,
    time_style: torch.Tensor,
    y: torch.Tensor,
    config: TrainingRunConfig,
    teacher_forcing_ratio: float = 0.0,
) -> dict[str, torch.Tensor]:
    batch, sequence_length = y.shape[:2]
    hidden = model.initial_hidden(batch, audio.device, audio.dtype)
    predictions: list[torch.Tensor] = []
    previous_inputs: list[torch.Tensor] = []
    raw_outputs: list[torch.Tensor] = []
    previous_prediction: torch.Tensor | None = None
    step_values = torch.arange(sequence_length, device=audio.device, dtype=audio.dtype)

    for timestep in range(sequence_length):
        previous_step = previous[:, timestep, :]
        if previous_prediction is not None:
            previous_step = previous_prediction.detach()
            if teacher_forcing_ratio > 0.0:
                teacher_step = previous[:, timestep, :]
                if teacher_forcing_ratio >= 1.0:
                    previous_step = teacher_step
                else:
                    mask = torch.rand(batch, 1, device=audio.device, dtype=audio.dtype) < teacher_forcing_ratio
                    previous_step = torch.where(mask, teacher_step, previous_step)
        output = model(
            audio[:, timestep, :, :],
            previous_step,
            time_style[:, timestep, :],
            hidden,
            steps_since_reset=step_values[timestep],
        )
        hidden = output.hidden
        previous_prediction = output.lip
        predictions.append(output.lip)
        previous_inputs.append(previous_step)
        raw_outputs.append(output.raw_lip)

    pred = torch.stack(predictions, dim=1)
    prev = torch.stack(previous_inputs, dim=1)
    raw = torch.stack(raw_outputs, dim=1)
    weights = _timestep_weights(sequence_length, config.warmup_loss_steps, audio.device, audio.dtype)
    pose_loss = _weighted_square_mean(pred - y, weights)
    delta_loss = _weighted_square_mean((pred - prev) - (y - prev), weights)
    velocity_loss = torch.zeros((), device=audio.device, dtype=audio.dtype)
    if sequence_length >= 2:
        velocity_loss = _weighted_square_mean(
            (pred[:, 1:] - pred[:, :-1]) - (y[:, 1:] - y[:, :-1]),
            weights[:, 1:, :],
        )
    jerk_loss = torch.zeros((), device=audio.device, dtype=audio.dtype)
    if sequence_length >= 3:
        jerk_loss = _weighted_square_mean(
            (pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2])
            - (y[:, 2:] - 2.0 * y[:, 1:-1] + y[:, :-2]),
            weights[:, 2:, :],
        )
    energy = time_style[:, :, 2:3]
    jaw = pred[:, :, LIP_INDEX["jawOpen"] : LIP_INDEX["jawOpen"] + 1]
    silence_loss = _weighted_mean(torch.relu(0.04 - energy) * jaw.square(), weights)
    range_loss = _weighted_mean(torch.relu(raw - 1.0).square() + torch.relu(-raw).square(), weights)
    loss = (
        config.pose_loss_weight * pose_loss
        + config.delta_loss_weight * delta_loss
        + config.velocity_loss_weight * velocity_loss
        + config.jerk_loss_weight * jerk_loss
        + config.silence_loss_weight * silence_loss
        + config.range_loss_weight * range_loss
    )
    return {
        "loss": loss,
        "pose_loss": pose_loss,
        "delta_loss": delta_loss,
        "velocity_loss": velocity_loss,
        "jerk_loss": jerk_loss,
        "silence_loss": silence_loss,
        "range_loss": range_loss,
    }


def _teacher_forcing_ratio(epoch: int, config: TrainingRunConfig) -> float:
    final_ratio = max(0.0, min(1.0, float(config.final_teacher_forcing_ratio)))
    start = max(1, int(config.teacher_decay_start_epoch))
    decay_epochs = max(1, int(config.teacher_decay_epochs))
    if epoch < start:
        return 1.0
    progress = min(1.0, max(0.0, (epoch - start + 1) / decay_epochs))
    return float(1.0 + (final_ratio - 1.0) * progress)


def _timestep_weights(
    sequence_length: int,
    warmup_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    weights = torch.ones(1, sequence_length, 1, device=device, dtype=dtype)
    warmup = max(0, min(sequence_length, int(warmup_steps)))
    if warmup > 0:
        ramp = torch.linspace(0.5, 1.0, warmup, device=device, dtype=dtype)
        weights[:, :warmup, :] = ramp.view(1, warmup, 1)
    return weights


def _weighted_square_mean(value: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return _weighted_mean(value.square(), weights)


def _weighted_mean(value: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    while weights.ndim < value.ndim:
        weights = weights.unsqueeze(-1)
    denominator = weights.expand_as(value).sum().clamp_min(1.0)
    return (value * weights).sum() / denominator


def _new_loss_totals(device: torch.device) -> dict[str, torch.Tensor]:
    return {name: torch.zeros((), device=device) for name in LOSS_METRIC_NAMES}


def _accumulate_loss_totals(
    totals: dict[str, torch.Tensor],
    losses: dict[str, torch.Tensor],
    batch_size: int,
) -> None:
    for name in LOSS_METRIC_NAMES:
        totals[name].add_(losses[name].detach().float() * batch_size)


def _finalize_loss_totals(totals: dict[str, torch.Tensor], total_windows: int) -> dict[str, float]:
    return {
        name: float((value / max(1, total_windows)).detach().cpu())
        for name, value in totals.items()
    }


def _wait_if_paused(
    pause_event: threading.Event | None,
    stop_event: threading.Event | None,
) -> bool:
    while pause_event is not None and pause_event.is_set():
        if stop_event is not None and stop_event.is_set():
            return True
        time.sleep(0.1)
    return bool(stop_event is not None and stop_event.is_set())


def _should_log(epoch: int, config: TrainingRunConfig) -> bool:
    return epoch == 1 or epoch == config.epochs or epoch % max(1, config.log_interval) == 0


def _should_evaluate_metrics(epoch: int, config: TrainingRunConfig) -> bool:
    return epoch == 1 or epoch == config.epochs or epoch % max(1, config.metric_eval_interval) == 0


def _should_save_training_checkpoint(epoch: int, config: TrainingRunConfig) -> bool:
    return config.checkpoint_interval > 0 and epoch % config.checkpoint_interval == 0


def _automatic_stop_reason(
    epoch: int,
    train_loss: float,
    train_rollout_loss: float,
    val_loss: float,
    stale_epochs: int,
    config: TrainingRunConfig,
) -> str | None:
    if config.stop_on_divergence_loss and (
        not math.isfinite(val_loss)
        or not math.isfinite(train_loss)
        or val_loss >= config.divergence_loss
        or train_loss >= config.divergence_loss
    ):
        return "divergence_loss"
    if epoch < max(1, config.early_stopping_min_epochs):
        return None
    if config.stop_on_target_val_loss and val_loss <= config.target_val_loss:
        return "target_val_loss"
    if config.stop_on_target_train_loss and train_loss <= config.target_train_loss:
        return "target_train_loss"
    if config.stop_on_plateau and stale_epochs >= max(1, config.early_stopping_patience):
        return "plateau"
    if (
        config.stop_on_overfit_gap
        and train_rollout_loss > 0.0
        and val_loss / train_rollout_loss >= config.overfit_gap_ratio
    ):
        return "overfit_gap"
    return None


def _prefixed_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{name}": value for name, value in metrics.items() if name != "loss"}


def _components_from_row(prefix: str, row: dict) -> dict[str, float]:
    return {
        name: float(row[f"{prefix}_{name}"])
        for name in LOSS_METRIC_NAMES
        if name != "loss" and f"{prefix}_{name}" in row
    }


def _clone_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _state_dict_cpu(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().float() for key, value in state.items()}


def _run_metadata(config: TrainingRunConfig, dataset_metadata: dict) -> dict:
    path = Path(config.data_path).resolve()
    stat = path.stat()
    return {
        "dataset": {
            "path": str(path),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "window_count": dataset_metadata.get("window_count"),
            "sequence_window": config.sequence_window,
            "sequence_stride": config.sequence_stride,
        },
        "model": _model_from_config(config).config_dict(),
        "training": {
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "validation_split": config.validation_split,
            "seed": config.seed,
            "precision": config.precision,
            "allow_tf32": config.allow_tf32,
            "final_teacher_forcing_ratio": config.final_teacher_forcing_ratio,
            "teacher_decay_start_epoch": config.teacher_decay_start_epoch,
            "teacher_decay_epochs": config.teacher_decay_epochs,
            "warmup_loss_steps": config.warmup_loss_steps,
            "loss_weights": {
                "pose": config.pose_loss_weight,
                "delta": config.delta_loss_weight,
                "velocity": config.velocity_loss_weight,
                "jerk": config.jerk_loss_weight,
                "silence": config.silence_loss_weight,
                "range": config.range_loss_weight,
            },
        },
    }


def _run_signature(metadata: dict) -> str:
    data = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _new_run_id(signature: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"run_{stamp}_{signature[:10]}"


def _save_training_checkpoint(
    config: TrainingRunConfig,
    model: LipSyncGRU,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    run_id: str,
    run_signature: str,
    run_metadata: dict,
    history: list[dict],
    best_val_loss: float,
    best_state: dict[str, torch.Tensor] | None,
    row: dict,
    early_stop_best_val_loss: float,
    stale_epochs: int,
) -> str:
    checkpoint_path = config.checkpoint_dir / run_id / f"epoch_{epoch:04d}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_kind": "training_resume",
        "run_id": run_id,
        "run_signature": run_signature,
        "run_metadata": run_metadata,
        "epoch": epoch,
        "model_state_dict": _state_dict_cpu(model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model.config_dict(),
        "training_config": _serializable_config(config),
        "training_state": {
            "history": history,
            "best_val_loss": best_val_loss,
            "best_state_dict": best_state,
            "early_stop_best_val_loss": early_stop_best_val_loss,
            "stale_epochs": stale_epochs,
            "last_row": row,
        },
        "lip_names": LIP_BLENDSHAPE_NAMES,
        "arkit_names": ARKIT_BLENDSHAPE_NAMES,
    }
    torch.save(payload, checkpoint_path)
    return str(checkpoint_path)


def _load_resume_checkpoint(
    checkpoint_path: Path,
    expected_signature: str,
    device: torch.device,
) -> dict:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_kind") != "training_resume":
        raise ValueError("Selected checkpoint is not a resumable training checkpoint")
    if checkpoint.get("run_signature") != expected_signature:
        raise ValueError("Resume checkpoint does not match current data/model/training signature")
    return checkpoint


def list_training_checkpoints(checkpoint_dir: str | Path) -> dict:
    directory = Path(checkpoint_dir)
    entries = []
    if directory.exists():
        for path in sorted(directory.glob("**/*.pt")):
            entries.append({"path": str(path), "name": path.name, "size_mb": round(path.stat().st_size / 1048576, 3)})
    return {"checkpoint_dir": str(directory), "entries": entries}


def _serializable_config(config: TrainingRunConfig) -> dict:
    data = asdict(config)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VocaRig lip-sync GRU.")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", default=None)
    return parser.parse_args()


def _load_yaml(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


if __name__ == "__main__":
    main()
