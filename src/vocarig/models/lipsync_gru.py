"""Streaming GRU model for audio-to-lip ARKit controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class LipSyncGRUOutput:
    lip: torch.Tensor
    hidden: torch.Tensor
    raw_lip: torch.Tensor
    delta_effect: torch.Tensor
    gate: torch.Tensor
    step_scale: torch.Tensor
    warmup_scale: torch.Tensor


class LipSyncGRU(nn.Module):
    """Small causal model that predicts bounded lip-state updates."""

    model_type = "lipsync_gru"

    def __init__(
        self,
        audio_context_frames: int = 11,
        n_mels: int = 80,
        lip_size: int = 21,
        time_size: int = 3,
        style_size: int = 2,
        hidden_size: int = 128,
        audio_channels: int = 64,
        max_step: float = 0.12,
        reference_dt: float = 1.0 / 30.0,
        warmup_steps: int = 3,
    ) -> None:
        super().__init__()
        self.audio_context_frames = int(audio_context_frames)
        self.n_mels = int(n_mels)
        self.lip_size = int(lip_size)
        self.time_size = int(time_size)
        self.style_size = int(style_size)
        self.hidden_size = int(hidden_size)
        self.audio_channels = int(audio_channels)
        self.max_step = float(max_step)
        self.reference_dt = float(reference_dt)
        self.warmup_steps = int(warmup_steps)

        self.audio_encoder = nn.Sequential(
            nn.Conv1d(self.n_mels, self.audio_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(self.audio_channels, self.audio_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        input_size = self.audio_channels + self.lip_size + self.time_size + self.style_size
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.GELU(),
        )
        self.gru_cell = nn.GRUCell(self.hidden_size, self.hidden_size)
        self.output_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.lip_size * 2),
        )

    def forward(
        self,
        audio_window: torch.Tensor,
        previous_lip: torch.Tensor,
        time_style: torch.Tensor,
        hidden: torch.Tensor | None = None,
        steps_since_reset: torch.Tensor | int | float | None = None,
    ) -> LipSyncGRUOutput:
        if audio_window.ndim != 3:
            raise ValueError("audio_window must have shape (batch, context, n_mels)")
        if audio_window.shape[1:] != (self.audio_context_frames, self.n_mels):
            raise ValueError(
                "audio_window must have shape "
                f"(batch, {self.audio_context_frames}, {self.n_mels})"
            )
        if previous_lip.ndim != 2 or previous_lip.shape[-1] != self.lip_size:
            raise ValueError(f"previous_lip must have shape (batch, {self.lip_size})")
        expected_time_style = self.time_size + self.style_size
        if time_style.ndim != 2 or time_style.shape[-1] != expected_time_style:
            raise ValueError(f"time_style must have shape (batch, {expected_time_style})")

        batch_size = audio_window.shape[0]
        if hidden is None:
            hidden = self.initial_hidden(batch_size, audio_window.device, audio_window.dtype)

        audio_features = self.audio_encoder(audio_window.transpose(1, 2))
        projected = self.input_projection(
            torch.cat([audio_features, previous_lip, time_style], dim=-1)
        )
        next_hidden = self.gru_cell(projected, hidden)
        delta_raw, gate_raw = self.output_head(next_hidden).chunk(2, dim=-1)

        delta_time = time_style[:, 0]
        step_scale = self._step_scale(delta_time).unsqueeze(-1)
        warmup_scale = self._warmup_scale(steps_since_reset, audio_window).unsqueeze(-1)
        gate = torch.sigmoid(gate_raw)
        delta = self.max_step * step_scale * torch.tanh(delta_raw)
        raw_lip = previous_lip + warmup_scale * gate * delta
        lip = torch.clamp(raw_lip, 0.0, 1.0)
        delta_effect = lip - previous_lip

        return LipSyncGRUOutput(
            lip=lip,
            hidden=next_hidden,
            raw_lip=raw_lip,
            delta_effect=delta_effect,
            gate=gate,
            step_scale=step_scale,
            warmup_scale=warmup_scale,
        )

    def initial_hidden(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def config_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "audio_context_frames": self.audio_context_frames,
            "n_mels": self.n_mels,
            "lip_size": self.lip_size,
            "time_size": self.time_size,
            "style_size": self.style_size,
            "hidden_size": self.hidden_size,
            "audio_channels": self.audio_channels,
            "max_step": self.max_step,
            "reference_dt": self.reference_dt,
            "warmup_steps": self.warmup_steps,
        }

    def _step_scale(self, delta_time: torch.Tensor) -> torch.Tensor:
        if self.reference_dt <= 0:
            raise ValueError("reference_dt must be greater than 0")
        return torch.clamp(delta_time / self.reference_dt, 0.0, 2.0)

    def _warmup_scale(
        self,
        steps_since_reset: torch.Tensor | int | float | None,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if steps_since_reset is None or self.warmup_steps <= 0:
            return torch.ones(reference.shape[0], device=reference.device, dtype=reference.dtype)
        if isinstance(steps_since_reset, torch.Tensor):
            steps = steps_since_reset.to(device=reference.device, dtype=reference.dtype)
            if steps.ndim == 0:
                steps = steps.expand(reference.shape[0])
        else:
            steps = torch.full(
                (reference.shape[0],),
                float(steps_since_reset),
                device=reference.device,
                dtype=reference.dtype,
            )
        return torch.clamp(steps / float(self.warmup_steps), 0.0, 1.0)
