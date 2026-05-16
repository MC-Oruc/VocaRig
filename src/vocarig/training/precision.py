"""Device and precision helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


VALID_PRECISIONS = {"fp32", "fp16_amp", "bf16_amp"}


@dataclass(frozen=True)
class PrecisionConfig:
    precision: str = "fp32"
    allow_tf32: bool = True
    amp_dtype: str = "fp16"


@dataclass(frozen=True)
class PrecisionRuntime:
    precision: str
    amp_enabled: bool
    amp_dtype: torch.dtype | None
    scaler: torch.amp.GradScaler | None


def resolve_device(requested: str) -> torch.device:
    """Resolve `auto`, `cpu`, or `cuda`."""

    normalized = str(requested or "auto").lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    return torch.device(normalized)


def configure_device(device: torch.device, allow_tf32: bool) -> None:
    """Apply CUDA matmul settings."""

    if device.type != "cuda":
        return
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)


def build_precision_runtime(
    config: PrecisionConfig,
    device: torch.device,
) -> PrecisionRuntime:
    """Create autocast/scaler settings for a training run."""

    precision = str(config.precision or "fp32").lower()
    if precision not in VALID_PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(VALID_PRECISIONS)}")
    if precision == "fp32":
        return PrecisionRuntime(precision, False, None, None)
    if device.type != "cuda":
        raise RuntimeError(f"{precision} requires CUDA")
    dtype = torch.float16 if precision == "fp16_amp" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16_amp")
    return PrecisionRuntime(precision, True, dtype, scaler)


def autocast_context(runtime: PrecisionRuntime, device: torch.device):
    """Return a torch autocast context for the current precision."""

    return torch.amp.autocast(
        device_type=device.type,
        dtype=runtime.amp_dtype,
        enabled=runtime.amp_enabled,
    )
