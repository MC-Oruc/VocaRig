"""Blendshape mixing between FaceRig expression output and VocaRig lip output."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from vocarig.models.blendshapes import (
    ARKIT_BLENDSHAPE_NAMES,
    FACE_RIG_BLENDSHAPE_NAMES,
    LIP_BLENDSHAPE_NAMES,
    LIP_INDEX,
    LIP_OWNED_BLENDSHAPES,
    SHARED_MOUTH_BLENDSHAPES,
    zero_arkit_dict,
)


def lip_values_to_dict(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    """Convert a 21-wide lip vector to a named dictionary."""

    array = _validate_vector("lip_values", values, len(LIP_BLENDSHAPE_NAMES))
    return {
        name: float(array[index])
        for index, name in enumerate(LIP_BLENDSHAPE_NAMES)
    }


def lip_to_arkit(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    """Place VocaRig lip values into a full ARKit dictionary."""

    result = zero_arkit_dict()
    result.update(lip_values_to_dict(values))
    return result


def merge_face_and_lip(
    face_values: Mapping[str, float] | Sequence[float] | np.ndarray | None,
    lip_values: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Merge FaceRig expression channels and VocaRig lip articulation channels."""

    merged = zero_arkit_dict()
    face_dict = _face_values_to_dict(face_values)
    lip_dict = lip_values_to_dict(lip_values)

    for name, value in face_dict.items():
        if name == "mouthUpperUp":
            merged["mouthUpperUpLeft"] = max(merged["mouthUpperUpLeft"], value)
            merged["mouthUpperUpRight"] = max(merged["mouthUpperUpRight"], value)
        elif name in merged:
            merged[name] = max(merged[name], value)

    for name in LIP_OWNED_BLENDSHAPES:
        merged[name] = lip_dict[name]

    for name in SHARED_MOUTH_BLENDSHAPES:
        merged[name] = max(merged[name], lip_dict[name])

    return {name: float(np.clip(merged[name], 0.0, 1.0)) for name in ARKIT_BLENDSHAPE_NAMES}


def arkit_dict_to_vector(values: Mapping[str, float]) -> np.ndarray:
    """Return values aligned to ARKIT_BLENDSHAPE_NAMES."""

    return np.asarray(
        [float(np.clip(values.get(name, 0.0), 0.0, 1.0)) for name in ARKIT_BLENDSHAPE_NAMES],
        dtype=np.float32,
    )


def _face_values_to_dict(
    values: Mapping[str, float] | Sequence[float] | np.ndarray | None,
) -> dict[str, float]:
    if values is None:
        return {}
    if isinstance(values, Mapping):
        return {str(key): float(np.clip(value, 0.0, 1.0)) for key, value in values.items()}
    array = _validate_vector("face_values", values, len(FACE_RIG_BLENDSHAPE_NAMES))
    return {
        name: float(array[index])
        for index, name in enumerate(FACE_RIG_BLENDSHAPE_NAMES)
    }


def _validate_vector(name: str, values: Sequence[float] | np.ndarray, size: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return np.clip(array, 0.0, 1.0).astype(np.float32)
