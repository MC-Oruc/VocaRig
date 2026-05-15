"""TR/EN phoneme and viseme targets for synthetic VocaRig data."""

from __future__ import annotations

import numpy as np

from vocarig.models.blendshapes import LIP_BLENDSHAPE_NAMES, LIP_INDEX


VISEME_NAMES = [
    "sil",
    "aa",
    "ee",
    "ih",
    "oh",
    "uu",
    "bmp",
    "fv",
    "l",
    "r",
    "s",
    "sh",
    "tdn",
    "kg",
    "wq",
    "y",
]

TR_PHONEMES = [
    "a",
    "e",
    "i",
    "ii",
    "o",
    "oe",
    "u",
    "ue",
    "p",
    "b",
    "m",
    "f",
    "v",
    "l",
    "r",
    "s",
    "z",
    "sh",
    "zh",
    "ch",
    "jh",
    "t",
    "d",
    "n",
    "k",
    "g",
    "y",
]

EN_PHONEMES = [
    "aa",
    "ae",
    "ah",
    "eh",
    "iy",
    "ih",
    "ow",
    "uw",
    "p",
    "b",
    "m",
    "f",
    "v",
    "l",
    "r",
    "s",
    "z",
    "sh",
    "zh",
    "ch",
    "jh",
    "t",
    "d",
    "n",
    "k",
    "g",
    "y",
    "w",
    "th",
    "dh",
]

PHONEME_TO_VISEME = {
    "sil": "sil",
    "a": "aa",
    "aa": "aa",
    "ae": "aa",
    "ah": "aa",
    "e": "ee",
    "eh": "ee",
    "iy": "ee",
    "i": "ih",
    "ii": "ih",
    "ih": "ih",
    "o": "oh",
    "oe": "oh",
    "ow": "oh",
    "u": "uu",
    "ue": "uu",
    "uw": "uu",
    "p": "bmp",
    "b": "bmp",
    "m": "bmp",
    "f": "fv",
    "v": "fv",
    "th": "fv",
    "dh": "fv",
    "l": "l",
    "r": "r",
    "s": "s",
    "z": "s",
    "sh": "sh",
    "zh": "sh",
    "ch": "sh",
    "jh": "sh",
    "t": "tdn",
    "d": "tdn",
    "n": "tdn",
    "k": "kg",
    "g": "kg",
    "w": "wq",
    "q": "wq",
    "y": "y",
}


def viseme_target(viseme: str, style: np.ndarray | None = None) -> np.ndarray:
    """Return a 21-wide target vector for a viseme."""

    values = np.zeros(len(LIP_BLENDSHAPE_NAMES), dtype=np.float32)
    style_values = np.asarray([0.5, 0.5] if style is None else style, dtype=np.float32)
    intensity = 0.85 + float(style_values[0]) * 0.28
    asymmetry = (float(style_values[1]) - 0.5) * 0.08

    def setv(name: str, value: float) -> None:
        values[LIP_INDEX[name]] = max(values[LIP_INDEX[name]], value * intensity)

    if viseme == "sil":
        setv("mouthClose", 0.18)
        setv("mouthPressLeft", 0.08)
        setv("mouthPressRight", 0.08)
    elif viseme == "aa":
        setv("jawOpen", 0.82)
        setv("mouthLowerDownLeft", 0.40)
        setv("mouthLowerDownRight", 0.40)
        setv("mouthStretchLeft", 0.15)
        setv("mouthStretchRight", 0.15)
    elif viseme == "ee":
        setv("jawOpen", 0.28)
        setv("mouthStretchLeft", 0.68)
        setv("mouthStretchRight", 0.68)
        setv("mouthUpperUpLeft", 0.16)
        setv("mouthUpperUpRight", 0.16)
    elif viseme == "ih":
        setv("jawOpen", 0.36)
        setv("mouthStretchLeft", 0.40)
        setv("mouthStretchRight", 0.40)
        setv("mouthLowerDownLeft", 0.20)
        setv("mouthLowerDownRight", 0.20)
    elif viseme == "oh":
        setv("jawOpen", 0.54)
        setv("mouthFunnel", 0.62)
        setv("mouthPucker", 0.20)
        setv("mouthShrugUpper", 0.20)
    elif viseme == "uu":
        setv("jawOpen", 0.22)
        setv("mouthPucker", 0.82)
        setv("mouthFunnel", 0.46)
        setv("mouthShrugUpper", 0.25)
    elif viseme == "bmp":
        setv("mouthClose", 0.92)
        setv("mouthPressLeft", 0.64)
        setv("mouthPressRight", 0.64)
        setv("jawOpen", 0.04)
    elif viseme == "fv":
        setv("mouthClose", 0.44)
        setv("mouthLowerDownLeft", 0.26)
        setv("mouthLowerDownRight", 0.26)
        setv("mouthStretchLeft", 0.18)
        setv("mouthStretchRight", 0.18)
    elif viseme == "l":
        setv("jawOpen", 0.34)
        setv("mouthUpperUpLeft", 0.28)
        setv("mouthUpperUpRight", 0.28)
        setv("mouthShrugUpper", 0.24)
    elif viseme == "r":
        setv("jawOpen", 0.24)
        setv("mouthPucker", 0.22)
        setv("mouthShrugUpper", 0.30)
        setv("mouthRollUpper", 0.22)
    elif viseme == "s":
        setv("jawOpen", 0.16)
        setv("mouthStretchLeft", 0.32)
        setv("mouthStretchRight", 0.32)
        setv("mouthClose", 0.18)
    elif viseme == "sh":
        setv("jawOpen", 0.26)
        setv("mouthFunnel", 0.42)
        setv("mouthPucker", 0.40)
    elif viseme == "tdn":
        setv("jawOpen", 0.24)
        setv("mouthUpperUpLeft", 0.16)
        setv("mouthUpperUpRight", 0.16)
        setv("mouthClose", 0.20)
    elif viseme == "kg":
        setv("jawOpen", 0.30)
        setv("mouthLowerDownLeft", 0.18)
        setv("mouthLowerDownRight", 0.18)
    elif viseme == "wq":
        setv("jawOpen", 0.18)
        setv("mouthPucker", 0.74)
        setv("mouthFunnel", 0.55)
    elif viseme == "y":
        setv("jawOpen", 0.18)
        setv("mouthStretchLeft", 0.46)
        setv("mouthStretchRight", 0.46)
        setv("mouthUpperUpLeft", 0.12)
        setv("mouthUpperUpRight", 0.12)
    else:
        raise ValueError(f"Unknown viseme: {viseme}")

    # Small synthetic asymmetry keeps left/right pairs from being identical.
    for left, right in [
        ("jawLeft", "jawRight"),
        ("mouthLowerDownLeft", "mouthLowerDownRight"),
        ("mouthUpperUpLeft", "mouthUpperUpRight"),
        ("mouthPressLeft", "mouthPressRight"),
        ("mouthStretchLeft", "mouthStretchRight"),
    ]:
        values[LIP_INDEX[left]] *= 1.0 + asymmetry
        values[LIP_INDEX[right]] *= 1.0 - asymmetry

    return np.clip(values, 0.0, 1.0).astype(np.float32)


def phoneme_to_viseme(phoneme: str) -> str:
    """Map a TR/EN phoneme token to a viseme token."""

    return PHONEME_TO_VISEME.get(phoneme, "sil")
