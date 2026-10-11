"""Waveform augmentation, re-implemented from the paper's Table 2 with NumPy/librosa
(the original notebook uses the nlpaug library).

Applied ONLY to training recordings, AFTER the train/test split, and only to the
minority class (non-COPD), exactly one copy per technique (4 extra copies).
"""
from __future__ import annotations

import numpy as np
import librosa

from . import config as C
from .preprocess import fix_duration


def loudness(y, rng):
    return (y * rng.uniform(*C.LOUDNESS_RANGE)).astype(np.float32)


def mask(y, rng):
    y = y.copy()
    n = int(len(y) * C.MASK_FRACTION)
    start = rng.integers(0, len(y) - n)
    y[start:start + n] = 0.0
    return y


def shift(y, rng):
    k = int(rng.uniform(-C.MAX_SHIFT_S, C.MAX_SHIFT_S) * C.TARGET_SR)
    return np.roll(y, k)


def speed(y, rng):
    rate = rng.uniform(*C.SPEED_RANGE)
    return fix_duration(librosa.effects.time_stretch(y, rate=rate).astype(np.float32))


TECHNIQUES = {"loudness": loudness, "mask": mask, "shift": shift, "speed": speed}


def augment(y: np.ndarray, rng: np.random.Generator) -> list[tuple[str, np.ndarray]]:
    """Return one augmented copy per technique."""
    return [(name, fn(y, rng)) for name, fn in TECHNIQUES.items()]
