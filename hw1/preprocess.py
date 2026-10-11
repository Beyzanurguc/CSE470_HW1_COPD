"""Signal preprocessing (homework step 4).

Order:  load (native rate) -> mono -> resample to 4 kHz -> band-pass 100-1800 Hz
        -> remove DC + peak-normalise -> trim / wrap-pad to 20 s
"""
from __future__ import annotations

import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt

from . import config as C


def load_audio(path) -> tuple[np.ndarray, int]:
    """Load at the file's native sampling rate, mixed down to mono."""
    y, sr = librosa.load(path, sr=None, mono=True)
    return y.astype(np.float32), int(sr)


def resample(y: np.ndarray, sr: int, target_sr: int = C.TARGET_SR) -> np.ndarray:
    if sr == target_sr:
        return y
    return librosa.resample(y, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")


def bandpass(y: np.ndarray, sr: int = C.TARGET_SR, band=C.BANDPASS_HZ,
             order: int = C.FILTER_ORDER) -> np.ndarray:
    """Zero-phase Butterworth band-pass (sosfiltfilt keeps crackle timing intact)."""
    sos = butter(order, band, btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


def normalize(y: np.ndarray) -> np.ndarray:
    """Remove DC offset and scale to peak amplitude 1 (devices have very different gains)."""
    y = y - np.mean(y)
    peak = np.max(np.abs(y))
    return (y / peak).astype(np.float32) if peak > 0 else y


def fix_duration(y: np.ndarray, sr: int = C.TARGET_SR, seconds: float = C.DURATION_S) -> np.ndarray:
    """Trim to the first `seconds`, or repeat the signal (wrap padding) if it is shorter.

    The paper zero-pads; wrap padding avoids long silent tails that a classifier could
    use as a shortcut (short recordings are not evenly distributed across classes).
    """
    n = int(round(sr * seconds))
    if len(y) >= n:
        return y[:n]
    return np.pad(y, (0, n - len(y)), mode="wrap")


def preprocess_signal(y: np.ndarray, sr: int) -> np.ndarray:
    y = resample(y, sr)
    y = bandpass(y)
    y = normalize(y)
    return fix_duration(y)


def preprocess_file(path) -> np.ndarray:
    y, sr = load_audio(path)
    return preprocess_signal(y, sr)
