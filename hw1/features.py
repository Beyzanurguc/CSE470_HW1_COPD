"""Feature extraction (homework step 5).

The paper feeds 2-D feature maps (40 x 862) into a CNN. We use the same time-frequency
representations but summarise each one over time (mean + standard deviation per
coefficient), which gives a fixed-length vector suitable for SVM / RF / MLP.

Feature sets (compared like the paper's Table 5/6):
  mfcc     40 MFCCs + their deltas -> mean & std  = 160 values
  melspec  64 log-mel bands        -> mean & std  = 128 values
  chroma   12 chroma_stft bins      -> mean & std  =  24 values
"""
from __future__ import annotations

import numpy as np
import librosa

from . import config as C

FEATURE_SETS = ["mfcc", "melspec", "chroma"]


def log_mel(y: np.ndarray, sr: int = C.TARGET_SR) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=C.N_FFT, hop_length=C.HOP_LENGTH,
                                       n_mels=C.N_MELS, fmin=C.FMIN, fmax=C.FMAX)
    return librosa.power_to_db(S, ref=np.max)


def mfcc(y: np.ndarray, sr: int = C.TARGET_SR) -> np.ndarray:
    return librosa.feature.mfcc(S=log_mel(y, sr), n_mfcc=C.N_MFCC)


def chroma(y: np.ndarray, sr: int = C.TARGET_SR) -> np.ndarray:
    return librosa.feature.chroma_stft(y=y, sr=sr, n_fft=C.N_FFT, hop_length=C.HOP_LENGTH)


def standardize_rows(M: np.ndarray) -> np.ndarray:
    """Per-coefficient z-score - for DISPLAY only (otherwise MFCC 0 dominates the colour scale)."""
    return (M - M.mean(axis=1, keepdims=True)) / (M.std(axis=1, keepdims=True) + 1e-8)


def _stats(M: np.ndarray) -> np.ndarray:
    return np.concatenate([M.mean(axis=1), M.std(axis=1)])


def extract(y: np.ndarray, sr: int = C.TARGET_SR) -> dict[str, np.ndarray]:
    """All feature vectors for one preprocessed signal."""
    M = mfcc(y, sr)
    return {
        "mfcc": np.concatenate([_stats(M), _stats(librosa.feature.delta(M))]).astype(np.float32),
        "melspec": _stats(log_mel(y, sr)).astype(np.float32),
        "chroma": _stats(chroma(y, sr)).astype(np.float32),
    }
