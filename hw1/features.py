"""Feature extraction (homework step 5).

The paper feeds 2-D feature maps (40 x 862) into a CNN. We use the same time-frequency
representations but summarise each one over time (mean + standard deviation per
coefficient), which gives a fixed-length vector suitable for SVM / RF / MLP.

Base feature sets (compared like the paper's Table 5/6):
  mfcc      40 MFCCs + their deltas -> mean & std            = 160 values
  melspec   64 log-mel bands        -> mean & std            = 128 values
  chroma    12 chroma_stft bins      -> mean & std            =  24 values

Added after the HW1 presentation feedback ("spectrogram, wavelet and their combination"):
  spectral  12 spectrogram descriptors -> mean & std          =  24 values
            centroid, bandwidth, roll-off, flatness, flux, ZCR, RMS, 5 contrast bands
  wavelet   5-level db4 DWT, 6 sub-bands x 7 statistics      =  42 values
            energy (log), relative energy, mean |c|, std, skewness, kurtosis, entropy

Combined sets are built by concatenating base sets, e.g. "mfcc+wavelet" (see COMBINED_SETS).
"""
from __future__ import annotations

import numpy as np
import librosa
import pywt
from scipy.stats import kurtosis, skew

from . import config as C

FEATURE_SETS = ["mfcc", "melspec", "chroma", "spectral", "wavelet"]

# feature sets compared in the feature study of train.py (base sets and concatenations)
COMBINED_SETS = ["mfcc", "spectral", "wavelet", "mfcc+spectral", "mfcc+wavelet",
                 "spectral+wavelet", "mfcc+spectral+wavelet", "mfcc+melspec+spectral+wavelet"]

WAVELET = "db4"
WAVELET_LEVEL = 5
# sub-band frequency ranges at 4 kHz (Nyquist 2 kHz), in pywt.wavedec order [A5, D5, D4, D3, D2, D1]
WAVELET_BANDS = ["A5 0-62 Hz", "D5 62-125 Hz", "D4 125-250 Hz", "D3 250-500 Hz",
                 "D2 500-1000 Hz", "D1 1000-2000 Hz"]
WAVELET_STATS = ["log_energy", "rel_energy", "mean_abs", "std", "skew", "kurtosis", "entropy"]
SPECTRAL_NAMES = ["centroid", "bandwidth", "rolloff", "flatness", "flux", "zcr", "rms"] + \
                 [f"contrast_{i}" for i in range(5)]


def log_mel(y: np.ndarray, sr: int = C.TARGET_SR) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=C.N_FFT, hop_length=C.HOP_LENGTH,
                                       n_mels=C.N_MELS, fmin=C.FMIN, fmax=C.FMAX)
    return librosa.power_to_db(S, ref=np.max)


def mfcc(y: np.ndarray, sr: int = C.TARGET_SR) -> np.ndarray:
    return librosa.feature.mfcc(S=log_mel(y, sr), n_mfcc=C.N_MFCC)


def chroma(y: np.ndarray, sr: int = C.TARGET_SR) -> np.ndarray:
    return librosa.feature.chroma_stft(y=y, sr=sr, n_fft=C.N_FFT, hop_length=C.HOP_LENGTH)


def spectral(y: np.ndarray, sr: int = C.TARGET_SR) -> np.ndarray:
    """12 frame-wise spectrogram descriptors -> array (12, frames)."""
    S = np.abs(librosa.stft(y, n_fft=C.N_FFT, hop_length=C.HOP_LENGTH))
    rows = [
        librosa.feature.spectral_centroid(S=S, sr=sr),               # where the energy is
        librosa.feature.spectral_bandwidth(S=S, sr=sr),              # how spread it is
        librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85),
        np.log10(librosa.feature.spectral_flatness(S=S) + 1e-10),    # noise-like (0) vs tonal (wheeze)
        librosa.onset.onset_strength(S=librosa.amplitude_to_db(S, ref=np.max), sr=sr)[None, :],  # flux
        librosa.feature.zero_crossing_rate(y, frame_length=C.N_FFT, hop_length=C.HOP_LENGTH),
        librosa.feature.rms(S=S, frame_length=C.N_FFT),
        # octave bands 62.5-125-250-500-1000-2000 Hz -> 5 contrast rows (peak vs valley)
        librosa.feature.spectral_contrast(S=S, sr=sr, fmin=62.5, n_bands=4),
    ]
    T = min(r.shape[-1] for r in rows)
    return np.vstack([r[..., :T] for r in rows])


def wavelet(y: np.ndarray) -> np.ndarray:
    """Discrete wavelet transform statistics -> vector of 6 bands x 7 values = 42."""
    coeffs = pywt.wavedec(y, WAVELET, level=WAVELET_LEVEL)
    energies = np.array([np.sum(c ** 2) for c in coeffs])
    total = energies.sum() + 1e-12
    out = []
    for c, e in zip(coeffs, energies):
        p = c ** 2 / (e + 1e-12)
        entropy = -np.sum(p * np.log(p + 1e-12)) / np.log(len(c))   # normalised Shannon entropy (0-1)
        out += [np.log10(e + 1e-12), e / total, np.mean(np.abs(c)), np.std(c),
                skew(c), kurtosis(c), entropy]
    return np.nan_to_num(np.array(out, dtype=np.float64))


def standardize_rows(M: np.ndarray) -> np.ndarray:
    """Per-coefficient z-score - for DISPLAY only (otherwise MFCC 0 dominates the colour scale)."""
    return (M - M.mean(axis=1, keepdims=True)) / (M.std(axis=1, keepdims=True) + 1e-8)


def _stats(M: np.ndarray) -> np.ndarray:
    return np.concatenate([M.mean(axis=1), M.std(axis=1)])


def extract(y: np.ndarray, sr: int = C.TARGET_SR) -> dict[str, np.ndarray]:
    """All base feature vectors for one preprocessed signal."""
    M = mfcc(y, sr)
    return {
        "mfcc": np.concatenate([_stats(M), _stats(librosa.feature.delta(M))]).astype(np.float32),
        "melspec": _stats(log_mel(y, sr)).astype(np.float32),
        "chroma": _stats(chroma(y, sr)).astype(np.float32),
        "spectral": np.nan_to_num(_stats(spectral(y, sr))).astype(np.float32),
        "wavelet": wavelet(y).astype(np.float32),
    }


def feature_names(name: str) -> list[str]:
    """Human-readable names of every value in a (possibly combined) feature set."""
    names = []
    for part in name.split("+"):
        if part == "mfcc":
            names += [f"mfcc{i}_{s}" for s in ("mean", "std") for i in range(C.N_MFCC)]
            names += [f"dmfcc{i}_{s}" for s in ("mean", "std") for i in range(C.N_MFCC)]
        elif part == "melspec":
            names += [f"mel{i}_{s}" for s in ("mean", "std") for i in range(C.N_MELS)]
        elif part == "chroma":
            names += [f"chroma{i}_{s}" for s in ("mean", "std") for i in range(12)]
        elif part == "spectral":
            names += [f"{n}_{s}" for s in ("mean", "std") for n in SPECTRAL_NAMES]
        elif part == "wavelet":
            names += [f"{b.split()[0]}_{st}" for b in WAVELET_BANDS for st in WAVELET_STATS]
    return names
