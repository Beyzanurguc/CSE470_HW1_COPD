"""Glue code: run preprocessing + feature extraction for a whole index, with caching."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import config as C
from .augment import augment
from .features import FEATURE_SETS, extract
from .preprocess import preprocess_file


def is_synthetic(data_dir: Path) -> bool:
    return any(Path(data_dir).rglob("SYNTHETIC_DATA.txt"))


def _config_hash(extra: str) -> str:
    keys = ["TARGET_SR", "BANDPASS_HZ", "FILTER_ORDER", "DURATION_S", "N_FFT", "HOP_LENGTH",
            "N_MELS", "N_MFCC", "FMIN", "FMAX", "LOUDNESS_RANGE", "SPEED_RANGE",
            "MASK_FRACTION", "MAX_SHIFT_S", "RANDOM_SEED", "FEATURE_VERSION"]
    blob = json.dumps({k: getattr(C, k) for k in keys}, default=str) + extra
    return hashlib.md5(blob.encode()).hexdigest()[:10]


def _one(path: str, do_augment: bool, seed: int):
    """Features of the original recording (+ its augmented copies if requested)."""
    y = preprocess_file(path)
    rows = [("original", extract(y))]
    if do_augment:
        rng = np.random.default_rng(seed)
        rows += [(name, extract(ya)) for name, ya in augment(y, rng)]
    return rows


def build_feature_table(df: pd.DataFrame, augment_mask: np.ndarray, tag: str,
                        n_jobs: int = -1, verbose: bool = True) -> pd.DataFrame:
    """Return a long table: one row per (recording, version) with feature arrays.

    augment_mask[i] == True -> recording i also gets augmented copies.
    Only training recordings may be augmented (enforced by the caller and asserted here).
    """
    assert not (augment_mask & (df["split"].values == "test")).any(), "augmenting test data!"
    files = "|".join(df["file"]) + "|" + "".join("1" if a else "0" for a in augment_mask)
    cache = C.CACHE_DIR / f"features_{tag}_{_config_hash(files)}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)

    if verbose:
        print(f"  extracting features for {len(df)} recordings "
              f"(+{4 * int(augment_mask.sum())} augmented copies) ...")
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_one)(p, bool(a), C.RANDOM_SEED + i)
        for i, (p, a) in enumerate(zip(df["path"], augment_mask)))

    records = []
    for (_, row), rows in zip(df.iterrows(), results):
        for version, feats in rows:
            rec = {"file": row["file"], "patient_id": row["patient_id"], "label": row["label"],
                   "device": row["device"], "split": row["split"], "version": version}
            rec.update({f"feat_{k}": feats[k] for k in FEATURE_SETS})
            records.append(rec)
    table = pd.DataFrame.from_records(records)
    table.to_pickle(cache)
    return table


def matrix(table: pd.DataFrame, feature_set: str) -> np.ndarray:
    """Feature matrix; combined sets like "mfcc+wavelet" are concatenated column-wise."""
    return np.hstack([np.stack(table[f"feat_{part}"].values) for part in feature_set.split("+")])
