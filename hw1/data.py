"""Dataset indexing for the ICBHI 2017 Respiratory Sound Database.

Works with both download options:
  * Kaggle  "vbookshelf/respiratory-sound-database"  (audio_and_txt_files/*.wav + patient_diagnosis.csv)
  * Official ICBHI_final_database.zip                (*.wav + ICBHI_Challenge_diagnosis.txt)

ICBHI file name format:  <patient>_<recIndex>_<chestLoc>_<acqMode>_<device>.wav
e.g. 101_1b1_Al_sc_Meditron.wav

Known issue handled here: 91 files carry the wrong device name in the original
release (e.g. ..._AKGC417L instead of ..._Meditron). We therefore match files on the
first four name fields only and take the corrected device name from the official
split file (metadata/official_split.txt).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config as C


def _key(stem: str) -> str:
    """'101_1b1_Al_sc_Meditron' -> '101_1b1_Al_sc' (device-independent key)."""
    return stem.rsplit("_", 1)[0]


def parse_name(stem: str) -> dict:
    pid, rec, loc, mode, dev = stem.split("_")
    return {"patient_id": int(pid), "rec_index": rec, "chest_loc": loc,
            "acq_mode": mode, "device": dev}


def find_audio_files(data_dir: Path = C.DATA_DIR) -> dict[str, Path]:
    """Recursively find ICBHI wav files under data_dir. Returns {key: path}."""
    found = {}
    for p in Path(data_dir).rglob("*.wav"):
        parts = p.stem.split("_")
        if len(parts) == 5 and parts[0].isdigit():
            found[_key(p.stem)] = p
    return found


def load_diagnosis(data_dir: Path = C.DATA_DIR) -> pd.DataFrame:
    """Patient-level diagnosis table (patient_id, diagnosis)."""
    candidates = list(Path(data_dir).rglob("patient_diagnosis.csv")) + \
        list(Path(data_dir).rglob("ICBHI_Challenge_diagnosis.txt"))
    src = candidates[0] if candidates else C.META_DIR / "patient_diagnosis.txt"
    sep = "," if src.suffix == ".csv" else "\t"
    df = pd.read_csv(src, sep=sep, header=None, names=["patient_id", "diagnosis"])
    df["diagnosis"] = df["diagnosis"].str.strip()
    return df


def load_official_split() -> pd.DataFrame:
    df = pd.read_csv(C.META_DIR / "official_split.txt", sep="\t", header=None,
                     names=["file", "official_split"])
    df["file"] = df["file"].str.strip()
    df["official_split"] = df["official_split"].str.strip()
    return df


def build_index(data_dir: Path = C.DATA_DIR, require_audio: bool = True) -> pd.DataFrame:
    """One row per recording with labels, metadata and the patient-disjoint split.

    Columns: file, path, patient_id, rec_index, chest_loc, acq_mode, device,
             diagnosis, label (1 = COPD), label_name, official_split, split
    """
    split = load_official_split()
    meta = pd.DataFrame([parse_name(f) for f in split["file"]])
    df = pd.concat([split, meta], axis=1)
    df = df.merge(load_diagnosis(data_dir), on="patient_id", how="left")

    df["label"] = (df["diagnosis"] == C.POSITIVE_CLASS).astype(int)
    df["label_name"] = df["label"].map({1: C.POSITIVE_CLASS, 0: C.NEGATIVE_CLASS})

    # ---- make the official split strictly patient-disjoint ----------------
    # In the official 60/40 split, patients 156 and 218 have recordings in BOTH
    # partitions. Assignment rule: a patient with any training recording goes
    # entirely to train. This keeps every patient in exactly one partition.
    in_train = set(df.loc[df["official_split"] == "train", "patient_id"])
    df["split"] = df["patient_id"].map(lambda p: "train" if p in in_train else "test")

    audio = find_audio_files(data_dir)
    df["path"] = df["file"].map(lambda f: str(audio.get(_key(f), "")))
    if require_audio:
        missing = (df["path"] == "").sum()
        if missing == len(df):
            raise FileNotFoundError(
                f"No ICBHI .wav files found under {data_dir}. "
                "Download the dataset (see README) and extract it into the data/ folder.")
        if missing:
            print(f"[warning] {missing} recordings listed in the split file were not found and are skipped.")
        df = df[df["path"] != ""].reset_index(drop=True)
    return df


def overlapping_patients() -> list[int]:
    """Patients that appear in both partitions of the ORIGINAL official split."""
    s = load_official_split()
    s["patient_id"] = s["file"].str.split("_").str[0].astype(int)
    n = s.groupby("patient_id")["official_split"].nunique()
    return sorted(n[n > 1].index.tolist())
