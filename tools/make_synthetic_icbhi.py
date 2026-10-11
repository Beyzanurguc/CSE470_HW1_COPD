"""SMOKE-TEST ONLY: create a fake dataset with the exact ICBHI file names, patients and
labels, but SYNTHETIC audio (filtered noise + artificial wheezes/crackles).

Purpose: check that inspect_data.py -> train.py -> app.py run end-to-end before the
real dataset is downloaded. Results on this data are meaningless and must NOT be
reported. The folder contains SYNTHETIC_DATA.txt, and the app shows a warning banner
when its models were trained on it.

    python tools/make_synthetic_icbhi.py            # writes ./synthetic_data/
    python train.py --data-dir synthetic_data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hw1 import config as C  # noqa: E402
from hw1.data import load_official_split  # noqa: E402

# Smaller rates than the real ones (44.1 kHz) to keep the fake dataset small;
# still exercises resampling (8 kHz -> 4 kHz).
FAKE_SR = {"AKGC417L": 8000, "LittC2SE": 8000, "Meditron": 4000, "Litt3200": 4000}


def synth(sr, dur, copd, device, rng):
    t = np.arange(int(sr * dur)) / sr
    period = rng.uniform(3, 5)
    env = np.abs(np.sin(np.pi * t / period)) ** 1.5
    sos = butter(4, (100, min(1000, sr / 2 - 50)), btype="bandpass", fs=sr, output="sos")
    y = sosfilt(sos, rng.normal(size=len(t))) * env
    if rng.random() < (0.6 if copd else 0.2):            # wheeze-like tone on expiration
        f0 = rng.uniform(200, 600)
        y += 0.4 * np.sin(2 * np.pi * f0 * t) * (np.sin(2 * np.pi * t / period) < 0) * env
    for c in rng.integers(0, len(t) - 40, size=rng.poisson(30 if copd else 10)):  # crackles
        y[c:c + 40] += rng.uniform(1, 3) * np.hanning(40) * np.sin(np.linspace(0, 6 * np.pi, 40))
    if device in ("AKGC417L", "LittC2SE"):              # device colouring (like real data)
        y += 0.05 * rng.normal(size=len(t))
    return (0.3 * y / np.max(np.abs(y))).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(C.ROOT / "synthetic_data"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    root = Path(args.out)
    audio_dir = root / "Respiratory_Sound_Database" / "audio_and_txt_files"
    audio_dir.mkdir(parents=True, exist_ok=True)
    (root / "SYNTHETIC_DATA.txt").write_text("Synthetic smoke-test data. Do not report results.\n")

    diag = {}
    for line in (C.META_DIR / "patient_diagnosis.txt").read_text().splitlines():
        pid, d = line.split("\t")
        diag[int(pid)] = d.strip()
    (root / "Respiratory_Sound_Database" / "patient_diagnosis.csv").write_text(
        "".join(f"{p},{d}\n" for p, d in diag.items()))

    # reproduce the original (uncorrected) device names to test the renaming logic
    wrong = {l.strip().strip("'").rsplit("_", 1)[0]: l.strip().strip("'")
             for l in (C.META_DIR / "filename_differences.txt").read_text().splitlines() if l.strip()}
    for f in load_official_split()["file"]:
        dev = f.rsplit("_", 1)[1]
        name = wrong.get(f.rsplit("_", 1)[0], f)
        dur = 20.0 if rng.random() < 0.85 else float(rng.uniform(8, 30))
        sr = FAKE_SR[dev]
        sf.write(audio_dir / f"{name}.wav", synth(sr, dur, diag[int(f[:3])] == "COPD", dev, rng), sr)
    print(f"Synthetic dataset written to {root}")


if __name__ == "__main__":
    main()
