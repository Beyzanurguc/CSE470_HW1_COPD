"""Homework steps 3-5: inspect the dataset, visualise preprocessing and features.

    python inspect_data.py [--data-dir data]

Writes outputs/dataset_summary.md, outputs/recordings.csv and outputs/figures/fig_*.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import welch

from hw1 import config as C
from hw1.data import build_index, overlapping_patients
from hw1.evaluate import COLORS, INK2
from hw1.features import log_mel, mfcc, standardize_rows
from hw1.pipeline import is_synthetic
from hw1.preprocess import bandpass, fix_duration, load_audio, normalize, resample


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def grouped_bars(ax, table: pd.DataFrame, title: str, ylabel: str):
    """table: index = categories on x, columns = class names."""
    x = np.arange(len(table.index))
    w = 0.38
    for k, cls in enumerate(C.CLASS_NAMES):
        vals = table[cls].values if cls in table else np.zeros(len(x))
        bars = ax.bar(x + (k - 0.5) * w, vals, w - 0.03, color=COLORS[cls], label=cls)
        ax.bar_label(bars, fontsize=8, color=INK2, padding=2)
    ax.set_xticks(x, table.index)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.grid(axis="y", lw=0.6)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(C.DATA_DIR))
    args = ap.parse_args()
    data_dir = Path(args.data_dir)

    df = build_index(data_dir)
    info = [sf.info(p) for p in df["path"]]
    df["native_sr"] = [i.samplerate for i in info]
    df["duration_s"] = [round(i.duration, 2) for i in info]
    df.drop(columns=["path"]).to_csv(C.OUT_DIR / "recordings.csv", index=False)

    # ------------------------------------------------------------ tables
    by_dx = (df.groupby("diagnosis").agg(recordings=("file", "size"), patients=("patient_id", "nunique"))
             .sort_values("recordings", ascending=False).reset_index())
    by_cls = (df.groupby(["label_name", "split"]).agg(recordings=("file", "size"),
              patients=("patient_id", "nunique")).reset_index())
    dev = pd.crosstab(df["device"], df["label_name"]).reset_index()
    sr = pd.crosstab(df["device"], df["native_sr"]).reset_index()
    dur = df.groupby("label_name")["duration_s"].describe()[["count", "mean", "min", "50%", "max"]].round(1).reset_index()

    banner = ("> **WARNING: synthetic smoke-test data - do not report these numbers.**\n\n"
              if is_synthetic(data_dir) else "")
    md = [
        "# Dataset summary (ICBHI 2017 Respiratory Sound Database)\n", banner,
        f"- Recordings found: **{len(df)}** from **{df.patient_id.nunique()}** patients",
        f"- Task: **{C.POSITIVE_CLASS} vs {C.NEGATIVE_CLASS}** (patient-level diagnosis label for every recording)",
        f"- Split: official ICBHI train/test split made patient-disjoint; patients moved to train: {overlapping_patients()}",
        "\n## Recordings per diagnosis\n", md_table(by_dx),
        "\n## Classes per partition\n", md_table(by_cls),
        "\n## Recording device vs class (confound check)\n", md_table(dev),
        "\n## Native sampling rate per device (Hz)\n", md_table(sr),
        "\n## Duration (s) per class\n", md_table(dur),
    ]
    (C.OUT_DIR / "dataset_summary.md").write_text("\n".join(md), encoding="utf-8")

    # ------------------------------------------------------------ figures: dataset
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    rec = pd.crosstab(df["split"], df["label_name"]).reindex(["train", "test"])
    pat = df.groupby(["split", "label_name"])["patient_id"].nunique().unstack().reindex(["train", "test"])
    grouped_bars(axes[0], rec, "Recordings per class", "recordings")
    grouped_bars(axes[1], pat, "Patients per class", "patients")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig_class_distribution.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.4))
    grouped_bars(ax, pd.crosstab(df["device"], df["label_name"]), "Recording device per class", "recordings")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig_device_by_class.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    bins = np.arange(0, df["duration_s"].max() + 2, 2)
    for cls in C.CLASS_NAMES:
        ax.hist(df.loc[df.label_name == cls, "duration_s"], bins=bins, color=COLORS[cls],
                alpha=0.75, label=cls, edgecolor="white", linewidth=0.8)
    ax.axvline(C.DURATION_S, color=INK2, ls="--", lw=1)
    ax.text(C.DURATION_S + 0.5, ax.get_ylim()[1] * 0.9, f"fixed length {C.DURATION_S:.0f} s", color=INK2, fontsize=8)
    ax.set_xlabel("duration (s)")
    ax.set_ylabel("recordings")
    ax.set_title("Recording durations")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig_durations.png")
    plt.close(fig)

    # ------------------------------------------------------------ figures: one example per class
    # choose examples recorded with the SAME device so the difference is not a device effect
    examples = {}
    for cls in C.CLASS_NAMES:
        cand = df[(df.label_name == cls) & (df.device == "Meditron")]
        examples[cls] = (cand if len(cand) else df[df.label_name == cls]).iloc[0]

    fig, axes = plt.subplots(2, 2, figsize=(10, 4.6), sharex="col")
    for r, (cls, row) in enumerate(examples.items()):
        y, sr0 = load_audio(row["path"])
        y4 = resample(y, sr0)
        yb = bandpass(y4)
        yp = fix_duration(normalize(yb))
        t0 = np.arange(len(y)) / sr0
        tp = np.arange(len(yp)) / C.TARGET_SR
        axes[r, 0].plot(t0, y, lw=0.4, color=COLORS[cls])
        axes[r, 0].set_title(f"Raw - {cls}: {row['file']} ({sr0} Hz)", fontsize=9.5)
        axes[r, 1].plot(tp, yp, lw=0.4, color=COLORS[cls])
        axes[r, 1].set_title(f"Preprocessed ({C.TARGET_SR} Hz, {C.BANDPASS_HZ[0]:.0f}-{C.BANDPASS_HZ[1]:.0f} Hz, "
                             f"peak-norm, {C.DURATION_S:.0f} s)", fontsize=9.5)
        for a in axes[r]:
            a.set_ylabel("amplitude")
    for a in axes[1]:
        a.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig_waveforms.png")
    plt.close(fig)

    # spectrum before / after band-pass for one example
    row = examples[C.POSITIVE_CLASS]
    y, sr0 = load_audio(row["path"])
    y4 = resample(y, sr0)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for sig, lab, col in [(y4, "resampled (before filter)", "#c3c2b7"), (bandpass(y4), "after band-pass", COLORS[C.POSITIVE_CLASS])]:
        f, p = welch(sig, fs=C.TARGET_SR, nperseg=1024)
        ax.semilogy(f, p, lw=1.5, color=col, label=lab)
    for edge in C.BANDPASS_HZ:
        ax.axvline(edge, color=INK2, ls="--", lw=1)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("power spectral density")
    ax.set_title("Effect of the band-pass filter")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig_bandpass_spectrum.png")
    plt.close(fig)

    # features: log-mel and MFCC per class
    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    for r, (cls, row) in enumerate(examples.items()):
        y, sr0 = load_audio(row["path"])
        yp = fix_duration(normalize(bandpass(resample(y, sr0))))
        im = librosa.display.specshow(log_mel(yp), sr=C.TARGET_SR, hop_length=C.HOP_LENGTH, x_axis="time",
                                      y_axis="mel", fmin=C.FMIN, fmax=C.FMAX, ax=axes[r, 0], cmap="magma")
        axes[r, 0].set_title(f"Log-mel spectrogram - {cls}", fontsize=9.5)
        fig.colorbar(im, ax=axes[r, 0], format="%+.0f dB")
        im = librosa.display.specshow(standardize_rows(mfcc(yp)), sr=C.TARGET_SR, hop_length=C.HOP_LENGTH, x_axis="time",
                                      ax=axes[r, 1], cmap="coolwarm", vmin=-3, vmax=3)
        axes[r, 1].set_title(f"{C.N_MFCC} MFCCs (z-scored per coefficient) - {cls}", fontsize=9.5)
        axes[r, 1].set_ylabel("MFCC index")
        fig.colorbar(im, ax=axes[r, 1])
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig_features.png")
    plt.close(fig)

    print("\n".join(md))
    print(f"\nFigures written to {C.FIG_DIR}")


if __name__ == "__main__":
    main()
