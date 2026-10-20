"""Feature study (part 1 of the post-presentation improvements):
spectrogram descriptors, wavelet features and their combinations vs. the MFCC baseline.

    python feature_study.py [--data-dir data]

Same protocol as train.py, SVM (RBF) only, nothing tuned on test data:
  * Experiment 1: official ICBHI split (patient-disjoint), train-only augmentation, test set used once
  * Experiment 2: Meditron + Littmann Classic II SE only, 5-fold patient-grouped CV with nested tuning
Experiment 2 is the fairer comparison, because the recording device cannot give the answer away there.

Outputs: outputs/results_features.md|csv, outputs/wavelet_band_energy.csv,
         outputs/figures/fig_feature_study.png
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from hw1 import config as C
from hw1.data import build_index
from hw1.evaluate import INK2, compute_metrics
from hw1.features import COMBINED_SETS, WAVELET_BANDS, feature_names
from hw1.models import fit_model, predict_proba_copd
from hw1.pipeline import build_feature_table, is_synthetic, matrix

COLS = ["accuracy", "recall", "specificity", "f1", "macro_f1", "auc", "icbhi_score"]


def to_markdown(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def exp2_cv(tab: pd.DataFrame, fs: str):
    X, y, g = matrix(tab, fs), tab["label"].values, tab["patient_id"].values
    oof, accs = np.zeros(len(y)), []
    outer = StratifiedGroupKFold(n_splits=C.DEVICE_MATCHED_FOLDS, shuffle=True, random_state=C.RANDOM_SEED)
    for tr, te in outer.split(X, y, g):
        assert not set(g[tr]) & set(g[te])
        model, _, _ = fit_model("svm", X[tr], y[tr], g[tr], n_folds=3)     # nested tuning
        oof[te] = predict_proba_copd(model, X[te])
        accs.append(compute_metrics(y[te], oof[te])["accuracy"])
    return compute_metrics(y, oof), float(np.std(accs, ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(C.DATA_DIR))
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    t0 = time.time()
    synthetic = is_synthetic(data_dir)

    df = build_index(data_dir)
    train_df = df[df.split == "train"].reset_index(drop=True)
    test_df = df[df.split == "test"].reset_index(drop=True)
    assert not set(train_df.patient_id) & set(test_df.patient_id)

    print("[1/3] Features (cached after the first run)")
    train_tab = build_feature_table(train_df, (train_df["label"] == C.AUGMENT_CLASS).values, "train")
    test_tab = build_feature_table(test_df, np.zeros(len(test_df), bool), "test")
    sub = df[df.device.isin(C.DEVICE_MATCHED)].reset_index(drop=True)
    sub_tab = build_feature_table(sub.assign(split="train"), np.zeros(len(sub), bool), "devmatched")
    y_test = test_tab["label"].values

    print("[2/3] SVM per feature set")
    rows = []
    for fs in COMBINED_SETS:
        t = time.time()
        model, best, _ = fit_model("svm", matrix(train_tab, fs), train_tab["label"].values,
                                   train_tab["patient_id"].values)
        m1 = compute_metrics(y_test, predict_proba_copd(model, matrix(test_tab, fs)))
        m2, sd2 = exp2_cv(sub_tab, fs)
        rows.append({"features": fs, "n_features": len(feature_names(fs)),
                     **{f"E1_{k}": m1[k] for k in COLS}, **{f"E2_{k}": m2[k] for k in COLS},
                     "E2_acc_fold_std": round(sd2, 3)})
        print(f"  {fs:<32} n={len(feature_names(fs)):>3}  Exp1 acc={m1['accuracy']:.3f} ICBHI={m1['icbhi_score']:.3f}"
              f" | Exp2 acc={m2['accuracy']:.3f}±{sd2:.3f} sens={m2['recall']:.3f} ICBHI={m2['icbhi_score']:.3f}"
              f"  ({time.time() - t:.0f}s)")
    res = pd.DataFrame(rows)
    res.to_csv(C.OUT_DIR / "results_features.csv", index=False)

    # wavelet sub-band relative energy per class, on the device-matched subset (fair: same devices)
    W = matrix(sub_tab, "wavelet")
    names = feature_names("wavelet")
    rel = {b: W[:, names.index(f"{b.split()[0]}_rel_energy")] for b in WAVELET_BANDS}
    band = pd.DataFrame({"band": WAVELET_BANDS,
                         "COPD": [rel[b][sub_tab.label.values == 1].mean() for b in WAVELET_BANDS],
                         "non-COPD": [rel[b][sub_tab.label.values == 0].mean() for b in WAVELET_BANDS]}).round(4)
    band.to_csv(C.OUT_DIR / "wavelet_band_energy.csv", index=False)

    print("[3/3] Report")
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    yy = np.arange(len(res))
    ax.barh(yy - 0.2, res["E1_icbhi_score"], 0.38, color="#2a78d6", label="Exp. 1 · official split, all devices")
    ax.barh(yy + 0.2, res["E2_icbhi_score"], 0.38, color="#eb6834", label="Exp. 2 · Meditron + Littmann C2SE, 5-fold CV")
    for i, r in res.iterrows():
        ax.text(r["E1_icbhi_score"] + 0.01, i - 0.2, f"{r['E1_icbhi_score']:.2f}", va="center", fontsize=8)
        ax.text(r["E2_icbhi_score"] + 0.01, i + 0.2, f"{r['E2_icbhi_score']:.2f}", va="center", fontsize=8)
    ax.set_yticks(yy, [f"{f}  ({n})" for f, n in zip(res.features, res.n_features)], fontsize=8.5)
    ax.invert_yaxis()
    ax.axvline(0.5, color=INK2, ls="--", lw=1)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("ICBHI score = (sensitivity + specificity) / 2")
    ax.set_title("SVM per feature set (number of features)", pad=24)
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig_feature_study.png")
    plt.close(fig)

    short = res[["features", "n_features", "E1_accuracy", "E1_icbhi_score", "E1_auc",
                 "E2_accuracy", "E2_acc_fold_std", "E2_recall", "E2_specificity", "E2_macro_f1",
                 "E2_auc", "E2_icbhi_score"]]
    warn = "> **WARNING: synthetic smoke-test data - do not report these numbers.**\n\n" if synthetic else ""
    md = ["# Feature study: spectrogram + wavelet features\n", warn,
          "SVM (RBF), same protocol as train.py. E1 = Experiment 1 (official split, test set, "
          f"{len(test_df)} recordings). E2 = Experiment 2 ({len(sub)} Meditron + Littmann C2SE recordings, "
          f"{C.DEVICE_MATCHED_FOLDS}-fold patient-grouped CV, pooled out-of-fold). recall = sensitivity.\n",
          to_markdown(short),
          "\n## Wavelet sub-band relative energy (mean, Experiment 2 subset)\n", to_markdown(band),
          f"\nRuntime: {time.time() - t0:.0f} s"]
    (C.OUT_DIR / "results_features.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
