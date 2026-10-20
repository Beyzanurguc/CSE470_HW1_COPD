"""Part 3 of the post-presentation improvements: same evaluation method for both experiments
+ checks whether the features (and the model) depend on the recording device.

    python consistency_study.py [--data-dir data]

A) Same 5-fold patient-grouped CV (StratifiedGroupKFold, seed 42, nested 3-fold tuning) for
     Exp. 1 (all 920 recordings, all devices) and Exp. 2 (214 Meditron + Littmann C2SE recordings).
   Exp. 1 is run without augmentation (identical to Exp. 2) and with augmentation of the non-COPD
   recordings of the TRAINING folds only (as in train.py). A device-only baseline is evaluated in
   every fold.
B) Device predictability: can the features tell which device recorded the sound?
     4 devices on all recordings, and Meditron vs Littmann C2SE on the Exp. 2 subset.
C) Exp. 2 results split per device: does the model separate COPD from non-COPD WITHIN one device?
D) Wavelet relative band energy per device and class (is the band-energy difference a device effect?)

Outputs: outputs/results_consistency.md|csv, outputs/figures/fig_consistency.png
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=FutureWarning)

from hw1 import config as C  # noqa: E402
from hw1.data import build_index  # noqa: E402
from hw1.evaluate import INK2, compute_metrics  # noqa: E402
from hw1.features import WAVELET_BANDS, feature_names  # noqa: E402
from hw1.models import fit_model, predict_proba_copd  # noqa: E402
from hw1.pipeline import build_feature_table, is_synthetic, matrix  # noqa: E402

SETS = ["mfcc", "spectral+wavelet", "mfcc+melspec+spectral+wavelet"]
COLS = ["accuracy", "recall", "specificity", "macro_f1", "auc", "icbhi_score"]
SEED = C.RANDOM_SEED


def to_markdown(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def device_rate(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    rate = train.groupby("device")["label"].mean()
    return test["device"].map(rate).fillna(train["label"].mean()).values


def grouped_cv(tab: pd.DataFrame, fs: str | None, use_aug: bool):
    """5-fold patient-grouped CV on the ORIGINAL recordings of tab. Augmented rows of the training
    patients are added to the training fold only. fs=None -> device-only baseline."""
    orig = tab[tab.version == "original"].reset_index(drop=True)
    y, g = orig["label"].values, orig["patient_id"].values
    oof, fold_acc = np.zeros(len(orig)), []
    outer = StratifiedGroupKFold(n_splits=C.DEVICE_MATCHED_FOLDS, shuffle=True, random_state=SEED)
    for tr, te in outer.split(np.zeros(len(y)), y, g):
        assert not set(g[tr]) & set(g[te])
        train_p = set(g[tr])
        train_rows = tab[tab.patient_id.isin(train_p) & ((tab.version == "original") | use_aug)]
        assert not set(train_rows.patient_id) & set(g[te])
        if fs is None:
            oof[te] = device_rate(orig.iloc[tr], orig.iloc[te])
        else:
            model, _, _ = fit_model("svm", matrix(train_rows, fs), train_rows["label"].values,
                                    train_rows["patient_id"].values, n_folds=3)
            oof[te] = predict_proba_copd(model, matrix(orig.iloc[te], fs))
        fold_acc.append(compute_metrics(y[te], oof[te])["accuracy"])
    return orig, oof, compute_metrics(y, oof), float(np.std(fold_acc, ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(C.DATA_DIR))
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    t0 = time.time()
    synthetic = is_synthetic(data_dir)
    df = build_index(data_dir)

    print("[1/4] Features (all recordings + augmented copies of every non-COPD recording)")
    all_tab = build_feature_table(df.assign(split="train"), (df["label"] == C.AUGMENT_CLASS).values, "allcv")
    sub = df[df.device.isin(C.DEVICE_MATCHED)].reset_index(drop=True)
    sub_tab = build_feature_table(sub.assign(split="train"), np.zeros(len(sub), bool), "devmatched")

    # ---------------------------------------------------------------- A) same CV for both experiments
    print("[2/4] Same 5-fold patient-grouped CV for both experiments")
    rows, oofs = [], {}
    plan = [("Exp. 1 · all devices", all_tab, fs, aug) for fs in SETS for aug in (False, True)] + \
           [("Exp. 1 · all devices", all_tab, None, False)] + \
           [("Exp. 2 · Meditron + Littmann C2SE", sub_tab, fs, False) for fs in SETS] + \
           [("Exp. 2 · Meditron + Littmann C2SE", sub_tab, None, False)]
    for exp, tab, fs, aug in plan:
        t = time.time()
        orig, oof, m, sd = grouped_cv(tab, fs, aug)
        name = "device only (no audio)" if fs is None else fs
        oofs[(exp, name, aug)] = (orig, oof)
        rows.append({"experiment": exp, "features": name, "augmentation": "yes" if aug else "no",
                     "recordings": len(orig), **{c: m[c] for c in COLS}, "acc_fold_std": round(sd, 3)})
        print(f"  {exp:<36} {name:<32} aug={'yes' if aug else 'no ':<3} acc={m['accuracy']:.3f}±{sd:.3f} "
              f"sens={m['recall']:.3f} ICBHI={m['icbhi_score']:.3f}  ({time.time() - t:.0f}s)")
    resA = pd.DataFrame(rows)

    # ---------------------------------------------------------------- B) can the features predict the device?
    print("[3/4] Device predictability")
    rowsB = []
    orig_all = all_tab[all_tab.version == "original"].reset_index(drop=True)
    for fs in SETS:
        clf = Pipeline([("scaler", StandardScaler()), ("svc", SVC(C=10, gamma="scale", class_weight="balanced"))])
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        d4 = orig_all["device"].to_numpy(dtype=object)
        d2 = sub_tab["device"].to_numpy(dtype=object)
        p4 = cross_val_predict(clf, matrix(orig_all, fs), d4, cv=cv, groups=orig_all["patient_id"].to_numpy())
        p2 = cross_val_predict(clf, matrix(sub_tab, fs), d2, cv=cv, groups=sub_tab["patient_id"].to_numpy())
        rowsB.append({"features": fs,
                      "4 devices, all recordings (chance 0.25)": round(balanced_accuracy_score(d4, p4), 3),
                      "Meditron vs Littmann C2SE (chance 0.50)": round(balanced_accuracy_score(d2, p2), 3)})
        print(f"  {fs:<32} 4-device bal.acc={rowsB[-1]['4 devices, all recordings (chance 0.25)']:.3f}  "
              f"2-device bal.acc={rowsB[-1]['Meditron vs Littmann C2SE (chance 0.50)']:.3f}")
    resB = pd.DataFrame(rowsB)

    # ---------------------------------------------------------------- C) Exp. 2 results within each device
    rowsC = []
    for fs in ["mfcc", "spectral+wavelet"]:
        orig, oof = oofs[("Exp. 2 · Meditron + Littmann C2SE", fs, False)]
        for dev in C.DEVICE_MATCHED:
            msk = (orig.device == dev).values
            m = compute_metrics(orig.label.values[msk], oof[msk])
            rowsC.append({"features": fs, "device": dev, "COPD": int(orig.label.values[msk].sum()),
                          "non-COPD": int((1 - orig.label.values[msk]).sum()),
                          **{c: m[c] for c in ["accuracy", "recall", "specificity", "auc", "icbhi_score"]}})
    resC = pd.DataFrame(rowsC)

    # ---------------------------------------------------------------- D) wavelet band energy per device and class
    names = feature_names("wavelet")
    W = matrix(sub_tab, "wavelet")
    rowsD = []
    for dev in C.DEVICE_MATCHED:
        for lab, lname in [(1, "COPD"), (0, "non-COPD")]:
            msk = ((sub_tab.device == dev) & (sub_tab.label == lab)).values
            rowsD.append({"device": dev, "class": lname, "n": int(msk.sum()),
                          **{b.split()[0]: round(float(W[msk, names.index(f"{b.split()[0]}_rel_energy")].mean()), 3)
                             if msk.any() else float("nan") for b in WAVELET_BANDS}})
    resD = pd.DataFrame(rowsD)

    # ---------------------------------------------------------------- figure + report
    print("[4/4] Report")
    pd.concat([resA.assign(table="A")], ignore_index=True).to_csv(C.OUT_DIR / "results_consistency.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    labels, e1, e2 = [], [], []
    for fs in SETS + ["device only (no audio)"]:
        r1 = resA[(resA.experiment.str.startswith("Exp. 1")) & (resA.features == fs) & (resA.augmentation == "no")]
        r2 = resA[(resA.experiment.str.startswith("Exp. 2")) & (resA.features == fs)]
        labels.append(fs.replace("mfcc+melspec+spectral+wavelet", "all 354").replace(" (no audio)", "\n(no audio)"))
        e1.append(float(r1.icbhi_score.iloc[0])); e2.append(float(r2.icbhi_score.iloc[0]))
    x = np.arange(len(labels)); w = 0.36
    for xs, vals, col, lab in [(x - w / 2 - .01, e1, "#2a78d6", "Exp. 1 · all devices"),
                               (x + w / 2 + .01, e2, "#eb6834", "Exp. 2 · Meditron + Littmann C2SE")]:
        bars = ax.bar(xs, vals, w, color=col, label=lab)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.axhline(0.5, color=INK2, ls="--", lw=1)
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("ICBHI score")
    ax.set_title("Same 5-fold patient-grouped CV for both experiments", pad=22)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="y", color="#ebe5d8", lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "fig_consistency.png"); plt.close(fig)

    warn = "> **WARNING: synthetic smoke-test data - do not report these numbers.**\n\n" if synthetic else ""
    md = ["# Part 3: same CV for both experiments + device checks\n", warn,
          f"All rows: {C.DEVICE_MATCHED_FOLDS}-fold StratifiedGroupKFold by patient (seed {SEED}), nested 3-fold "
          "tuning of the SVM, pooled out-of-fold metrics. Augmentation (if 'yes') only on non-COPD recordings of the "
          "training folds. recall = sensitivity.\n",
          "## A) Both experiments with the same CV\n", to_markdown(resA),
          "\n## B) Can the features predict the recording device? (balanced accuracy, patient-grouped 5-fold CV)\n",
          to_markdown(resB),
          "\n## C) Exp. 2 out-of-fold results within each device\n", to_markdown(resC),
          "\n## D) Wavelet relative band energy per device and class (Exp. 2 subset)\n", to_markdown(resD),
          f"\nRuntime: {time.time() - t0:.0f} s"]
    (C.OUT_DIR / "results_consistency.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
