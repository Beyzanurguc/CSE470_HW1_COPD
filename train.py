"""Homework steps 6-8: train classifiers and evaluate them honestly.

    python train.py [--data-dir data]

Experiment 1 (main, the paper's task)
    COPD vs non-COPD, all devices, patient-disjoint official ICBHI split.
    Training-only augmentation of non-COPD recordings (paper Table 2).
    Models: SVM / RF / MLP on MFCC vectors; SVM on log-mel and chroma vectors
    (mirrors the paper's feature comparison); SVM without augmentation (ablation);
    a "device-only" baseline that never listens to the audio.

Experiment 2 (confound control)
    Only Meditron + LittC2SE recordings (the only devices with both classes),
    5-fold patient-grouped cross-validation, SVM on MFCC, no augmentation.

Outputs: outputs/results_*.md|csv, outputs/metrics.json, outputs/test_predictions.csv,
         outputs/figures/*.png, models/*.joblib, models/model_card.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from hw1 import config as C
from hw1.data import build_index, overlapping_patients
from hw1.evaluate import compute_metrics, plot_confusion, plot_roc
from hw1.models import MODEL_LABELS, fit_model, predict_proba_copd
from hw1.pipeline import build_feature_table, is_synthetic, matrix

COLS = ["accuracy", "precision", "recall", "specificity", "f1", "macro_f1", "auc", "icbhi_score"]


def device_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """P(COPD | device) estimated on training recordings - uses NO audio at all."""
    rate = train_df.groupby("device")["label"].mean()
    return test_df["device"].map(rate).fillna(train_df["label"].mean()).values


def _portable(p: str) -> str:
    """Path relative to the project folder when possible (so the folder can be moved)."""
    p = Path(p).resolve()
    try:
        return p.relative_to(C.ROOT).as_posix()
    except ValueError:
        return str(p)


def to_markdown(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(C.DATA_DIR))
    ap.add_argument("--skip-exp2", action="store_true", help="skip the cross-validation experiment")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    t_start = time.time()
    synthetic = is_synthetic(data_dir)
    if synthetic:
        print("!! SYNTHETIC smoke-test data detected - results are meaningless !!")

    df = build_index(data_dir)
    train_df = df[df.split == "train"].reset_index(drop=True)
    test_df = df[df.split == "test"].reset_index(drop=True)

    # ---------------------------------------------------------- leakage checks (step 7)
    shared = set(train_df.patient_id) & set(test_df.patient_id)
    assert not shared, f"patients in both partitions: {shared}"
    assert not set(train_df.file) & set(test_df.file), "recording in both partitions"
    print(f"Train: {len(train_df)} recordings / {train_df.patient_id.nunique()} patients "
          f"(COPD {train_df.label.sum()}, non-COPD {(1 - train_df.label).sum()})")
    print(f"Test : {len(test_df)} recordings / {test_df.patient_id.nunique()} patients "
          f"(COPD {test_df.label.sum()}, non-COPD {(1 - test_df.label).sum()})")
    print("Patient overlap: none (checked).")

    # ---------------------------------------------------------- features
    print("\n[1/4] Features")
    aug_mask = (train_df["label"] == C.AUGMENT_CLASS).values
    train_tab = build_feature_table(train_df, aug_mask, "train")
    test_tab = build_feature_table(test_df, np.zeros(len(test_df), bool), "test")
    orig_train = train_tab[train_tab.version == "original"]
    print(f"  training vectors: {len(orig_train)} original + {len(train_tab) - len(orig_train)} augmented "
          f"-> COPD {train_tab.label.sum()}, non-COPD {(1 - train_tab.label).sum()}")
    y_test = test_tab["label"].values

    # ---------------------------------------------------------- experiment 1
    print("\n[2/4] Experiment 1 - official split (patient-disjoint)")
    runs = [  # (model, feature set, use augmentation)
        ("svm", "mfcc", True), ("rf", "mfcc", True), ("mlp", "mfcc", True),
        ("svm", "melspec", True), ("svm", "chroma", True), ("svm", "mfcc", False),
    ]
    rows, probs, params, saved = [], {}, {}, {}
    for model_name, fs, use_aug in runs:
        tab = train_tab if use_aug else orig_train
        X, y, g = matrix(tab, fs), tab["label"].values, tab["patient_id"].values
        t0 = time.time()
        model, best, cv_score = fit_model(model_name, X, y, g)
        p = predict_proba_copd(model, matrix(test_tab, fs))
        run_id = f"{model_name}_{fs}_{'aug' if use_aug else 'noaug'}"
        m = compute_metrics(y_test, p)
        rows.append({"run": run_id, "model": MODEL_LABELS[model_name], "features": fs,
                     "augmentation": "yes" if use_aug else "no", **{k: m[k] for k in COLS},
                     "cv_bal_acc_train": round(cv_score, 3)})
        probs[run_id], params[run_id] = p, best
        print(f"  {run_id:<20} acc={m['accuracy']:.3f} sens={m['recall']:.3f} spec={m['specificity']:.3f} "
              f"F1={m['f1']:.3f} AUC={m['auc']:.3f}  ({time.time() - t0:.0f}s, {best})")
        if fs == "mfcc" and use_aug:
            joblib.dump(model, C.MODEL_DIR / f"{model_name}_mfcc.joblib")
            saved[model_name] = {"file": f"{model_name}_mfcc.joblib", "label": MODEL_LABELS[model_name],
                                 "params": {k: str(v) for k, v in best.items()}, "test_metrics": m}

    p_dev = device_baseline(train_df, test_df)
    m = compute_metrics(y_test, p_dev)
    rows.append({"run": "device_only", "model": "Device-only baseline (no audio)", "features": "device name",
                 "augmentation": "-", **{k: m[k] for k in COLS}, "cv_bal_acc_train": float("nan")})
    probs["device_only"] = p_dev
    print(f"  {'device_only':<20} acc={m['accuracy']:.3f} sens={m['recall']:.3f} spec={m['specificity']:.3f}")

    res1 = pd.DataFrame(rows)
    res1.to_csv(C.OUT_DIR / "results_main.csv", index=False)

    # per-device accuracy of the main model on the test set
    main_run = "svm_mfcc_aug"
    pred_df = test_df.drop(columns=["path"]).copy()
    pred_df["path"] = [_portable(p) for p in test_df["path"]]   # used by the demo app
    for run_id, p in probs.items():
        pred_df[f"p_copd_{run_id}"] = np.round(p, 4)
    pred_df["pred_main"] = (pred_df[f"p_copd_{main_run}"] >= 0.5).astype(int)
    pred_df.to_csv(C.OUT_DIR / "test_predictions.csv", index=False)
    per_dev = (pred_df.assign(correct=lambda d: d.pred_main == d.label)
               .groupby(["device", "label_name"]).agg(recordings=("file", "size"), accuracy=("correct", "mean"))
               .round(3).reset_index())
    per_dev.to_csv(C.OUT_DIR / "per_device_main.csv", index=False)

    # figures
    plot_confusion(y_test, probs[main_run], "SVM + MFCC (augmented)", C.FIG_DIR / "cm_svm_mfcc_aug.png")
    plot_confusion(y_test, probs["svm_mfcc_noaug"], "SVM + MFCC (no augmentation)", C.FIG_DIR / "cm_svm_mfcc_noaug.png")
    plot_confusion(y_test, probs["rf_mfcc_aug"], "Random Forest + MFCC", C.FIG_DIR / "cm_rf_mfcc_aug.png")
    plot_confusion(y_test, probs["mlp_mfcc_aug"], "MLP + MFCC", C.FIG_DIR / "cm_mlp_mfcc_aug.png")
    plot_confusion(y_test, p_dev, "Device-only baseline", C.FIG_DIR / "cm_device_only.png")
    plot_roc({"SVM": (y_test, probs["svm_mfcc_aug"]), "Random Forest": (y_test, probs["rf_mfcc_aug"]),
              "MLP": (y_test, probs["mlp_mfcc_aug"]), "Device only": (y_test, p_dev)},
             "ROC - test set (MFCC features)", C.FIG_DIR / "roc_main.png")

    # ---------------------------------------------------------- experiment 2
    res2_summary, res2_folds = None, None
    if not args.skip_exp2:
        print(f"\n[3/4] Experiment 2 - devices {C.DEVICE_MATCHED} only, "
              f"{C.DEVICE_MATCHED_FOLDS}-fold patient-grouped CV")
        sub = df[df.device.isin(C.DEVICE_MATCHED)].reset_index(drop=True)
        sub_tab = build_feature_table(sub.assign(split="train"), np.zeros(len(sub), bool), "devmatched")
        X, y, g = matrix(sub_tab, "mfcc"), sub_tab["label"].values, sub_tab["patient_id"].values
        oof, oof_dev, fold_rows = np.zeros(len(y)), np.zeros(len(y)), []
        outer = StratifiedGroupKFold(n_splits=C.DEVICE_MATCHED_FOLDS, shuffle=True, random_state=C.RANDOM_SEED)
        for k, (tr, te) in enumerate(outer.split(X, y, g)):
            assert not set(g[tr]) & set(g[te])
            model, _, _ = fit_model("svm", X[tr], y[tr], g[tr], n_folds=3)  # nested tuning
            oof[te] = predict_proba_copd(model, X[te])
            oof_dev[te] = device_baseline(sub_tab.iloc[tr], sub_tab.iloc[te])
            fm = compute_metrics(y[te], oof[te])
            fold_rows.append({"fold": k + 1, "test_patients": len(set(g[te])), **{c: fm[c] for c in COLS}})
            print(f"  fold {k + 1}: acc={fm['accuracy']:.3f} sens={fm['recall']:.3f} spec={fm['specificity']:.3f}")
        res2_folds = pd.DataFrame(fold_rows)
        res2_folds.to_csv(C.OUT_DIR / "results_device_matched_folds.csv", index=False)
        m_svm, m_dev = compute_metrics(y, oof), compute_metrics(y, oof_dev)
        res2_summary = pd.DataFrame([
            {"model": "SVM + MFCC (pooled out-of-fold)", **{c: m_svm[c] for c in COLS}},
            {"model": "SVM + MFCC (mean of folds)", **res2_folds[COLS].mean().round(3).to_dict()},
            {"model": "SVM + MFCC (std of folds)", **res2_folds[COLS].std().round(3).to_dict()},
            {"model": "Device-only baseline (pooled)", **{c: m_dev[c] for c in COLS}},
        ])
        res2_summary.to_csv(C.OUT_DIR / "results_device_matched.csv", index=False)
        plot_confusion(y, oof, "Device-matched CV: SVM + MFCC", C.FIG_DIR / "cm_device_matched.png")

    # ---------------------------------------------------------- reports + model card
    print("\n[4/4] Writing reports")
    warn = "> **WARNING: synthetic smoke-test data - do not report these numbers.**\n\n" if synthetic else ""
    md = [f"# Results\n\n{warn}",
          "Positive class = COPD. recall = sensitivity. Threshold P(COPD) >= 0.5. "
          f"Test set: {len(test_df)} recordings from {test_df.patient_id.nunique()} unseen patients "
          f"(COPD {int(y_test.sum())}, non-COPD {int((1 - y_test).sum())}).\n",
          "## Experiment 1 - patient-disjoint official split\n",
          to_markdown(res1.drop(columns=["run"])),
          "\n### Main model (SVM + MFCC) accuracy per device on the test set\n", to_markdown(per_dev)]
    if res2_summary is not None:
        md += [f"\n## Experiment 2 - {' + '.join(C.DEVICE_MATCHED)} only, "
               f"{C.DEVICE_MATCHED_FOLDS}-fold patient-grouped CV ({len(y)} recordings)\n",
               to_markdown(res2_summary), "\n### Per fold\n", to_markdown(res2_folds)]
    (C.OUT_DIR / "results.md").write_text("\n".join(md), encoding="utf-8")

    card = {
        "task": f"{C.POSITIVE_CLASS} vs {C.NEGATIVE_CLASS} (disease label, recording level)",
        "dataset": "ICBHI 2017 Respiratory Sound Database",
        "synthetic_data": synthetic,
        "trained_on": f"{len(train_df)} training recordings, {train_df.patient_id.nunique()} patients "
                      f"(+ augmented non-COPD copies)",
        "test_set": f"{len(test_df)} recordings, {test_df.patient_id.nunique()} patients never seen in training",
        "patients_moved_to_train": overlapping_patients(),
        "preprocessing": {"target_sr": C.TARGET_SR, "bandpass_hz": C.BANDPASS_HZ, "filter_order": C.FILTER_ORDER,
                          "duration_s": C.DURATION_S, "normalisation": "DC removal + peak"},
        "features": {"type": "MFCC + delta, mean & std over time", "n_mfcc": C.N_MFCC, "n_mels": C.N_MELS,
                     "n_fft": C.N_FFT, "hop_length": C.HOP_LENGTH, "dim": 4 * C.N_MFCC},
        "threshold": 0.5,
        "models": saved,
        "main_model": "svm",
        "runtime_s": round(time.time() - t_start, 1),
    }
    (C.MODEL_DIR / "model_card.json").write_text(json.dumps(card, indent=2, default=str))
    (C.OUT_DIR / "metrics.json").write_text(json.dumps(
        {"experiment1": res1.to_dict(orient="records"), "best_params": params,
         "experiment2": None if res2_summary is None else res2_summary.to_dict(orient="records")},
        indent=2, default=str))
    print("\n".join(md))
    print(f"\nDone in {time.time() - t_start:.0f}s. Models in {C.MODEL_DIR}, results in {C.OUT_DIR}")


if __name__ == "__main__":
    main()
