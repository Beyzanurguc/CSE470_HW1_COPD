"""Feature selection study (part 2 of the post-presentation improvements).

    python selection_study.py [--data-dir data]

Starts from ALL features (mfcc + melspec + spectral + wavelet = 354 values) and reduces them with
  * ANOVA F-test          SelectKBest(f_classif)            - filter
  * Mutual information    SelectKBest(mutual_info_classif)  - filter
  * Random-forest ranking SelectFromModel(RF, top k)        - embedded
  * L1 logistic regression SelectFromModel(L1-LR)           - embedded (k decided by the L1 penalty)
  * PCA                   (not a selection, a compression)  - for comparison
followed by the same calibrated RBF-SVM as train.py.

NO LEAKAGE: the selector is a step INSIDE the scikit-learn Pipeline (scaler -> selector -> SVM), so it
is re-fitted on the training part of every CV fold and never sees test data. "tuned" rows choose k
with the inner patient-grouped CV; fixed-k rows show the whole curve.

Same two experiments as train.py / feature_study.py:
  Exp. 1  official split (patient-disjoint), train-only augmentation, test set used once
  Exp. 2  Meditron + Littmann C2SE only, 5-fold patient-grouped CV, nested tuning  <- main criterion

Outputs: outputs/results_selection.md|csv, outputs/selected_features.csv, outputs/selection_groups.csv,
         outputs/figures/fig_selection_curve.png, outputs/figures/fig_selection_groups.png
"""
from __future__ import annotations

import argparse
import time
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from hw1 import config as C
from hw1.data import build_index
from hw1.evaluate import INK2, compute_metrics
from hw1.features import feature_names
from hw1.pipeline import build_feature_table, is_synthetic, matrix

BASE = "mfcc+melspec+spectral+wavelet"
KS = [20, 40, 80, 160]
PCA_KS = [20, 40, 80]      # PCA cannot keep more components than training samples in the inner folds (~115)
COLS = ["accuracy", "recall", "specificity", "macro_f1", "auc", "icbhi_score"]
SEED = C.RANDOM_SEED
GROUP_COLORS = {"mfcc": "#2a78d6", "melspec": "#1baf7a", "spectral": "#eb6834", "wavelet": "#6250d6"}


def group_of(name: str) -> str:
    p = name.split("_")[0]
    if p.startswith(("mfcc", "dmfcc")):
        return "mfcc"
    if p.startswith("mel"):
        return "melspec"
    if p in ("A5", "D5", "D4", "D3", "D2", "D1"):
        return "wavelet"
    return "spectral"


def selector(method: str, k: int | None):
    if method == "anova":
        return SelectKBest(f_classif, k=k)
    if method == "mi":
        return SelectKBest(partial(mutual_info_classif, random_state=SEED), k=k)
    if method == "rf":
        rf = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)
        return SelectFromModel(rf, max_features=k, threshold=-np.inf)
    if method == "l1":
        lr = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, class_weight="balanced", max_iter=5000)
        return SelectFromModel(lr)
    if method == "pca":
        return PCA(n_components=k, random_state=SEED)
    return "passthrough"


def make_search(method: str, k, n_folds: int):
    """k=int -> fixed; k='tuned' -> k searched in the inner CV."""
    svm = CalibratedClassifierCV(SVC(kernel="rbf", gamma="scale", class_weight="balanced", random_state=SEED),
                                 method="sigmoid", cv=3, ensemble=False)
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("sel", selector(method, KS[0] if k == "tuned" else k)), ("clf", svm)])
    ks = PCA_KS if method == "pca" else KS
    grid = {"clf__estimator__C": [1, 10, 100]}
    if k == "tuned":
        if method in ("anova", "mi", "pca"):
            grid["sel__k" if method != "pca" else "sel__n_components"] = ks
        elif method == "rf":
            grid["sel__max_features"] = ks
        elif method == "l1":
            grid["sel__estimator__C"] = [0.05, 0.1, 0.5]
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    return GridSearchCV(pipe, grid, cv=cv, scoring="balanced_accuracy", n_jobs=-1, refit=True)


def support(fitted) -> np.ndarray | None:
    sel = fitted.best_estimator_.named_steps["sel"]
    return sel.get_support() if hasattr(sel, "get_support") else None


def n_kept(fitted, n_all):
    sel = fitted.best_estimator_.named_steps["sel"]
    if hasattr(sel, "get_support"):
        return int(sel.get_support().sum())
    if isinstance(sel, PCA):
        return int(sel.n_components_)
    return n_all


def to_markdown(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


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
    print("[1/3] Features (cached by feature_study.py)")
    train_tab = build_feature_table(train_df, (train_df["label"] == C.AUGMENT_CLASS).values, "train")
    test_tab = build_feature_table(test_df, np.zeros(len(test_df), bool), "test")
    sub = df[df.device.isin(C.DEVICE_MATCHED)].reset_index(drop=True)
    sub_tab = build_feature_table(sub.assign(split="train"), np.zeros(len(sub), bool), "devmatched")

    names = np.array(feature_names(BASE))
    groups = np.array([group_of(n) for n in names])
    X1, y1, g1 = matrix(train_tab, BASE), train_tab["label"].values, train_tab["patient_id"].values
    Xt, yt = matrix(test_tab, BASE), test_tab["label"].values
    X2, y2, g2 = matrix(sub_tab, BASE), sub_tab["label"].values, sub_tab["patient_id"].values
    assert X1.shape[1] == len(names), (X1.shape, len(names))

    configs = [("none", None)] + [(m, k) for m in ("anova", "mi", "rf") for k in KS + ["tuned"]] + \
              [("pca", k) for k in PCA_KS + ["tuned"]] + \
              [("l1", None), ("l1", "tuned")]
    label = {"none": "No selection", "anova": "ANOVA F", "mi": "Mutual information", "rf": "Random-forest rank",
             "l1": "L1 logistic regression", "pca": "PCA (compression)"}

    print(f"[2/3] {len(configs)} configurations x 2 experiments")
    rows, sel_log = [], []
    for method, k in configs:
        t = time.time()
        # ---- Experiment 1
        s1 = make_search(method, k, C.CV_FOLDS).fit(X1, y1, groups=g1)
        p1 = s1.predict_proba(Xt)[:, 1]
        m1 = compute_metrics(yt, p1)
        kept1 = n_kept(s1, X1.shape[1])
        if (m := support(s1)) is not None:
            sel_log.append({"method": method, "k": str(k), "where": "exp1", "mask": m})
        # ---- Experiment 2 (outer 5-fold, inner 3-fold)
        oof, kept2 = np.zeros(len(y2)), []
        outer = StratifiedGroupKFold(n_splits=C.DEVICE_MATCHED_FOLDS, shuffle=True, random_state=SEED)
        for f, (tr, te) in enumerate(outer.split(X2, y2, g2)):
            assert not set(g2[tr]) & set(g2[te])
            s2 = make_search(method, k, 3).fit(X2[tr], y2[tr], groups=g2[tr])
            oof[te] = s2.predict_proba(X2[te])[:, 1]
            kept2.append(n_kept(s2, X2.shape[1]))
            if (m := support(s2)) is not None:
                sel_log.append({"method": method, "k": str(k), "where": f"exp2_fold{f + 1}", "mask": m})
        m2 = compute_metrics(y2, oof)
        rows.append({"method": label[method], "k": "all" if method == "none" else ("L1" if k is None else str(k)),
                     "kept_exp1": kept1, "kept_exp2_mean": round(float(np.mean(kept2)), 1),
                     **{f"E1_{c}": m1[c] for c in COLS}, **{f"E2_{c}": m2[c] for c in COLS}})
        print(f"  {label[method]:<24} k={str(k):<6} kept={kept1:>3}/{np.mean(kept2):5.1f}  "
              f"Exp1 ICBHI={m1['icbhi_score']:.3f} | Exp2 acc={m2['accuracy']:.3f} sens={m2['recall']:.3f} "
              f"ICBHI={m2['icbhi_score']:.3f}  ({time.time() - t:.0f}s)")
    res = pd.DataFrame(rows)
    res.to_csv(C.OUT_DIR / "results_selection.csv", index=False)

    # ---- which features were selected? (frequency over the 5 Exp. 2 folds, for fixed k = 40 and L1)
    print("[3/3] Selected features + figures")
    freq_rows, grp_rows = [], []
    for method, k in [("anova", "40"), ("mi", "40"), ("rf", "40"), ("l1", "None")]:
        masks = [e["mask"] for e in sel_log if e["method"] == method and e["k"] == k and e["where"].startswith("exp2")]
        if not masks:
            continue
        cnt = np.sum(masks, axis=0)
        for i in np.argsort(-cnt):
            if cnt[i] == 0:
                break
            freq_rows.append({"method": label[method], "feature": names[i], "group": groups[i],
                              "folds_selected": int(cnt[i])})
        chosen = np.concatenate([np.where(m)[0] for m in masks])
        share = pd.Series(groups[chosen]).value_counts(normalize=True)
        grp_rows.append({"method": label[method] + ("" if method == "l1" else f" (k={k})"),
                         **{gname: round(float(share.get(gname, 0.0)), 3) for gname in GROUP_COLORS}})
    freq = pd.DataFrame(freq_rows)
    freq.to_csv(C.OUT_DIR / "selected_features.csv", index=False)
    grp = pd.DataFrame(grp_rows)
    grp.to_csv(C.OUT_DIR / "selection_groups.csv", index=False)
    base_share = pd.Series(groups).value_counts(normalize=True)
    grp_all = pd.concat([pd.DataFrame([{"method": "All 354 features",
                                        **{g: round(float(base_share.get(g, 0)), 3) for g in GROUP_COLORS}}]), grp])

    # figure 1: Exp. 2 ICBHI score vs number of kept features
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ref = res[res.method == "No selection"]["E2_icbhi_score"].iloc[0]
    for method, col in [("ANOVA F", "#2a78d6"), ("Mutual information", "#1baf7a"),
                        ("Random-forest rank", "#eb6834"), ("PCA (compression)", "#8a919a")]:
        r = res[(res.method == method) & (res.k != "tuned")]
        xs = [int(k) for k in r.k] + [354]
        ys = list(r["E2_icbhi_score"]) + [ref]
        ax.plot(xs, ys, marker="o", lw=2, ms=6, color=col, label=method,
                ls="--" if method.startswith("PCA") else "-")
    l1 = res[(res.method == "L1 logistic regression") & (res.k == "L1")]
    if len(l1):
        ax.scatter(l1["kept_exp2_mean"], l1["E2_icbhi_score"], marker="D", s=50, color="#6250d6", zorder=5,
                   label="L1 logistic regression")
    ax.axhline(0.713, color=INK2, lw=1, ls=":")
    ax.text(21, 0.713 + 0.004, "MFCC only (HW1)", fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks([20, 40, 80, 160, 354], ["20", "40", "80", "160", "354\n(all)"])
    ax.set_xlabel("number of features kept (log scale)")
    ax.set_ylabel("Exp. 2 ICBHI score")
    ax.set_title("Feature selection inside CV · Exp. 2 (device-controlled)")
    ax.grid(True, color="#ebe5d8", lw=0.6)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig_selection_curve.png")
    plt.close(fig)

    # figure 2: which feature groups are selected
    fig, ax = plt.subplots(figsize=(6.6, 0.55 * len(grp_all) + 1.4))
    left = np.zeros(len(grp_all))
    for gname, col in GROUP_COLORS.items():
        vals = grp_all[gname].values
        ax.barh(range(len(grp_all)), vals, left=left, color=col, edgecolor="white", linewidth=1.5, label=gname)
        for i, (v, l) in enumerate(zip(vals, left)):
            if v >= 0.07:
                ax.text(l + v / 2, i, f"{v:.0%}", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        left += vals
    ax.set_yticks(range(len(grp_all)), grp_all["method"], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Share of each feature group among the selected features (Exp. 2 folds)")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig_selection_groups.png")
    plt.close(fig)

    short = res[["method", "k", "kept_exp1", "kept_exp2_mean", "E1_icbhi_score", "E1_auc",
                 "E2_accuracy", "E2_recall", "E2_specificity", "E2_macro_f1", "E2_auc", "E2_icbhi_score"]]
    top = freq[freq.method == "ANOVA F"].head(15) if len(freq) else freq
    warn = "> **WARNING: synthetic smoke-test data - do not report these numbers.**\n\n" if synthetic else ""
    md = ["# Feature selection study\n", warn,
          f"Start: all {len(names)} features ({BASE}). Selector is inside the Pipeline (fitted on training folds only), "
          "then calibrated RBF-SVM. 'tuned' = k chosen by inner patient-grouped CV. kept_exp2_mean = features kept, "
          "mean over the 5 outer folds. E2 = Meditron + Littmann C2SE, 5-fold patient-grouped CV (main criterion).\n",
          to_markdown(short),
          "\n## Share of feature groups among selected features (Exp. 2, 5 folds)\n", to_markdown(grp_all),
          "\n## Most often selected features - ANOVA F, k = 40 (out of 5 folds)\n",
          to_markdown(top) if len(top) else "-",
          f"\nRuntime: {time.time() - t0:.0f} s"]
    (C.OUT_DIR / "results_selection.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
