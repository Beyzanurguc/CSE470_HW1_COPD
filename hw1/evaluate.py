"""Metrics and evaluation figures (homework steps 7-8)."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)

from . import config as C

# Validated categorical slots (blue / orange) + ink colours, used by every figure.
COLORS = {C.NEGATIVE_CLASS: "#2a78d6", C.POSITIVE_CLASS: "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 11, "font.size": 9.5, "grid.color": GRID,
})


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Positive class = COPD (1). Sensitivity = recall of COPD, specificity = recall of non-COPD."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "specificity": spec,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "icbhi_score": (sens + spec) / 2,
        "auc": roc_auc_score(y_true, y_prob) if len(set(y_true)) == 2 else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "n": int(len(y_true)),
    }
    return {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in out.items()}


def plot_confusion(y_true, y_prob, title: str, path, threshold: float = 0.5):
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pct = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.0%})", ha="center", va="center",
                    color="white" if pct[i, j] > 0.55 else INK, fontsize=10)
    ax.set_xticks([0, 1], C.CLASS_NAMES)
    ax.set_yticks([0, 1], C.CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(path)
    plt.close(fig)


def plot_roc(curves: dict, title: str, path):
    """curves: {label: (y_true, y_prob)}"""
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=INK2)
    for (label, (yt, yp)), col in zip(curves.items(), palette):
        fpr, tpr, _ = roc_curve(yt, yp)
        ax.plot(fpr, tpr, lw=2, color=col, label=f"{label} (AUC {roc_auc_score(yt, yp):.2f})")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title(title)
    ax.grid(True, lw=0.6)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.savefig(path)
    plt.close(fig)
