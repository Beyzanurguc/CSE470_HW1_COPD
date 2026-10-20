"""Build a single, offline HTML version of the demo: web/COPD_demo.html

    python tools/build_web_demo.py

Needs the trained models (python train.py) and the dataset in data/ (for the example recordings).
Everything (models, results, example audio) is embedded, so the HTML opens in any browser without Python.
"""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfilt_zi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hw1 import config as C  # noqa: E402

EXAMPLE_SR = 8000          # example audio is stored at 8 kHz (the model only uses 0-2 kHz)
N_EXAMPLES = {("AKGC417L", 1): 3, ("Litt3200", 1): 2, ("Meditron", 1): 3, ("Meditron", 0): 6}


def r(a, nd=7):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def export_scaler(pipe):
    sc = pipe.named_steps["scaler"]
    return {"mean": r(sc.mean_, 9), "scale": r(sc.scale_, 9)}


def export_svm(pipe):
    cal = pipe.named_steps["clf"].calibrated_classifiers_[0]
    svc, sig = cal.estimator, cal.calibrators[0]
    assert list(svc.classes_) == [0, 1]
    return {"kind": "svm", "scaler": export_scaler(pipe), "sv": r(svc.support_vectors_, 7),
            "dual": r(svc.dual_coef_[0], 9), "intercept": float(svc.intercept_[0]), "gamma": float(svc._gamma),
            "a": float(sig.a_), "b": float(sig.b_)}


def export_mlp(pipe):
    mlp = pipe.named_steps["clf"]
    assert mlp.activation == "relu" and mlp.out_activation_ == "logistic"
    return {"kind": "mlp", "scaler": export_scaler(pipe), "W": [r(w, 7) for w in mlp.coefs_],
            "b": [r(b, 7) for b in mlp.intercepts_]}


def export_rf(pipe):
    rf = pipe.named_steps["clf"]
    k = list(rf.classes_).index(1)
    trees = []
    for est in rf.estimators_:
        t = est.tree_
        v = t.value[:, 0, :]
        p = v[:, k] / v.sum(axis=1)
        trees.append({"l": t.children_left.tolist(), "r": t.children_right.tolist(), "f": t.feature.tolist(),
                      "t": [float(x) for x in t.threshold], "p": r(p, 6)})
    return {"kind": "rf", "scaler": export_scaler(pipe), "trees": trees}


def wav_b64(y, sr):
    buf = io.BytesIO()
    sf.write(buf, (np.clip(y / (np.max(np.abs(y)) + 1e-9) * 0.95, -1, 1) * 32767).astype(np.int16), sr,
             format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def main():
    card = json.loads((C.MODEL_DIR / "model_card.json").read_text())
    pipes = {k: joblib.load(C.MODEL_DIR / v["file"]) for k, v in card["models"].items()}
    exporters = {"svm": export_svm, "mlp": export_mlp, "rf": export_rf}
    models = {k: exporters[k](p) for k, p in pipes.items()}

    sos = butter(C.FILTER_ORDER, C.BANDPASS_HZ, btype="bandpass", fs=C.TARGET_SR, output="sos")
    mel = librosa.filters.mel(sr=C.TARGET_SR, n_fft=C.N_FFT, n_mels=C.N_MELS, fmin=C.FMIN, fmax=C.FMAX)
    cfg = {"target_sr": C.TARGET_SR, "duration_s": C.DURATION_S, "n_fft": C.N_FFT, "hop_length": C.HOP_LENGTH,
           "n_mfcc": C.N_MFCC, "fmin": C.FMIN, "fmax": C.FMAX, "bandpass": list(C.BANDPASS_HZ),
           "sos": sos.tolist(), "sos_zi": sosfilt_zi(sos).tolist(), "mel_basis": r(mel, 9)}

    preds = pd.read_csv(C.OUT_DIR / "test_predictions.csv")
    pcols = {k: f"p_copd_{k}_mfcc_aug" for k in models}
    test = {"label": preds["label"].astype(int).tolist(), "device": preds["device"].tolist(),
            **{k: r(preds[c], 4) for k, c in pcols.items()}}

    # ---- example recordings: a fixed, stratified pick from the held-out test patients
    rng = np.random.default_rng(7)
    examples = []
    for (dev, lab), n in N_EXAMPLES.items():
        pool = preds[(preds.device == dev) & (preds.label == lab)]
        if not len(pool):
            continue
        wrong = pool[(pool["p_copd_svm_mfcc_aug"] >= 0.5).astype(int) != lab]
        pick = pd.concat([wrong.head(1), pool.drop(wrong.head(1).index)
                          .iloc[rng.permutation(len(pool) - min(1, len(wrong)))[: n - min(1, len(wrong))]]])
        for _, row in pick.iterrows():
            path = Path(row["path"]) if Path(row["path"]).is_absolute() else C.ROOT / row["path"]
            y, sr0 = librosa.load(path, sr=None, mono=True)
            y8 = librosa.resample(y, orig_sr=sr0, target_sr=EXAMPLE_SR)[: int(EXAMPLE_SR * C.DURATION_S)]
            examples.append({"file": row["file"], "label": int(row["label"]), "diagnosis": row["diagnosis"],
                             "device": row["device"], "patient": int(row["patient_id"]), "native_sr": int(sr0),
                             "duration": round(len(y) / sr0, 1),
                             "p": {k: round(float(row[c]), 4) for k, c in pcols.items()},
                             "audio": wav_b64(y8, EXAMPLE_SR)})
        print(f"  examples {dev}/{lab}: {len(pick)}")

    def csv(name):
        p = C.OUT_DIR / name
        return pd.read_csv(p).replace({np.nan: None}).to_dict(orient="records") if p.exists() else []

    rec = pd.read_csv(C.OUT_DIR / "recordings.csv")
    data = {
        "card": card, "cfg": cfg, "models": models, "test": test, "examples": examples,
        "results_main": csv("results_main.csv"), "results_dm": csv("results_device_matched.csv"),
        "per_device": csv("per_device_main.csv"),
        "dataset": {
            "by_dx": rec.groupby("diagnosis").agg(recordings=("file", "size"), patients=("patient_id", "nunique"))
                        .sort_values("recordings", ascending=False).reset_index().to_dict(orient="records"),
            "split": rec.groupby(["label_name", "split"]).agg(recordings=("file", "size"),
                     patients=("patient_id", "nunique")).reset_index().to_dict(orient="records"),
            "device": pd.crosstab(rec["device"], rec["label_name"]).reset_index().to_dict(orient="records"),
            "sr": pd.crosstab(rec["device"], rec["native_sr"]).reset_index().rename(columns=str).to_dict(orient="records"),
            "n_rec": int(len(rec)), "n_pat": int(rec.patient_id.nunique()),
        },
    }
    web = ROOT / "web"
    html = (web / "template.html").read_text(encoding="utf-8")
    html = html.replace("/*__DSP__*/", (web / "dsp.js").read_text(encoding="utf-8"))
    html = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    out = web / "COPD_demo.html"
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(examples)} example recordings)")


if __name__ == "__main__":
    main()
