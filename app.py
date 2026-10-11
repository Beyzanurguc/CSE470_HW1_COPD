"""Interactive demo for the class presentation.

    streamlit run app.py

Requires the models produced by `python train.py` (and figures from `python inspect_data.py`).
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import welch

from hw1 import config as C
from hw1.evaluate import COLORS, INK2, compute_metrics
from hw1.features import extract, log_mel, mfcc, standardize_rows
from hw1.preprocess import bandpass, fix_duration, normalize, resample

st.set_page_config(page_title="COPD Lung Sound Demo", page_icon="🫁", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
.result-card {border-radius: 14px; padding: 18px 22px; color: #fff; margin-bottom: 0.6rem;}
.result-card .lbl {font-size: 0.85rem; opacity: 0.9; text-transform: uppercase; letter-spacing: .06em;}
.result-card .val {font-size: 2.1rem; font-weight: 700; line-height: 1.15;}
.result-card .sub {font-size: 0.95rem; opacity: 0.95;}
.pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.85rem; font-weight:600;}
.ok {background:#e3f4ec; color:#0b5d3b;} .bad {background:#fde8e4; color:#8a2a12;}
.small-muted {color:#52514e; font-size:0.85rem;}
</style>
""", unsafe_allow_html=True)

DEVICE_NAMES = {"AKGC417L": "AKG C417L microphone", "LittC2SE": "Littmann Classic II SE",
                "Litt3200": "Littmann 3200 electronic", "Meditron": "WelchAllyn Meditron"}
LOC_NAMES = {"Tc": "trachea", "Al": "anterior left", "Ar": "anterior right", "Pl": "posterior left",
             "Pr": "posterior right", "Ll": "lateral left", "Lr": "lateral right"}


# ------------------------------------------------------------------ loading
@st.cache_resource
def load_card_and_models():
    card_path = C.MODEL_DIR / "model_card.json"
    if not card_path.exists():
        return None, {}
    card = json.loads(card_path.read_text())
    models = {k: joblib.load(C.MODEL_DIR / v["file"]) for k, v in card["models"].items()
              if (C.MODEL_DIR / v["file"]).exists()}
    return card, models


@st.cache_data
def load_csv(name):
    p = C.OUT_DIR / name
    return pd.read_csv(p) if p.exists() else None


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else C.ROOT / p


@st.cache_data
def corrected_devices() -> dict:
    """Device names from the official split (fixes the 91 mislabelled Kaggle file names)."""
    from hw1.data import load_official_split
    return {f.rsplit("_", 1)[0]: f.rsplit("_", 1)[1] for f in load_official_split()["file"]}


def parse_icbhi(name: str):
    parts = Path(name).stem.split("_")
    if len(parts) == 5 and parts[0].isdigit():
        parts[4] = corrected_devices().get("_".join(parts[:4]), parts[4])
        return {"patient": parts[0], "chest location": LOC_NAMES.get(parts[2], parts[2]),
                "mode": "multichannel" if parts[3] == "mc" else "single channel",
                "device": parts[4]}
    return None


def fig_show(fig):
    st.pyplot(fig, width="stretch")
    plt.close(fig)


card, MODELS = load_card_and_models()

# ------------------------------------------------------------------ header
st.title("🫁 COPD detection from lung sounds")
st.markdown('<span class="small-muted">CSE470 Biomedical Signal Processing · HW1 demo · simplified experiment '
            'inspired by Srivastava et al., <i>PeerJ Computer Science</i> 7:e369 (2021) · ICBHI 2017 dataset</span>',
            unsafe_allow_html=True)

if card is None or not MODELS:
    st.warning("No trained model found. Run these commands in the project folder first:\n\n"
               "```\npython inspect_data.py\npython train.py\n```\nthen restart: `streamlit run app.py`")
    st.stop()
if card.get("synthetic_data"):
    st.error("⚠️ These models were trained on SYNTHETIC smoke-test audio. Predictions and metrics are meaningless. "
             "Download the real ICBHI data and re-run `python train.py`.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Model")
    keys = list(MODELS)
    default = keys.index(card.get("main_model", keys[0])) if card.get("main_model") in keys else 0
    model_key = st.radio("Classifier (MFCC features)", keys, index=default,
                         format_func=lambda k: card["models"][k]["label"])
    threshold = st.slider("Decision threshold  P(COPD) ≥", 0.05, 0.95, 0.50, 0.05,
                          help="Lower → more sensitive (fewer missed COPD), higher → more specific.")
    tm = card["models"][model_key]["test_metrics"]
    st.caption("Held-out test set, threshold 0.5")
    c1, c2 = st.columns(2)
    c1.metric("Sensitivity", f"{tm['recall']:.2f}")
    c2.metric("Specificity", f"{tm['specificity']:.2f}")
    c1.metric("F1 (COPD)", f"{tm['f1']:.2f}")
    c2.metric("AUC", f"{tm['auc']:.2f}")
    st.divider()
    st.caption(f"Trained on: {card['trained_on']}. Test: {card['test_set']}.")
    st.caption("Educational demo — not a medical device.")

model = MODELS[model_key]
tab_pred, tab_res, tab_data, tab_about = st.tabs(["🔍 Predict", "📊 Results", "🗂️ Dataset", "ℹ️ About"])

# ================================================================== PREDICT
with tab_pred:
    preds = load_csv("test_predictions.csv")
    left, right = st.columns([1, 2.2], gap="large")

    with left:
        st.subheader("1 · Choose a recording")
        source = st.radio("Source", ["Test-set recording (unseen patients)", "Upload a WAV file"],
                          label_visibility="collapsed")
        y_raw = sr0 = None
        truth = meta = None
        if source.startswith("Test") and preds is not None:
            f1, f2 = st.columns(2)
            cls = f1.selectbox("True class", ["All"] + C.CLASS_NAMES)
            dev = f2.selectbox("Device", ["All"] + sorted(preds["device"].unique()))
            pool = preds
            if cls != "All":
                pool = pool[pool.label_name == cls]
            if dev != "All":
                pool = pool[pool.device == dev]
            if len(pool) == 0:
                st.info("No test recordings match these filters.")
            else:
                if "pick" not in st.session_state or st.session_state.pick not in set(pool.file):
                    st.session_state.pick = pool.file.iloc[0]
                if st.button("🎲 Random recording", width="stretch"):
                    st.session_state.pick = pool.sample(1).file.iloc[0]
                files = pool.file.tolist()
                pick = st.selectbox(f"Recording ({len(pool)} available)", files,
                                    index=files.index(st.session_state.pick),
                                    format_func=lambda f: f"{f}  ·  {pool.set_index('file').loc[f, 'diagnosis']}")
                st.session_state.pick = pick
                row = pool.set_index("file").loc[pick]
                path = resolve(row["path"])
                if not path.exists():
                    st.error(f"Audio file not found: {path}. Was the data folder moved?")
                else:
                    y_raw, sr0 = librosa.load(path, sr=None, mono=True)
                    truth = (int(row["label"]), row["diagnosis"])
                    meta = parse_icbhi(pick)
        elif source.startswith("Upload"):
            up = st.file_uploader("WAV / FLAC / OGG lung-sound recording", type=["wav", "flac", "ogg"])
            if up is not None:
                y_raw, sr0 = librosa.load(io.BytesIO(up.getvalue()), sr=None, mono=True)
                meta = parse_icbhi(up.name)
                st.caption("No ground truth for uploaded files.")
        else:
            st.info("Run `python train.py` to create the test-set list.")

        if y_raw is not None:
            st.markdown("**Original audio**")
            st.audio(y_raw, sample_rate=int(sr0))

    with right:
        if y_raw is None:
            st.subheader("2 · Result")
            st.info("Pick or upload a recording on the left.")
        else:
            y4 = resample(y_raw.astype(np.float32), int(sr0))
            yb = bandpass(y4)
            yp = fix_duration(normalize(yb))
            p = float(model.predict_proba(extract(yp)["mfcc"][None, :])[0, list(model.classes_).index(1)])
            pred = int(p >= threshold)
            name = C.CLASS_NAMES[pred]

            st.subheader("2 · Result")
            r1, r2 = st.columns([1.25, 1])
            with r1:
                st.markdown(f"""<div class="result-card" style="background:{COLORS[name]}">
                    <div class="lbl">Prediction · {card['models'][model_key]['label']}</div>
                    <div class="val">{name}</div>
                    <div class="sub">P(COPD) = {p:.2f} &nbsp;·&nbsp; threshold {threshold:.2f}</div></div>""",
                            unsafe_allow_html=True)
                st.progress(min(max(p, 0.0), 1.0), text=f"P(COPD) {p:.0%}")
            with r2:
                if truth is not None:
                    ok = truth[0] == pred
                    st.markdown(f"**True label:** {C.CLASS_NAMES[truth[0]]} ({truth[1]})<br>"
                                f'<span class="pill {"ok" if ok else "bad"}">'
                                f'{"✓ correct" if ok else "✗ misclassified"}</span>', unsafe_allow_html=True)
                if meta:
                    st.markdown(f"**Patient** {meta['patient']} · **{meta['chest location']}** · {meta['mode']}<br>"
                                f"**Device** {DEVICE_NAMES.get(meta['device'], meta['device'])}",
                                unsafe_allow_html=True)
                st.markdown(f"<span class='small-muted'>Native rate {int(sr0)} Hz · "
                            f"length {len(y_raw) / sr0:.1f} s</span>", unsafe_allow_html=True)
            if meta and meta["device"] in ("AKGC417L", "Litt3200"):
                st.warning("In ICBHI this device was used **only for COPD patients**. A high P(COPD) here may partly "
                           "reflect device/recording characteristics rather than lung pathology (see Results → device baseline).")

            st.subheader("3 · What the model sees")
            t1, t2, t3, t4 = st.tabs(["Waveform", "Band-pass filter", "Log-mel spectrogram", "MFCC"])
            col = COLORS[name]
            with t1:
                fig, ax = plt.subplots(2, 1, figsize=(9, 3.6), sharex=False)
                ax[0].plot(np.arange(len(y_raw)) / sr0, y_raw, lw=0.4, color="#8f8e88")
                ax[0].set_title(f"Raw ({int(sr0)} Hz)", fontsize=9.5)
                ax[1].plot(np.arange(len(yp)) / C.TARGET_SR, yp, lw=0.4, color=col)
                ax[1].set_title(f"Preprocessed: {C.TARGET_SR} Hz · {C.BANDPASS_HZ[0]:.0f}–{C.BANDPASS_HZ[1]:.0f} Hz "
                                f"band-pass · peak-normalised · {C.DURATION_S:.0f} s", fontsize=9.5)
                ax[1].set_xlabel("time (s)")
                fig.tight_layout()
                fig_show(fig)
                st.markdown("**Preprocessed audio** (what the classifier hears)")
                st.audio(yp, sample_rate=C.TARGET_SR)
            with t2:
                fig, ax = plt.subplots(figsize=(9, 3))
                for sig, lab, c in [(y4, "before filter", "#c3c2b7"), (yb, "after band-pass", col)]:
                    f, pxx = welch(sig, fs=C.TARGET_SR, nperseg=1024)
                    ax.semilogy(f, pxx + 1e-12, lw=1.5, color=c, label=lab)
                for e in C.BANDPASS_HZ:
                    ax.axvline(e, color=INK2, ls="--", lw=1)
                ax.set_xlabel("frequency (Hz)")
                ax.set_ylabel("PSD")
                ax.legend(frameon=False)
                fig.tight_layout()
                fig_show(fig)
                st.caption("Below 100 Hz: heart sounds, muscle and handling noise. Upper edge stays below the 2 kHz "
                           "Nyquist limit of the common 4 kHz rate.")
            with t3:
                fig, ax = plt.subplots(figsize=(9, 3.2))
                im = librosa.display.specshow(log_mel(yp), sr=C.TARGET_SR, hop_length=C.HOP_LENGTH, x_axis="time",
                                              y_axis="mel", fmin=C.FMIN, fmax=C.FMAX, ax=ax, cmap="magma")
                fig.colorbar(im, ax=ax, format="%+.0f dB")
                fig.tight_layout()
                fig_show(fig)
                st.caption(f"{C.N_MELS} mel bands, {C.N_FFT / C.TARGET_SR * 1000:.0f} ms window, "
                           f"{C.HOP_LENGTH / C.TARGET_SR * 1000:.0f} ms hop. Breathing cycles appear as vertical energy bursts; "
                           "wheezes as horizontal lines; crackles as short vertical streaks.")
            with t4:
                M = standardize_rows(mfcc(yp))
                fig, ax = plt.subplots(figsize=(9, 3.2))
                im = librosa.display.specshow(M, sr=C.TARGET_SR, hop_length=C.HOP_LENGTH, x_axis="time",
                                              ax=ax, cmap="coolwarm", vmin=-3, vmax=3)
                ax.set_ylabel("MFCC index")
                fig.colorbar(im, ax=ax)
                fig.tight_layout()
                fig_show(fig)
                st.caption(f"Shown z-scored per coefficient for visibility. {C.N_MFCC} MFCCs (+ deltas) are summarised by their mean and standard deviation over time "
                           f"→ a {4 * C.N_MFCC}-value vector, the classifier's only input.")

            with st.expander("Pipeline parameters"):
                st.markdown(f"""
| Step | Setting |
|---|---|
| Resampling | {int(sr0)} Hz → {C.TARGET_SR} Hz |
| Band-pass | Butterworth order {C.FILTER_ORDER}, {C.BANDPASS_HZ[0]:.0f}–{C.BANDPASS_HZ[1]:.0f} Hz, zero-phase |
| Normalisation | DC removal + peak amplitude 1 |
| Length | first {C.DURATION_S:.0f} s (shorter recordings wrap-padded) |
| Features | {C.N_MFCC} MFCC + Δ, mean & std → {4 * C.N_MFCC} values |
| Classifier | {card['models'][model_key]['label']} · {card['models'][model_key]['params']} |
""")

# ================================================================== RESULTS
with tab_res:
    res = load_csv("results_main.csv")
    preds = load_csv("test_predictions.csv")
    st.subheader("Experiment 1 · official ICBHI split, patient-disjoint")
    if res is not None:
        show = (res.drop(columns=["run", "cv_bal_acc_train"], errors="ignore")
                .rename(columns={"recall": "sensitivity", "icbhi_score": "ICBHI score", "macro_f1": "macro F1",
                                 "f1": "F1", "auc": "AUC", "augmentation": "aug."}))
        show["model"] = show["model"].str.replace("Device-only baseline (no audio)", "Device only (no audio)", regex=False)
        st.dataframe(show.style.format(precision=3, na_rep="–"), width="stretch", hide_index=True)
        dev_row = res[res.run == "device_only"]
        if len(dev_row):
            st.warning(f"**Shortcut check:** a rule that only looks at the *recording device* (no audio) reaches "
                       f"accuracy {dev_row.accuracy.iloc[0]:.2f} on this test set. In ICBHI the AKG C417L and "
                       "Littmann 3200 were used only for COPD patients, so high scores on this split do not prove "
                       "the model hears COPD. Experiment 2 below removes this shortcut.")

    if preds is not None:
        st.subheader("Threshold explorer")
        run_id = f"{model_key}_mfcc_aug"
        colname = f"p_copd_{run_id}"
        if colname in preds:
            m = compute_metrics(preds["label"], preds[colname], threshold)
            a, b, c, d, e = st.columns(5)
            a.metric("Accuracy", f"{m['accuracy']:.3f}")
            b.metric("Sensitivity", f"{m['recall']:.3f}")
            c.metric("Specificity", f"{m['specificity']:.3f}")
            d.metric("F1 (COPD)", f"{m['f1']:.3f}")
            e.metric("Macro F1", f"{m['macro_f1']:.3f}")
            cm = pd.DataFrame([[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
                              index=[f"true {n}" for n in C.CLASS_NAMES],
                              columns=[f"pred {n}" for n in C.CLASS_NAMES])
            st.dataframe(cm, width="content")
            st.caption(f"{card['models'][model_key]['label']} at threshold {threshold:.2f} (move the slider in the sidebar).")

    figs = [("cm_svm_mfcc_aug.png", "SVM + MFCC"), ("cm_svm_mfcc_noaug.png", "SVM, no augmentation"),
            ("cm_device_only.png", "Device-only baseline"), ("roc_main.png", "ROC curves")]
    cols = st.columns(4)
    for col, (fname, cap) in zip(cols, figs):
        if (C.FIG_DIR / fname).exists():
            col.image(str(C.FIG_DIR / fname), caption=cap, width="stretch")

    per_dev = load_csv("per_device_main.csv")
    if per_dev is not None:
        st.markdown("**Main model accuracy per device (test set)**")
        st.dataframe(per_dev, hide_index=True)

    res2 = load_csv("results_device_matched.csv")
    if res2 is not None:
        st.subheader(f"Experiment 2 · {' + '.join(C.DEVICE_MATCHED)} recordings only")
        st.caption(f"Both classes recorded with the same devices · {C.DEVICE_MATCHED_FOLDS}-fold patient-grouped "
                   "cross-validation · no augmentation. This is the more honest estimate of what the audio itself carries.")
        st.dataframe(res2.rename(columns={"recall": "sensitivity"}).style.format(precision=3),
                     width="stretch", hide_index=True)
        if (C.FIG_DIR / "cm_device_matched.png").exists():
            st.image(str(C.FIG_DIR / "cm_device_matched.png"), width=330)

# ================================================================== DATASET
with tab_data:
    summ = C.OUT_DIR / "dataset_summary.md"
    c1, c2 = st.columns([1, 1.25], gap="large")
    with c1:
        if summ.exists():
            text = summ.read_text(encoding="utf-8").split("\n", 1)[1]   # drop the big H1
            st.markdown(text.replace("\n## ", "\n#### "))
        else:
            st.info("Run `python inspect_data.py` to create the dataset summary.")
    with c2:
        for fname in ["fig_class_distribution.png", "fig_device_by_class.png", "fig_durations.png",
                      "fig_waveforms.png", "fig_features.png"]:
            if (C.FIG_DIR / fname).exists():
                st.image(str(C.FIG_DIR / fname), width="stretch")

# ================================================================== ABOUT
with tab_about:
    st.markdown(f"""
### Reference paper
Srivastava A., Jain S., Miranda R., Patil S., Pandya S., Kotecha K. *Deep learning based respiratory sound analysis
for detection of chronic obstructive pulmonary disease.* PeerJ Computer Science 7:e369, 2021.
DOI 10.7717/peerj-cs.369 · code: github.com/ArpanSriv/copd-research

### What this demo does
**Task:** {card['task']}. The label is the patient's *diagnosis*, attached to every recording of that patient —
this is disease prediction, **not** detection of sound events such as crackles or wheezes.

### Differences from the paper
| | Paper | This implementation |
|---|---|---|
| Sampling rate | 22 050 Hz (librosa default) | {C.TARGET_SR} Hz (lowest native ICBHI rate) |
| Filtering | none | {C.BANDPASS_HZ[0]:.0f}–{C.BANDPASS_HZ[1]:.0f} Hz band-pass |
| Length | 20 s, zero-pad | 20 s, wrap-pad |
| Features | 40×862 MFCC image | 40 MFCC + Δ, mean & std (160 values) |
| Classifier | 4-block 2-D CNN (Keras) | SVM (RBF); RF and MLP for comparison |
| Augmentation | loudness, mask, shift, speed 0.5–2× | same four, speed 0.8–1.25×, training set only |
| Split | official ICBHI split (2 patients in both parts) | official split made patient-disjoint |

### Limitations
* Device confound: AKG C417L and Littmann 3200 recordings are 100 % COPD.
* Small minority class: 127 non-COPD recordings; the test set has only 32 of them.
* Diagnosis label ≠ sound content: a COPD patient's recording can sound normal.
* Educational demo — **not** a diagnostic tool.
""")
