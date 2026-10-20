# CSE470 HW1 – COPD detection from lung sounds

A small, reproducible experiment **inspired by** (not a reproduction of)

> A. Srivastava, S. Jain, R. Miranda, S. Patil, S. Pandya, K. Kotecha,
> "Deep learning based respiratory sound analysis for detection of chronic obstructive pulmonary disease",
> *PeerJ Computer Science* 7:e369, 2021. DOI: 10.7717/peerj-cs.369 · code: https://github.com/ArpanSriv/copd-research

**Task:** binary classification of ICBHI 2017 recordings into **COPD** vs **non-COPD**
(the patient's *diagnosis* label, attached to every recording of that patient – a disease
label, not a sound-event label such as crackle/wheeze).

**Pipeline:** resample to 4 kHz → 100–1800 Hz band-pass → peak normalisation → 20 s fixed length →
40 MFCC + Δ summarised by mean/std (160 values) → SVM (RBF) [+ Random Forest, MLP for comparison] →
evaluation on unseen patients. A Streamlit app demonstrates the whole chain on single recordings.

---

## 1. Quick start (Windows + Anaconda)

1. **Get the data** (free Kaggle account needed):
   https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database → *Download*.
   Unzip it **into the `data/` folder** of this project. Any sub-folder layout works; the code searches
   `data/` recursively for the `.wav` files and `patient_diagnosis.csv`.
   (The official ICBHI zip from https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge also works.)
2. Open **Anaconda Prompt**, go to this folder and run **once**:
   ```
   cd %USERPROFILE%\Documents\CSE470_HW1_COPD
   setup_and_train.bat
   ```
   (= `pip install -r requirements.txt`, `python inspect_data.py`, `python train.py`; takes a few minutes.)
3. Start the demo:
   ```
   run_demo.bat
   ```
   (= `python -m streamlit run app.py`; the browser opens at http://localhost:8501.)

Optional smoke test without the real data (results are meaningless, the app shows a red banner):
```
python tools/make_synthetic_icbhi.py
python inspect_data.py --data-dir synthetic_data
python train.py --data-dir synthetic_data
```
Re-run `python inspect_data.py` and `python train.py` on the real data afterwards – they overwrite all outputs.

## 2. Files

| Path | Content |
|---|---|
| `hw1/config.py` | every parameter (sampling rate, filter, window sizes, augmentation, CV folds) |
| `hw1/data.py` | dataset index, label mapping, patient-disjoint split, device-name correction |
| `hw1/preprocess.py` | resampling, band-pass, normalisation, trim/pad |
| `hw1/features.py` | MFCC, log-mel, chroma, spectral descriptors, wavelet (DWT) features + time summaries |
| `hw1/augment.py` | loudness / mask / shift / speed augmentation (training data only) |
| `hw1/models.py` | SVM, RF, MLP pipelines with patient-grouped hyper-parameter search |
| `hw1/evaluate.py` | metrics, confusion matrix, ROC plots |
| `inspect_data.py` | homework steps 3–5: dataset tables + figures |
| `train.py` | homework steps 6–8: experiments, metrics, saved models |
| `app.py` | Streamlit demo |
| `feature_study.py` | extension 1: spectral + wavelet features and their combinations vs. MFCC (both experiments) |
| `selection_study.py` | extension 2: feature selection (ANOVA, MI, RF, L1, PCA) inside the pipeline |
| `consistency_study.py` | extension 3: same 5-fold patient-grouped CV for both experiments + device checks |
| `web/`, `tools/build_web_demo.py` | offline browser demo (`web/COPD_demo.html`, no Python needed) and the script that builds it |
| `metadata/` | official ICBHI split, diagnoses, device-name corrections (see §6) |
| `outputs/` | `dataset_summary.md`, `results.md`, `results_features.md`, `results_selection.md`, `results_consistency.md`, CSV tables, `figures/` (created by the scripts) |
| `models/` | trained models + `model_card.json` (created by `train.py`) |

## 3. What the code does

**Labels.** `COPD` → 1; Healthy, URTI, LRTI, Pneumonia, Bronchiectasis, Bronchiolitis, Asthma → 0 (`non-COPD`).
Result: 793 COPD recordings (64 patients) vs 127 non-COPD recordings (62 patients).

**Split (step 7).** The official ICBHI 60/40 train/test split, made **patient-disjoint**: patients 156 and 218
have recordings in both official partitions, so all their recordings are moved to train.
`train.py` asserts that no patient and no recording appears in both partitions.
Final: train 551 recordings / 79 patients, test 369 recordings / 47 patients (32 non-COPD test recordings).

**Preprocessing (step 4).**
| Step | Value | Why |
|---|---|---|
| Resample | 4 kHz | lowest native ICBHI rate → every device gets the same 0–2 kHz bandwidth (removes a sampling-rate shortcut) |
| Band-pass | Butterworth, order 4, 100–1800 Hz, zero-phase | removes heart sounds / DC / handling noise; lung sounds of interest lie roughly 100–2000 Hz |
| Normalise | remove DC, peak = 1 | devices have very different gains |
| Length | first 20 s; shorter files wrap-padded | same length as the paper; wrap padding avoids silent tails |

**Features (step 5).** 64-band log-mel spectrogram (512-sample = 128 ms window, 128-sample = 32 ms hop,
50–2000 Hz) → 40 MFCCs; MFCCs and their deltas are summarised by mean and standard deviation over time
(160 values). Log-mel (128 values) and chroma (24 values) vectors are computed for the feature comparison.

**Augmentation.** Each non-COPD *training* recording gets four extra copies (loudness ×0.6–1.4, 10 % time mask,
circular shift ≤5 s, speed ×0.8–1.25). Test data is never augmented (asserted in code).

**Classifiers (step 6).** scikit-learn pipelines `StandardScaler → classifier`, so scaling is fitted on
training data only. Hyper-parameters are chosen by grid search with 4-fold **StratifiedGroupKFold** (groups =
patients) inside the training set; the test set is used exactly once. Label = `P(COPD) ≥ 0.5`.

**Experiments (step 8).**
* **Experiment 1:** SVM / RF / MLP on MFCC; SVM on log-mel and chroma; SVM without augmentation;
  and a **device-only baseline** that predicts from the recording-device name alone.
* **Experiment 2:** only Meditron + Littmann Classic II SE recordings (the only devices used for both classes),
  5-fold patient-grouped cross-validation with nested tuning – removes the device shortcut.

Metrics: accuracy, precision, recall (= sensitivity), specificity, F1, macro-F1, ROC-AUC, ICBHI score,
confusion matrices (`outputs/results.md`, `outputs/figures/`).

**Extended experiments (after the presentation).** Run after `train.py`; each takes a few minutes and writes its
own `outputs/results_*.md`:
```
python feature_study.py        # spectral (24) + wavelet (42) features and combinations
python selection_study.py      # feature selection on all 354 features
python consistency_study.py    # same 5-fold CV for Exp. 1 and Exp. 2, device checks
```
* *Spectral:* mean/std of centroid, bandwidth, roll-off, flatness, flux, ZCR, RMS and 5 contrast bands.
* *Wavelet:* db4 5-level DWT (PyWavelets) → 6 sub-bands (0–62 Hz … 1–2 kHz) × 7 statistics.
* *Selection:* the selector is a pipeline step (`scaler → selector → SVM`), so it is fitted on training folds only;
  the number of kept features is chosen by inner CV.
* Main result (Exp. 2): spectral + wavelet ICBHI 0.78 vs MFCC 0.71; feature selection gave no clear extra gain.

## 4. Differences from the paper

| | Paper / published notebook | This project |
|---|---|---|
| Sampling rate | 22 050 Hz | 4 000 Hz |
| Filtering | none | 100–1800 Hz band-pass |
| Long files | recordings ≥ 60 s dropped | kept, trimmed to 20 s |
| Padding | zeros | wrap |
| Features | 40×862 maps (MFCC, mel, 3 chroma variants) | 160-value MFCC summary (+ log-mel, chroma) |
| Classifier | 4-block 2-D CNN, Adam, 250 epochs | SVM (RBF); RF and MLP for comparison |
| Augmentation | speed 0.5–2×; in the notebook the "augmented" rows reuse the original features | speed 0.8–1.25×; features recomputed from each augmented signal; training set only |
| Split | official split (2 patients in both partitions); test set also used as validation data | patient-disjoint split; tuning by CV inside the training set |
| AUC | computed from hard labels | computed from probabilities |

## 5. Own contributions vs. adapted code

No code was copied from the paper's notebook. The pipeline was re-implemented from the paper's description
(feature choice n = 40, 20 s length, the four augmentation types and their loudness range). Own contributions:
patient-disjoint split, band-pass/resampling stage, train-only augmentation with recomputed features,
patient-grouped tuning, device-confound analysis (device-only baseline, Experiment 2), the Streamlit demo.
`metadata/official_split.txt`, `patient_diagnosis.txt` and `filename_differences.txt` are ICBHI metadata files,
taken from the repository https://github.com/raymin0223/patch-mix_contrastive_learning (data/icbhi_dataset).

## 6. Notes
* 91 files in the original release carry a wrong device name (e.g. `…_AKGC417L` instead of `…_Meditron`).
  Files are matched on the first four name fields; the corrected device comes from the official split file.
* This is an educational experiment, **not** a diagnostic tool.

## 7. AI usage
This project was developed with help from an AI assistant (Claude, Anthropic). We used it to:

1. read and analyse the selected paper and its published notebook (Phase 1),
2. propose the classification task and the experiment design (patient-disjoint split, device-only baseline),
3. write most of the Python code, the Streamlit demo and this README,
4. draft the text of the conference-style paper,
5. design and code the extended experiments (new features, feature selection, same CV) and the offline web demo.

The authors (Beyza Nur Güç, Ahmed Hamza Kerman, Hatice Kübra Aydın) ran all experiments on the real ICBHI data,
checked the reported numbers against the program output (`outputs/results*.md`), and are responsible for the final
code and text.

## Report
* `report/CSE470_HW1_Report_v2.pdf` – current version: the conference-style paper with Section 4 "Extended Experiments".
* `report/CSE470_HW1_Report.pdf` – the first (submitted) 2-page version.
* `report/source/` – HTML source and figures of the paper.

## 8. References
1. Srivastava et al., PeerJ Comput. Sci. 7:e369, 2021. https://doi.org/10.7717/peerj-cs.369
2. B. M. Rocha et al., "An open access database for the evaluation of respiratory sound classification
   algorithms", *Physiological Measurement* 40(3):035001, 2019 (ICBHI 2017 database).
3. Kaggle mirror: https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database
4. G. R. Lee et al., "PyWavelets: A Python package for wavelet analysis", *J. Open Source Softw.* 4(36):1237, 2019.
4. McFee et al., librosa; Pedregosa et al., scikit-learn, JMLR 12, 2011.
