# Feature study: spectrogram + wavelet features


SVM (RBF), same protocol as train.py. E1 = Experiment 1 (official split, test set, 369 recordings). E2 = Experiment 2 (214 Meditron + Littmann C2SE recordings, 5-fold patient-grouped CV, pooled out-of-fold). recall = sensitivity.

| features | n_features | E1_accuracy | E1_icbhi_score | E1_auc | E2_accuracy | E2_acc_fold_std | E2_recall | E2_specificity | E2_macro_f1 | E2_auc | E2_icbhi_score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mfcc | 160 | 0.927 | 0.946 | 0.989 | 0.724 | 0.105 | 0.655 | 0.772 | 0.714 | 0.816 | 0.713 |
| spectral | 24 | 0.940 | 0.925 | 0.983 | 0.776 | 0.138 | 0.736 | 0.803 | 0.768 | 0.844 | 0.769 |
| wavelet | 42 | 0.835 | 0.867 | 0.941 | 0.696 | 0.017 | 0.483 | 0.843 | 0.665 | 0.721 | 0.663 |
| mfcc+spectral | 184 | 0.913 | 0.924 | 0.988 | 0.771 | 0.073 | 0.759 | 0.779 | 0.765 | 0.844 | 0.769 |
| mfcc+wavelet | 202 | 0.916 | 0.954 | 0.994 | 0.720 | 0.108 | 0.667 | 0.756 | 0.711 | 0.802 | 0.711 |
| spectral+wavelet | 66 | 0.946 | 0.956 | 0.988 | 0.794 | 0.104 | 0.701 | 0.858 | 0.783 | 0.852 | 0.780 |
| mfcc+spectral+wavelet | 226 | 0.943 | 0.955 | 0.994 | 0.706 | 0.090 | 0.621 | 0.764 | 0.693 | 0.809 | 0.692 |
| mfcc+melspec+spectral+wavelet | 354 | 0.957 | 0.962 | 0.995 | 0.752 | 0.108 | 0.655 | 0.819 | 0.740 | 0.842 | 0.737 |

## Wavelet sub-band relative energy (mean, Experiment 2 subset)

| band | COPD | non-COPD |
|---|---|---|
| A5 0-62 Hz | 0.043 | 0.061 |
| D5 62-125 Hz | 0.307 | 0.278 |
| D4 125-250 Hz | 0.474 | 0.409 |
| D3 250-500 Hz | 0.120 | 0.154 |
| D2 500-1000 Hz | 0.040 | 0.068 |
| D1 1000-2000 Hz | 0.016 | 0.030 |

Runtime: 124 s