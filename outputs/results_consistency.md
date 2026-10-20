# Part 3: same CV for both experiments + device checks


All rows: 5-fold StratifiedGroupKFold by patient (seed 42), nested 3-fold tuning of the SVM, pooled out-of-fold metrics. Augmentation (if 'yes') only on non-COPD recordings of the training folds. recall = sensitivity.

## A) Both experiments with the same CV

| experiment | features | augmentation | recordings | accuracy | recall | specificity | macro_f1 | auc | icbhi_score | acc_fold_std |
|---|---|---|---|---|---|---|---|---|---|---|
| Exp. 1 · all devices | mfcc | no | 920 | 0.935 | 0.974 | 0.693 | 0.854 | 0.955 | 0.833 | 0.022 |
| Exp. 1 · all devices | mfcc | yes | 920 | 0.903 | 0.915 | 0.827 | 0.822 | 0.947 | 0.871 | 0.013 |
| Exp. 1 · all devices | spectral+wavelet | no | 920 | 0.912 | 0.966 | 0.575 | 0.796 | 0.948 | 0.770 | 0.028 |
| Exp. 1 · all devices | spectral+wavelet | yes | 920 | 0.889 | 0.910 | 0.756 | 0.793 | 0.938 | 0.833 | 0.047 |
| Exp. 1 · all devices | mfcc+melspec+spectral+wavelet | no | 920 | 0.929 | 0.966 | 0.701 | 0.846 | 0.957 | 0.833 | 0.032 |
| Exp. 1 · all devices | mfcc+melspec+spectral+wavelet | yes | 920 | 0.938 | 0.945 | 0.898 | 0.882 | 0.969 | 0.921 | 0.027 |
| Exp. 1 · all devices | device only (no audio) | no | 920 | 0.922 | 0.955 | 0.717 | 0.836 | 0.954 | 0.836 | 0.033 |
| Exp. 2 · Meditron + Littmann C2SE | mfcc | no | 214 | 0.724 | 0.655 | 0.772 | 0.714 | 0.816 | 0.713 | 0.105 |
| Exp. 2 · Meditron + Littmann C2SE | spectral+wavelet | no | 214 | 0.794 | 0.701 | 0.858 | 0.783 | 0.852 | 0.780 | 0.104 |
| Exp. 2 · Meditron + Littmann C2SE | mfcc+melspec+spectral+wavelet | no | 214 | 0.752 | 0.655 | 0.819 | 0.740 | 0.842 | 0.737 | 0.108 |
| Exp. 2 · Meditron + Littmann C2SE | device only (no audio) | no | 214 | 0.664 | 0.586 | 0.717 | 0.651 | 0.546 | 0.651 | 0.204 |

## B) Can the features predict the recording device? (balanced accuracy, patient-grouped 5-fold CV)

| features | 4 devices, all recordings (chance 0.25) | Meditron vs Littmann C2SE (chance 0.50) |
|---|---|---|
| mfcc | 0.831 | 0.850 |
| spectral+wavelet | 0.833 | 0.857 |
| mfcc+melspec+spectral+wavelet | 0.850 | 0.867 |

## C) Exp. 2 out-of-fold results within each device

| features | device | COPD | non-COPD | accuracy | recall | specificity | auc | icbhi_score |
|---|---|---|---|---|---|---|---|---|
| mfcc | Meditron | 36 | 91 | 0.843 | 0.833 | 0.846 | 0.917 | 0.840 |
| mfcc | LittC2SE | 51 | 36 | 0.552 | 0.529 | 0.583 | 0.629 | 0.556 |
| spectral+wavelet | Meditron | 36 | 91 | 0.866 | 0.750 | 0.912 | 0.893 | 0.831 |
| spectral+wavelet | LittC2SE | 51 | 36 | 0.690 | 0.667 | 0.722 | 0.756 | 0.694 |

## D) Wavelet relative band energy per device and class (Exp. 2 subset)

| device | class | n | A5 | D5 | D4 | D3 | D2 | D1 |
|---|---|---|---|---|---|---|---|---|
| Meditron | COPD | 36 | 0.018 | 0.352 | 0.450 | 0.104 | 0.044 | 0.032 |
| Meditron | non-COPD | 91 | 0.038 | 0.260 | 0.405 | 0.178 | 0.079 | 0.039 |
| LittC2SE | COPD | 51 | 0.060 | 0.275 | 0.492 | 0.131 | 0.037 | 0.005 |
| LittC2SE | non-COPD | 36 | 0.119 | 0.322 | 0.420 | 0.094 | 0.039 | 0.006 |

Runtime: 67 s