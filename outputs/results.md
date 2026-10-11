# Results


Positive class = COPD. recall = sensitivity. Threshold P(COPD) >= 0.5. Test set: 369 recordings from 47 unseen patients (COPD 337, non-COPD 32).

## Experiment 1 - patient-disjoint official split

| model | features | augmentation | accuracy | precision | recall | specificity | f1 | macro_f1 | auc | icbhi_score | cv_bal_acc_train |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SVM (RBF) | mfcc | yes | 0.927 | 0.997 | 0.923 | 0.969 | 0.958 | 0.828 | 0.989 | 0.946 | 0.891 |
| Random Forest | mfcc | yes | 0.919 | 0.994 | 0.917 | 0.938 | 0.954 | 0.810 | 0.967 | 0.927 | 0.859 |
| MLP | mfcc | yes | 0.938 | 0.994 | 0.938 | 0.938 | 0.965 | 0.844 | 0.987 | 0.938 | 0.843 |
| SVM (RBF) | melspec | yes | 0.873 | 0.990 | 0.869 | 0.906 | 0.926 | 0.739 | 0.969 | 0.888 | 0.913 |
| SVM (RBF) | chroma | yes | 0.629 | 0.963 | 0.617 | 0.750 | 0.752 | 0.506 | 0.777 | 0.684 | 0.658 |
| SVM (RBF) | mfcc | no | 0.978 | 0.991 | 0.985 | 0.906 | 0.988 | 0.933 | 0.988 | 0.946 | 0.783 |
| Device-only baseline (no audio) | device name | - | 0.984 | 1.000 | 0.982 | 1.000 | 0.991 | 0.953 | 0.991 | 0.991 | nan |

### Main model (SVM + MFCC) accuracy per device on the test set

| device | label_name | recordings | accuracy |
|---|---|---|---|
| AKGC417L | COPD | 277 | 0.924 |
| Litt3200 | COPD | 54 | 0.981 |
| Meditron | COPD | 6 | 0.333 |
| Meditron | non-COPD | 32 | 0.969 |

## Experiment 2 - Meditron + LittC2SE only, 5-fold patient-grouped CV (214 recordings)

| model | accuracy | precision | recall | specificity | f1 | macro_f1 | auc | icbhi_score |
|---|---|---|---|---|---|---|---|---|
| SVM + MFCC (pooled out-of-fold) | 0.724 | 0.663 | 0.655 | 0.772 | 0.659 | 0.714 | 0.816 | 0.713 |
| SVM + MFCC (mean of folds) | 0.724 | 0.642 | 0.652 | 0.772 | 0.633 | 0.702 | 0.830 | 0.712 |
| SVM + MFCC (std of folds) | 0.105 | 0.126 | 0.269 | 0.080 | 0.206 | 0.133 | 0.095 | 0.130 |
| Device-only baseline (pooled) | 0.664 | 0.586 | 0.586 | 0.717 | 0.586 | 0.651 | 0.546 | 0.651 |

### Per fold

| fold | test_patients | accuracy | precision | recall | specificity | f1 | macro_f1 | auc | icbhi_score |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 15 | 0.767 | 0.652 | 0.882 | 0.692 | 0.750 | 0.766 | 0.871 | 0.787 |
| 2 | 18 | 0.605 | 0.500 | 0.235 | 0.846 | 0.320 | 0.521 | 0.785 | 0.541 |
| 3 | 17 | 0.619 | 0.529 | 0.529 | 0.680 | 0.529 | 0.605 | 0.687 | 0.605 |
| 4 | 18 | 0.837 | 0.789 | 0.833 | 0.840 | 0.811 | 0.834 | 0.929 | 0.837 |
| 5 | 18 | 0.791 | 0.737 | 0.778 | 0.800 | 0.757 | 0.786 | 0.878 | 0.789 |