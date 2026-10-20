# Feature selection study


Start: all 354 features (mfcc+melspec+spectral+wavelet). Selector is inside the Pipeline (fitted on training folds only), then calibrated RBF-SVM. 'tuned' = k chosen by inner patient-grouped CV. kept_exp2_mean = features kept, mean over the 5 outer folds. E2 = Meditron + Littmann C2SE, 5-fold patient-grouped CV (main criterion).

| method | k | kept_exp1 | kept_exp2_mean | E1_icbhi_score | E1_auc | E2_accuracy | E2_recall | E2_specificity | E2_macro_f1 | E2_auc | E2_icbhi_score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| No selection | all | 354 | 354.000 | 0.973 | 0.997 | 0.771 | 0.678 | 0.835 | 0.759 | 0.852 | 0.756 |
| ANOVA F | 20 | 20 | 20.000 | 0.891 | 0.945 | 0.696 | 0.471 | 0.850 | 0.663 | 0.688 | 0.661 |
| ANOVA F | 40 | 40 | 40.000 | 0.930 | 0.989 | 0.720 | 0.621 | 0.787 | 0.706 | 0.757 | 0.704 |
| ANOVA F | 80 | 80 | 80.000 | 0.929 | 0.992 | 0.715 | 0.621 | 0.779 | 0.702 | 0.783 | 0.700 |
| ANOVA F | 160 | 160 | 160.000 | 0.941 | 0.995 | 0.729 | 0.667 | 0.772 | 0.719 | 0.825 | 0.719 |
| ANOVA F | tuned | 20 | 132.000 | 0.891 | 0.945 | 0.706 | 0.632 | 0.756 | 0.694 | 0.807 | 0.694 |
| Mutual information | 20 | 20 | 20.000 | 0.898 | 0.965 | 0.738 | 0.644 | 0.803 | 0.726 | 0.742 | 0.723 |
| Mutual information | 40 | 40 | 40.000 | 0.908 | 0.978 | 0.724 | 0.632 | 0.787 | 0.712 | 0.774 | 0.710 |
| Mutual information | 80 | 80 | 80.000 | 0.902 | 0.979 | 0.729 | 0.632 | 0.795 | 0.716 | 0.777 | 0.714 |
| Mutual information | 160 | 160 | 160.000 | 0.956 | 0.995 | 0.762 | 0.678 | 0.819 | 0.751 | 0.814 | 0.749 |
| Mutual information | tuned | 80 | 144.000 | 0.902 | 0.979 | 0.766 | 0.690 | 0.819 | 0.756 | 0.816 | 0.754 |
| Random-forest rank | 20 | 20 | 20.000 | 0.930 | 0.986 | 0.696 | 0.575 | 0.779 | 0.679 | 0.728 | 0.677 |
| Random-forest rank | 40 | 40 | 40.000 | 0.917 | 0.984 | 0.724 | 0.552 | 0.843 | 0.702 | 0.750 | 0.697 |
| Random-forest rank | 80 | 80 | 80.000 | 0.970 | 0.997 | 0.748 | 0.655 | 0.811 | 0.735 | 0.809 | 0.733 |
| Random-forest rank | 160 | 160 | 160.000 | 0.974 | 0.998 | 0.771 | 0.713 | 0.811 | 0.762 | 0.838 | 0.762 |
| Random-forest rank | tuned | 80 | 160.000 | 0.970 | 0.997 | 0.771 | 0.713 | 0.811 | 0.762 | 0.838 | 0.762 |
| PCA (compression) | 20 | 20 | 20.000 | 0.959 | 0.992 | 0.804 | 0.678 | 0.890 | 0.790 | 0.854 | 0.784 |
| PCA (compression) | 40 | 40 | 40.000 | 0.961 | 0.995 | 0.752 | 0.644 | 0.827 | 0.739 | 0.839 | 0.735 |
| PCA (compression) | 80 | 80 | 80.000 | 0.978 | 0.997 | 0.780 | 0.701 | 0.835 | 0.770 | 0.849 | 0.768 |
| PCA (compression) | tuned | 20 | 48.000 | 0.959 | 0.992 | 0.776 | 0.678 | 0.843 | 0.764 | 0.841 | 0.760 |
| L1 logistic regression | L1 | 37 | 15.400 | 0.954 | 0.996 | 0.748 | 0.667 | 0.803 | 0.737 | 0.797 | 0.735 |
| L1 logistic regression | tuned | 37 | 19.000 | 0.954 | 0.996 | 0.734 | 0.644 | 0.795 | 0.721 | 0.772 | 0.720 |

## Share of feature groups among selected features (Exp. 2, 5 folds)

| method | mfcc | melspec | spectral | wavelet |
|---|---|---|---|---|
| All 354 features | 0.452 | 0.362 | 0.068 | 0.119 |
| ANOVA F (k=40) | 0.425 | 0.455 | 0.095 | 0.025 |
| Mutual information (k=40) | 0.390 | 0.420 | 0.150 | 0.040 |
| Random-forest rank (k=40) | 0.470 | 0.335 | 0.120 | 0.075 |
| L1 logistic regression | 0.792 | 0.117 | 0.091 | 0.000 |

## Most often selected features - ANOVA F, k = 40 (out of 5 folds)

| method | feature | group | folds_selected |
|---|---|---|---|
| ANOVA F | contrast_3_mean | spectral | 5 |
| ANOVA F | mel27_std | melspec | 5 |
| ANOVA F | mel29_std | melspec | 5 |
| ANOVA F | mel28_std | melspec | 5 |
| ANOVA F | dmfcc4_std | mfcc | 5 |
| ANOVA F | dmfcc3_std | mfcc | 5 |
| ANOVA F | dmfcc0_std | mfcc | 5 |
| ANOVA F | dmfcc8_std | mfcc | 5 |
| ANOVA F | mel15_std | melspec | 5 |
| ANOVA F | mfcc6_std | mfcc | 5 |
| ANOVA F | mfcc4_std | mfcc | 5 |
| ANOVA F | mfcc3_std | mfcc | 5 |
| ANOVA F | mel16_std | melspec | 5 |
| ANOVA F | dmfcc7_std | mfcc | 5 |
| ANOVA F | dmfcc6_std | mfcc | 5 |

Runtime: 290 s