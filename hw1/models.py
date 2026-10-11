"""Classifiers (homework step 6).

SIMPLIFICATION vs. the paper: the paper trains a 4-block 2-D CNN (Keras) on 40x862 MFCC
maps. We use classical classifiers on MFCC summary vectors instead:
  * SVM (RBF)            - main model
  * Random Forest        - comparison
  * MLP (1-2 hidden layers) - small neural-network comparison
Reasons: only 127 non-COPD recordings from ~60 patients, CPU-only, one-week scope, and a
light installation (no TensorFlow/PyTorch) so the demo runs on any laptop.

Every model is a scikit-learn Pipeline whose first step is StandardScaler, so feature
scaling is fitted on training data only. Hyper-parameters are chosen with patient-grouped
cross-validation INSIDE the training set; the test set is never used for tuning.
"""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config as C

MODEL_LABELS = {"svm": "SVM (RBF)", "rf": "Random Forest", "mlp": "MLP"}


def _spec(name: str, seed: int):
    if name == "svm":
        # Platt-scaled SVM -> P(COPD). (Replaces the deprecated SVC(probability=True).)
        clf = CalibratedClassifierCV(SVC(kernel="rbf", class_weight="balanced", random_state=seed),
                                     method="sigmoid", cv=3, ensemble=False)
        grid = {"clf__estimator__C": [0.1, 1, 10, 100], "clf__estimator__gamma": ["scale", 0.01, 0.001]}
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                     random_state=seed, n_jobs=-1)
        grid = {"clf__max_depth": [None, 8], "clf__min_samples_leaf": [1, 3]}
    elif name == "mlp":
        clf = MLPClassifier(max_iter=2000, early_stopping=False, random_state=seed)
        grid = {"clf__hidden_layer_sizes": [(64,), (128, 64)], "clf__alpha": [1e-3, 1e-2, 1e-1]}
    else:
        raise ValueError(name)
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)]), grid


def fit_model(name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
              seed: int = C.RANDOM_SEED, n_folds: int = C.CV_FOLDS):
    """Grid-search with StratifiedGroupKFold (groups = patient id) and refit on all of X.

    Augmented copies carry their source patient's id, so a patient and all of its
    augmented copies always fall in the same CV fold.
    """
    pipe, grid = _spec(name, seed)
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    search = GridSearchCV(pipe, grid, cv=cv, scoring="balanced_accuracy", n_jobs=-1, refit=True)
    search.fit(X, y, groups=groups)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def predict_proba_copd(model, X: np.ndarray) -> np.ndarray:
    """P(COPD). Predicted label = P(COPD) >= 0.5 everywhere (scripts and app agree)."""
    return model.predict_proba(X)[:, list(model.classes_).index(1)]
