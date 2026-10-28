"""Small scikit-learn-compatible helpers used by the comprehensive model comparison.

Defined in an importable module (not in a notebook) so the fitted pipeline can be pickled and loaded by the API.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight


class WeightedClassifier(ClassifierMixin, BaseEstimator):
    """Wrap a classifier that has no `class_weight` argument (GradientBoosting, AdaBoost, XGBoost…).

    * balance=True  -> training rows are given 'balanced' sample weights computed from the labels it is fitted on
                       (so inside cross-validation the weights come from that fold's training rows only).
    * encode_labels -> string labels are mapped to 0..k-1 for libraries that require numbers (XGBoost) and mapped back.
    * predict uses the arg-max of predict_proba so every wrapped model decides the same way.
    """

    def __init__(self, base=None, balance: bool = True, encode_labels: bool = False):
        self.base = base
        self.balance = balance
        self.encode_labels = encode_labels

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.le_ = LabelEncoder().fit(y)                           # sorted classes, identical order to self.classes_
        yy = self.le_.transform(y) if self.encode_labels else y
        self.base_ = clone(self.base)
        if self.balance:
            self.base_.fit(X, yy, sample_weight=compute_sample_weight("balanced", y))
        else:
            self.base_.fit(X, yy)
        return self

    def predict_proba(self, X):
        return np.asarray(self.base_.predict_proba(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
