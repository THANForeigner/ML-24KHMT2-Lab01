"""Shared inference compatibility for S6 classes persisted from the notebook.

The repository-owned S6 artifact was serialized with its transformers in
``__main__``. Map only those known classes when reading this one artifact.
These methods reproduce the notebook's *transform* methods; fitting remains
the notebook's responsibility.
"""
from pathlib import Path

import numpy as np
from joblib.numpy_pickle import NumpyUnpickler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class InferenceOnly(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        raise RuntimeError("Chỉ dùng S6 đã huấn luyện; không fit lại trong web.")


class QuantileHinges(InferenceOnly):
    def __init__(self, quantiles=(0.5,), feature_index=0):
        self.quantiles = quantiles
        self.feature_index = feature_index

    def transform(self, X):
        check_is_fitted(self, "knots_")
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("S6: số cột đầu vào hinge khác dữ liệu fit.")
        return np.column_stack([X, np.maximum(0, X[:, [self.feature_index]] - self.knots_)])


class SplineUseCaseFeatures(InferenceOnly):
    def __init__(self, pre_gpa_quantile=0.75):
        self.pre_gpa_quantile = pre_gpa_quantile

    def transform(self, X):
        check_is_fitted(self, "base_")
        design = self.base_.transform(X)
        products = [design[:, b] * design[:, u]
                    for u in self.use_indices_ for b in self.basis_indices_]
        result = np.column_stack([design, *products])
        if not np.isfinite(result).all():
            raise ValueError("S6: ma trận đặc trưng không hữu hạn.")
        return result


class FixedScoreHinges(InferenceOnly):
    def transform(self, X):
        check_is_fitted(self, "knots_")
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("S6: số cột đầu vào score khác dữ liệu fit.")
        return np.column_stack([X, np.maximum(0, X[:, self.indices_] - self.knots_)])


class UseCaseSkillFeatures(InferenceOnly):
    def __init__(self, pre_gpa_quantile=0.75):
        self.pre_gpa_quantile = pre_gpa_quantile

    def transform(self, X):
        check_is_fitted(self, "pairs_")
        design = self.base_.transform(X)
        products = [design[:, u] * design[:, k] for u, k in self.pairs_]
        return np.column_stack([design, *products])


CLASSES = {cls.__name__: cls for cls in (
    QuantileHinges, SplineUseCaseFeatures, FixedScoreHinges, UseCaseSkillFeatures
)}


class NotebookUnpickler(NumpyUnpickler):
    def find_class(self, module, name):
        if module == "__main__" and name in CLASSES:
            return CLASSES[name]
        return super().find_class(module, name)


def load_piecewise(path: Path):
    # Only the known, repository-owned joblib; do not accept user uploads.
    with path.open("rb") as handle:
        return NotebookUnpickler(str(path), handle, ensure_native_byte_order=True).load()
