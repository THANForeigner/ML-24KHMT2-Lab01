"""Backwards-compatible Streamlit import for the shared S6 inference loader."""
from lab01_ml.models.piecewise_compat import (
    CLASSES,
    FixedScoreHinges,
    InferenceOnly,
    NotebookUnpickler,
    QuantileHinges,
    SplineUseCaseFeatures,
    UseCaseSkillFeatures,
    load_piecewise,
)
