"""Dinh nghia cot, tien xu ly va model dung chung cho notebook Ridge (va App)."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "Post_Semester_GPA"
NUM = ["Pre_Semester_GPA", "Traditional_Study_Hours", "Weekly_GenAI_Hours", "Tool_Diversity"]
BIN = ["Paid_Subscription"]
CAT = ["Year_of_Study", "Major_Category", "Primary_Use_Case",
       "Prompt_Engineering_Skill", "Institutional_Policy", "Perceived_AI_Dependency"]
ANX = "Anxiety_Level_During_Exams"       # chi dung khi CONFIG["use_anxiety"] hoac o phan 8
FORBIDDEN = ["Student_ID", "GPA_Change", "Skill_Retention_Score", "Burnout_Risk_Level"]


def make_X(df, use_anxiety):
    return df[NUM + BIN + CAT + ([ANX] if use_anxiety else [])]


class GenAIPoly(BaseEstimator, TransformerMixin):
    """[H, (H - mean_train)^2] cho genai_mode = 'raw+sq'. Mean hoc tu train."""
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        return self
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.hstack([X, (X - self.mean_) ** 2])
    def get_feature_names_out(self, input_features=None):
        n = input_features[0]
        return np.array([n, f"{n}_sq"])


def _genai_transformer(mode):
    if mode == "raw":
        return StandardScaler()
    if mode == "log1p":
        return Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                         ("scale", StandardScaler())])
    if mode == "raw+sq":
        return Pipeline([("poly", GenAIPoly()), ("scale", StandardScaler())])
    raise ValueError(f"genai_mode khong hop le: {mode!r}")


def make_preprocessor(config):
    cat = CAT + ([ANX] if config["use_anxiety"] else [])
    ohe = OneHotEncoder(drop="first" if config["drop_first"] else None,
                        handle_unknown="ignore", sparse_output=False)
    cat_t = Pipeline([("ohe", ohe), ("scale", StandardScaler())]) if config["scale_onehot"] else ohe
    bin_t = StandardScaler() if config["scale_onehot"] else "passthrough"
    return ColumnTransformer(
        [("num", StandardScaler(), [c for c in NUM if c != "Weekly_GenAI_Hours"]),
         ("genai", _genai_transformer(config["genai_mode"]), ["Weekly_GenAI_Hours"]),
         ("cat", cat_t, cat),
         ("bin", bin_t, BIN)],
        verbose_feature_names_out=False,
    )


class ClippedRegressor(BaseEstimator, RegressorMixin):
    """Boc mot estimator, clip du doan o can tren. upper=None -> khong clip."""
    def __init__(self, estimator, upper=None):
        self.estimator = estimator
        self.upper = upper
    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, y)
        return self
    def predict(self, X):
        pred = self.estimator_.predict(X)
        return pred if self.upper is None else np.minimum(pred, self.upper)
    @property
    def coef_(self):
        return self.estimator_.coef_
    @property
    def intercept_(self):
        return self.estimator_.intercept_


def make_model(estimator, config):
    return Pipeline([("prep", make_preprocessor(config)),
                     ("model", ClippedRegressor(estimator, config["clip_upper"]))])
