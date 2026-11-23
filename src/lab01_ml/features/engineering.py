"""Train-only preprocessing and reference-coded nested interaction designs."""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

VERSION = '1.0.0'
H = 'Weekly_GenAI_Hours'
P = 'Pre_Semester_GPA'
S = 'Traditional_Study_Hours'
NUMERIC = [H, P, S, 'Tool_Diversity', 'Paid_Subscription']
CATEGORICAL = ['Year_of_Study', 'Major_Category', 'Primary_Use_Case',
               'Prompt_Engineering_Skill', 'Institutional_Policy',
               'Perceived_AI_Dependency', 'Anxiety_Level_During_Exams']
INPUTS = [P, S, 'Year_of_Study', 'Major_Category', H, 'Primary_Use_Case',
          'Prompt_Engineering_Skill', 'Tool_Diversity', 'Paid_Subscription',
          'Institutional_Policy', 'Perceived_AI_Dependency', 'Anxiety_Level_During_Exams']


class InteractionFeatures(BaseEstimator, TransformerMixin):
    """Fit all imputation, category and centering parameters on training rows only."""
    def __init__(self, model=3, scenario='full'):
        self.model = model
        self.scenario = scenario

    def _columns(self):
        if self.scenario not in ('full', 'no_anxiety', 'no_d'):
            raise ValueError('scenario must be full, no_anxiety or no_d')
        removed = {'Anxiety_Level_During_Exams'} if self.scenario == 'no_anxiety' else set()
        if self.scenario == 'no_d':
            removed = {'Anxiety_Level_During_Exams', 'Perceived_AI_Dependency'}
        return [c for c in CATEGORICAL if c not in removed]

    @staticmethod
    def _numeric(frame):
        out = frame[NUMERIC].copy()
        paid = out['Paid_Subscription']
        out['Paid_Subscription'] = paid.map(lambda v: {'true': 1, 'false': 0, '1': 1, '0': 0}.get(str(v).strip().lower(), v) if pd.notna(v) else np.nan)
        for c in NUMERIC:
            out[c] = pd.to_numeric(out[c], errors='coerce')
        return out.replace([np.inf, -np.inf], np.nan)

    def fit(self, X, y=None):
        if self.model not in range(5):
            raise ValueError('model must be an integer from 0 to 4')
        self.cat_columns_ = self._columns()
        self.required_ = NUMERIC + self.cat_columns_
        self._validate(X)
        num = self._numeric(X)
        self.medians_ = num.median()
        if self.medians_.isna().any():
            raise ValueError('All training values missing: ' + ', '.join(self.medians_.index[self.medians_.isna()]))
        num = num.fillna(self.medians_)
        self.means_ = num[[H, P, S]].mean().to_dict()
        cat = X[self.cat_columns_].copy()
        self.modes_ = {}
        for c in self.cat_columns_:
            present = cat[c].dropna().astype(str)
            if present.empty:
                raise ValueError(f'All training values missing: {c}')
            self.modes_[c] = present.mode().sort_values().iloc[0]
            cat[c] = cat[c].map(lambda x: str(x) if pd.notna(x) else self.modes_[c])
        self.encoder_ = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
        self.encoder_.fit(cat)
        self.references_ = {c: str(categories[0]) for c, categories in zip(self.cat_columns_, self.encoder_.categories_)}
        self._names = ['Hc', 'Pc', 'Sc', 'Tool_Diversity', 'Paid_Subscription']
        self._dummies = []
        self._indices = {}
        for c, categories in zip(self.cat_columns_, self.encoder_.categories_):
            self._indices[c] = []
            for level in categories[1:]:
                self._indices[c].append(len(self._dummies))
                self._dummies.append(f'{c}={level}')
        self._names.extend(self._dummies)
        use = self._indices['Primary_Use_Case']
        skill = self._indices['Prompt_Engineering_Skill']
        if self.model >= 1:
            self._names.extend('Hc * ' + self._dummies[j] for j in skill)
        if self.model >= 2:
            self._names.extend('Hc * ' + self._dummies[i] for i in use)
        if self.model >= 3:
            self._names.extend(self._dummies[i] + ' * ' + self._dummies[j] for i in use for j in skill)
        if self.model >= 4:
            self._names.extend(['Hc * Pc', 'Hc * Sc'])
        return self

    def _validate(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError('Expected a pandas DataFrame with raw feature columns')
        missing = [c for c in self.required_ if c not in X]
        if missing:
            raise ValueError('Missing required input columns: ' + ', '.join(missing))

    def transform(self, X):
        check_is_fitted(self, 'encoder_')
        self._validate(X)
        num = self._numeric(X).fillna(self.medians_)
        cat = X[self.cat_columns_].copy()
        for c, levels in zip(self.cat_columns_, self.encoder_.categories_):
            cat[c] = cat[c].map(lambda x: str(x) if pd.notna(x) else self.modes_[c])
            unseen = sorted(set(cat[c]) - set(levels))
            if unseen:
                warnings.warn(f'{c}: unknown categories {unseen}; encoded as reference-like zeros', UserWarning)
        encoded = self.encoder_.transform(cat)
        hc = num[H].to_numpy(dtype=float) - self.means_[H]
        pc = num[P].to_numpy(dtype=float) - self.means_[P]
        sc = num[S].to_numpy(dtype=float) - self.means_[S]
        cols = [hc, pc, sc, num['Tool_Diversity'].to_numpy(dtype=float), num['Paid_Subscription'].to_numpy(dtype=float)]
        cols.extend(encoded[:, i] for i in range(encoded.shape[1]))
        use, skill = self._indices['Primary_Use_Case'], self._indices['Prompt_Engineering_Skill']
        if self.model >= 1:
            cols.extend(hc * encoded[:, j] for j in skill)
        if self.model >= 2:
            cols.extend(hc * encoded[:, i] for i in use)
        if self.model >= 3:
            cols.extend(encoded[:, i] * encoded[:, j] for i in use for j in skill)
        if self.model >= 4:
            cols.extend([hc * pc, hc * sc])
        result = np.column_stack(cols)
        if not np.isfinite(result).all():
            raise ValueError('Transformed features contain non-finite values')
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'encoder_')
        return np.asarray(self._names, dtype=object)
