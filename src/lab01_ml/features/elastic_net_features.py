"""Elastic Net feature transformer with EN1, EN2, EN3 and Ablation specifications."""
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
TARGET = 'Post_Semester_GPA'

NUMERIC = [H, P, S, 'Tool_Diversity', 'Paid_Subscription']
CATEGORICAL = [
    'Year_of_Study', 'Major_Category', 'Primary_Use_Case',
    'Prompt_Engineering_Skill', 'Institutional_Policy',
    'Perceived_AI_Dependency', 'Anxiety_Level_During_Exams'
]
INPUTS = [
    P, S, 'Year_of_Study', 'Major_Category', H, 'Primary_Use_Case',
    'Prompt_Engineering_Skill', 'Tool_Diversity', 'Paid_Subscription',
    'Institutional_Policy', 'Perceived_AI_Dependency', 'Anxiety_Level_During_Exams'
]


class ElasticNetFeatures(BaseEstimator, TransformerMixin):
    """Transformer for Elastic Net Feature Sets:
    - EN1: 39 terms (Main effects, reference OHE drop-first, centered Hc/Pc/Sc, Tool_Diversity, Paid_Subscription)
    - EN2: 40-41 terms (EN1 + Hc^2; optional Pc^2; raw H is excluded when Hc is present with intercept)
    - EN3: 54-55 terms (EN2 + 14 two-way interactions: Hc x Use, Hc x Skill, Use x Skill)
    - scenario='no_anxiety': excludes Anxiety_Level_During_Exams (45 terms for EN3)
    """

    def __init__(self, model_type='EN3', scenario='full', include_pc_squared=False):
        self.model_type = model_type.upper()
        self.scenario = scenario
        self.include_pc_squared = include_pc_squared

    def _columns(self):
        if self.scenario not in ('full', 'no_anxiety', 'no_d'):
            raise ValueError("scenario must be 'full', 'no_anxiety', or 'no_d'")
        removed = set()
        if self.scenario == 'no_anxiety':
            removed = {'Anxiety_Level_During_Exams'}
        elif self.scenario == 'no_d':
            removed = {'Anxiety_Level_During_Exams', 'Perceived_AI_Dependency'}
        return [c for c in CATEGORICAL if c not in removed]

    @staticmethod
    def _numeric(frame):
        out = frame[NUMERIC].copy()
        paid = out['Paid_Subscription']
        out['Paid_Subscription'] = paid.map(
            lambda v: {'true': 1, 'false': 0, '1': 1, '0': 0}.get(str(v).strip().lower(), v)
            if pd.notna(v) else np.nan
        )
        for c in NUMERIC:
            out[c] = pd.to_numeric(out[c], errors='coerce')
        return out.replace([np.inf, -np.inf], np.nan)

    def _validate(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError('Expected a pandas DataFrame with raw feature columns')
        missing = [c for c in self.required_ if c not in X.columns]
        if missing:
            raise ValueError('Missing required input columns: ' + ', '.join(missing))

    def fit(self, X, y=None):
        if self.model_type not in ('EN1', 'EN2', 'EN3'):
            raise ValueError("model_type must be 'EN1', 'EN2', or 'EN3'")
        self.cat_columns_ = self._columns()
        self.required_ = NUMERIC + self.cat_columns_
        self._validate(X)

        num = self._numeric(X)
        self.medians_ = num.median()
        if self.medians_.isna().any():
            missing_cols = self.medians_.index[self.medians_.isna()]
            raise ValueError('All training values missing in numeric columns: ' + ', '.join(missing_cols))

        num = num.fillna(self.medians_)
        self.means_ = num[[H, P, S]].mean().to_dict()

        cat = X[self.cat_columns_].copy()
        self.modes_ = {}
        for c in self.cat_columns_:
            present = cat[c].dropna().astype(str)
            if present.empty:
                raise ValueError(f'All training values missing in categorical column: {c}')
            self.modes_[c] = present.mode().sort_values().iloc[0]
            cat[c] = cat[c].map(lambda x: str(x) if pd.notna(x) else self.modes_[c])

        self.encoder_ = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
        self.encoder_.fit(cat)
        self.references_ = {
            c: str(categories[0])
            for c, categories in zip(self.cat_columns_, self.encoder_.categories_)
        }

        # Build feature names
        # EN1 base: Hc, Pc, Sc, Tool_Diversity, Paid_Subscription + OHE dummies
        self._names = ['Hc', 'Pc', 'Sc', 'Tool_Diversity', 'Paid_Subscription']
        self._dummies = []
        self._indices = {}

        for c, categories in zip(self.cat_columns_, self.encoder_.categories_):
            self._indices[c] = []
            for level in categories[1:]:
                self._indices[c].append(len(self._dummies))
                self._dummies.append(f'{c}={level}')
        self._names.extend(self._dummies)

        # EN2 polynomial additions
        if self.model_type in ('EN2', 'EN3'):
            self._names.append('Hc^2')
            if self.include_pc_squared:
                self._names.append('Pc^2')

        # EN3 interaction additions
        if self.model_type == 'EN3':
            use = self._indices.get('Primary_Use_Case', [])
            skill = self._indices.get('Prompt_Engineering_Skill', [])
            self._names.extend('Hc * ' + self._dummies[i] for i in use)
            self._names.extend('Hc * ' + self._dummies[j] for j in skill)
            self._names.extend(
                self._dummies[i] + ' * ' + self._dummies[j]
                for i in use for j in skill
            )

        return self

    def transform(self, X):
        check_is_fitted(self, 'encoder_')
        self._validate(X)

        num = self._numeric(X).fillna(self.medians_)
        cat = X[self.cat_columns_].copy()

        for c, levels in zip(self.cat_columns_, self.encoder_.categories_):
            cat[c] = cat[c].map(lambda x: str(x) if pd.notna(x) else self.modes_[c])
            unseen = sorted(set(cat[c]) - set(levels))
            if unseen:
                warnings.warn(
                    f'{c}: unknown categories {unseen}; encoded as reference-like zeros',
                    UserWarning
                )

        encoded = self.encoder_.transform(cat)
        hc = num[H].to_numpy(dtype=float) - self.means_[H]
        pc = num[P].to_numpy(dtype=float) - self.means_[P]
        sc = num[S].to_numpy(dtype=float) - self.means_[S]

        cols = [
            hc, pc, sc,
            num['Tool_Diversity'].to_numpy(dtype=float),
            num['Paid_Subscription'].to_numpy(dtype=float)
        ]
        cols.extend(encoded[:, i] for i in range(encoded.shape[1]))

        if self.model_type in ('EN2', 'EN3'):
            cols.append(hc ** 2)
            if self.include_pc_squared:
                cols.append(pc ** 2)

        if self.model_type == 'EN3':
            use = self._indices.get('Primary_Use_Case', [])
            skill = self._indices.get('Prompt_Engineering_Skill', [])
            cols.extend(hc * encoded[:, i] for i in use)
            cols.extend(hc * encoded[:, j] for j in skill)
            cols.extend(encoded[:, i] * encoded[:, j] for i in use for j in skill)

        result = np.column_stack(cols)
        if not np.isfinite(result).all():
            raise ValueError('Transformed features contain non-finite values (inf or nan)')
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'encoder_')
        return np.asarray(self._names, dtype=object)
