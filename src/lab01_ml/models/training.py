"""Development-only model evaluation and deterministic selection."""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from lab01_ml.features.engineering import InteractionFeatures
from lab01_ml.data.loader import load_frozen, get_partition_path

TARGET = 'Post_Semester_GPA'

def metrics(y, pred):
    mse = mean_squared_error(y, pred)
    return {'mae': float(mean_absolute_error(y, pred)), 'mse': float(mse),
            'rmse': float(np.sqrt(mse)), 'r2': float(r2_score(y, pred))}

def pipeline(model=3, scenario='full'):
    return Pipeline([('features', InteractionFeatures(model=model, scenario=scenario)),
                     ('ols', LinearRegression(fit_intercept=True))])


def development(train, validation, include_i4=True):
    X, y = train.drop(columns=[TARGET]), train[TARGET]
    xv, yv = validation.drop(columns=[TARGET]), validation[TARGET]
    candidates = [(i, 'full') for i in range(5 if include_i4 else 4)] + [(3, 'no_anxiety'), (3, 'no_d')]
    folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(X))
    rows, fitted, valpred = [], {}, {}
    for model, scenario in candidates:
        name = f'I{model}' + ('' if scenario == 'full' else f'_{scenario}')
        p = pipeline(model, scenario)
        foldrows = []
        for index, (tr, va) in enumerate(folds):
            fitted_fold = clone(p).fit(X.iloc[tr], y.iloc[tr])
            scores = metrics(y.iloc[va], fitted_fold.predict(X.iloc[va]))
            foldrows.append(scores)
            rows.append({'model': name, 'fold': index, **scores})
        p.fit(X, y)
        fitted[name] = p
        valpred[name] = p.predict(xv)
    table = pd.DataFrame(rows)
    comparison = table.groupby('model')[['mae','mse','rmse','r2']].agg(['mean', 'std'])
    comparison.columns = [f'cv_{metric}_{stat}' for metric, stat in comparison.columns]
    for name, pred in valpred.items():
        for key, value in metrics(yv, pred).items():
            comparison.loc[name, f'val_{key}'] = value
        comparison.loc[name, 'terms'] = len(fitted[name].named_steps['features'].get_feature_names_out())
    comparison = comparison.reset_index()
    main = comparison[comparison['model'].isin([f'I{i}' for i in range(5 if include_i4 else 4)])]
    best = main['cv_rmse_mean'].min()
    tied = main[main['cv_rmse_mean'] <= best * 1.001] if best else main[main['cv_rmse_mean'] == 0]
    selected = tied.sort_values(['terms', 'model']).iloc[0]['model']
    return table, comparison, fitted, valpred, selected
