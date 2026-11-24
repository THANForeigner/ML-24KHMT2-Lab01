"""Elastic Net specific evaluation: sparsity, selection frequency, hierarchy, and ceiling checks."""
import numpy as np
import pandas as pd
from lab01_ml.features.elastic_net_features import INPUTS, TARGET
from lab01_ml.models.elastic_net_training import metrics


def extract_sparsity_metrics(pipeline, threshold=1e-6):
    """Extract number of non-zero coefficients and sparsity percentage."""
    regressor = pipeline.named_steps.get('regressor', pipeline.named_steps.get('elasticnet'))
    transformer = pipeline.named_steps['features']
    feature_names = transformer.get_feature_names_out()

    coefs = regressor.coef_
    is_non_zero = np.abs(coefs) > threshold
    k = int(np.sum(is_non_zero))
    total_p = len(coefs)
    sparsity_pct = float((1.0 - (k / total_p)) * 100.0)

    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefs,
        'abs_coefficient': np.abs(coefs),
        'is_non_zero': is_non_zero
    }).sort_values(by='abs_coefficient', ascending=False).reset_index(drop=True)

    return k, sparsity_pct, coef_df


def calculate_selection_frequency(feature_names, fold_coefs, threshold=1e-6):
    """Calculate selection frequency across folds for each feature."""
    n_folds = len(fold_coefs)
    if n_folds == 0:
        return pd.DataFrame(columns=['feature', 'selected_folds', 'total_folds', 'frequency'])

    # Matrix of shape (n_folds, n_features)
    matrix = np.array([np.abs(c) > threshold for c in fold_coefs], dtype=int)
    counts = np.sum(matrix, axis=0)
    freqs = counts / float(n_folds)

    df = pd.DataFrame({
        'feature': feature_names,
        'selected_folds': counts,
        'total_folds': n_folds,
        'frequency': freqs
    }).sort_values(by=['frequency', 'feature'], ascending=[False, True]).reset_index(drop=True)

    return df


def check_hierarchy(feature_names, coefs, threshold=1e-6):
    """Check whether interaction terms violate hierarchical heredity.
    A violation occurs if an interaction term has a non-zero coefficient,
    but one or both of its constituent main effects have coefficients shrunken to zero.
    """
    coef_map = {name: (abs(val) > threshold) for name, val in zip(feature_names, coefs)}
    violations = []

    for name, is_active in coef_map.items():
        if not is_active:
            continue
        if '*' in name:
            parts = [p.strip() for p in name.split('*')]
            missing_mains = [p for p in parts if not coef_map.get(p, False)]
            if missing_mains:
                violations.append({
                    'interaction': name,
                    'missing_main_effects': missing_mains
                })

    return violations


def check_gpa_bounds(y_pred):
    """Check how many predictions fall outside valid GPA bounds [0.0, 4.0].
    Strictly reports counts without clipping.
    """
    arr = np.asarray(y_pred, dtype=float)
    below_zero = int(np.sum(arr < 0.0))
    above_four = int(np.sum(arr > 4.0))
    total = len(arr)
    out_of_bounds = below_zero + above_four
    pct = float((out_of_bounds / total) * 100.0) if total > 0 else 0.0

    return {
        'total_predictions': total,
        'below_zero_count': below_zero,
        'above_four_count': above_four,
        'out_of_bounds_count': out_of_bounds,
        'out_of_bounds_pct': pct
    }


def evaluate_on_validation(models_dict, val_df):
    """Evaluate multiple models on the validation partition and construct comparison table."""
    records = []
    y_true = val_df[TARGET]

    for name, pipe in models_dict.items():
        y_pred = pipe.predict(val_df)
        score = metrics(y_true, y_pred)
        bounds = check_gpa_bounds(y_pred)

        # Get number of terms and sparsity
        regressor = pipe.named_steps.get('regressor', pipe.named_steps.get('ols'))
        coefs = getattr(regressor, 'coef_', None)
        num_terms = len(coefs) if coefs is not None else 0
        non_zero = int(np.sum(np.abs(coefs) > 1e-6)) if coefs is not None else num_terms
        sparsity = float((1.0 - (non_zero / num_terms)) * 100.0) if num_terms > 0 else 0.0

        records.append({
            'model_name': name,
            'num_terms': num_terms,
            'non_zero_terms': non_zero,
            'sparsity_pct': sparsity,
            'val_rmse': score['rmse'],
            'val_mae': score['mae'],
            'val_mse': score['mse'],
            'val_r2': score['r2'],
            'below_zero_count': bounds['below_zero_count'],
            'above_four_count': bounds['above_four_count'],
            'pct_out_of_bounds': bounds['out_of_bounds_pct']
        })

    return pd.DataFrame(records)
