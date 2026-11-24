"""5-Fold CV tuning, tie-break selection, and group ablation for Elastic Net."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from lab01_ml.features.elastic_net_features import INPUTS, TARGET
from lab01_ml.models.factory import create_elastic_net_pipeline


def metrics(y_true, y_pred):
    """Calculate MAE, MSE, RMSE, and R2 scores."""
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'mse': mse,
        'rmse': float(np.sqrt(mse)),
        'r2': float(r2_score(y_true, y_pred))
    }


def tune_elastic_net_cv(
    train_df,
    model_type='EN3',
    scenario='full',
    alphas=None,
    l1_ratios=None,
    n_splits=5,
    random_state=42,
    include_pc_squared=False
):
    """Tune alpha and l1_ratio using n-fold Cross-Validation strictly on training data."""
    if alphas is None:
        alphas = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
    if l1_ratios is None:
        l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]

    X = train_df[INPUTS]
    y = train_df[TARGET]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(kf.split(X, y))

    summary_records = []
    fold_details = []

    for alpha in alphas:
        for l1_ratio in l1_ratios:
            fold_metrics = []
            non_zero_counts = []
            fold_coefs = []

            for fold_idx, (tr_idx, val_idx) in enumerate(folds):
                X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
                X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

                pipe = create_elastic_net_pipeline(
                    model_type=model_type,
                    scenario=scenario,
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    include_pc_squared=include_pc_squared,
                    random_state=random_state
                )
                pipe.fit(X_tr, y_tr)
                preds = pipe.predict(X_va)
                scores = metrics(y_va, preds)

                coef = pipe.named_steps['regressor'].coef_
                nz = int(np.sum(np.abs(coef) > 1e-6))
                non_zero_counts.append(nz)
                fold_coefs.append(coef)

                record = {
                    'model_type': model_type,
                    'scenario': scenario,
                    'alpha': alpha,
                    'l1_ratio': l1_ratio,
                    'fold': fold_idx,
                    'non_zero_terms': nz,
                    **scores
                }
                fold_details.append(record)
                fold_metrics.append(scores)

            avg_rmse = float(np.mean([m['rmse'] for m in fold_metrics]))
            std_rmse = float(np.std([m['rmse'] for m in fold_metrics], ddof=1)) if n_splits > 1 else 0.0
            avg_mae = float(np.mean([m['mae'] for m in fold_metrics]))
            avg_mse = float(np.mean([m['mse'] for m in fold_metrics]))
            avg_r2 = float(np.mean([m['r2'] for m in fold_metrics]))
            avg_nz = float(np.mean(non_zero_counts))

            summary_records.append({
                'model_type': model_type,
                'scenario': scenario,
                'alpha': alpha,
                'l1_ratio': l1_ratio,
                'cv_rmse_mean': avg_rmse,
                'cv_rmse_std': std_rmse,
                'cv_mae_mean': avg_mae,
                'cv_mse_mean': avg_mse,
                'cv_r2_mean': avg_r2,
                'mean_non_zero_terms': avg_nz,
                'fold_coefs': fold_coefs
            })

    results_df = pd.DataFrame(summary_records)
    fold_df = pd.DataFrame(fold_details)
    return results_df, fold_df


def select_best_elastic_net(cv_results_df, tie_threshold=0.001):
    """Apply tie-break policy:
    Within tie_threshold (default 0.1% of min CV RMSE),
    pick the model with fewer non-zero terms (higher sparsity).
    If still tied, pick higher l1_ratio (more Lasso penalty).
    """
    min_rmse = cv_results_df['cv_rmse_mean'].min()
    cutoff = min_rmse * (1.0 + tie_threshold) if min_rmse > 0 else 0.0

    candidates = cv_results_df[cv_results_df['cv_rmse_mean'] <= cutoff].copy()
    # Sort by: mean_non_zero_terms ASC, l1_ratio DESC, cv_rmse_mean ASC
    sort_cols = ['mean_non_zero_terms', 'cv_rmse_mean']
    ascending = [True, True]
    if 'l1_ratio' in candidates.columns:
        sort_cols = ['mean_non_zero_terms', 'l1_ratio', 'cv_rmse_mean']
        ascending = [True, False, True]

    best = candidates.sort_values(by=sort_cols, ascending=ascending).iloc[0]
    return best.to_dict()


def fit_final_elastic_net(
    train_df,
    model_type='EN3',
    scenario='full',
    alpha=0.1,
    l1_ratio=0.5,
    include_pc_squared=False,
    random_state=42
):
    """Fit final model on 100% of the training dataset with optimal parameters."""
    X = train_df[INPUTS]
    y = train_df[TARGET]
    pipe = create_elastic_net_pipeline(
        model_type=model_type,
        scenario=scenario,
        alpha=alpha,
        l1_ratio=l1_ratio,
        include_pc_squared=include_pc_squared,
        random_state=random_state
    )
    pipe.fit(X, y)
    return pipe


def train_elastic_net_ablation(
    train_df,
    alphas=None,
    l1_ratios=None,
    n_splits=5,
    random_state=42
):
    """Train the Group Ablation variant excluding Anxiety_Level_During_Exams."""
    results_df, fold_df = tune_elastic_net_cv(
        train_df,
        model_type='EN3',
        scenario='no_anxiety',
        alphas=alphas,
        l1_ratios=l1_ratios,
        n_splits=n_splits,
        random_state=random_state
    )
    best_params = select_best_elastic_net(results_df)
    final_model = fit_final_elastic_net(
        train_df,
        model_type='EN3',
        scenario='no_anxiety',
        alpha=best_params['alpha'],
        l1_ratio=best_params['l1_ratio'],
        random_state=random_state
    )
    return results_df, final_model
