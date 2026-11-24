"""Pipeline factory for Elastic Net and Baseline Regression models."""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, LinearRegression

from lab01_ml.features.elastic_net_features import ElasticNetFeatures
from lab01_ml.features.engineering import InteractionFeatures


def create_elastic_net_pipeline(
    model_type='EN3',
    scenario='full',
    alpha=0.1,
    l1_ratio=0.5,
    include_pc_squared=False,
    random_state=42,
    max_iter=10000,
    tol=1e-4
):
    """Create a fully encapsulated 3-step Elastic Net pipeline:
    1. ElasticNetFeatures: Imputation, One-Hot drop-first, centering, interactions
    2. StandardScaler: Fit within each fold for fair regularization
    3. ElasticNet: Coordinate descent solver
    """
    return Pipeline([
        ('features', ElasticNetFeatures(
            model_type=model_type,
            scenario=scenario,
            include_pc_squared=include_pc_squared
        )),
        ('scaler', StandardScaler(with_mean=True, with_std=True)),
        ('regressor', ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=random_state,
            max_iter=max_iter,
            tol=tol
        ))
    ])


def create_ols_pipeline(model=3, scenario='full'):
    """Create baseline OLS pipeline matching branch 001."""
    return Pipeline([
        ('features', InteractionFeatures(model=model, scenario=scenario)),
        ('ols', LinearRegression(fit_intercept=True))
    ])
