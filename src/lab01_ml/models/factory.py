"""Pipeline factory for Elastic Net and Baseline Regression models."""
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge, Lasso

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


def create_polynomial_pipeline(
    degree=4,
    model_type='Ridge',
    alpha=50.0,
    use_group_d=True,
    random_state=42
):
    """Create fully encapsulated Polynomial Linear Regression pipeline.

    Input features (raw):
    - Numerical (4): Pre_Semester_GPA, Traditional_Study_Hours, Weekly_GenAI_Hours, Tool_Diversity
    - Categorical ABC (5): Year_of_Study, Major_Category, Primary_Use_Case, Prompt_Engineering_Skill, Institutional_Policy
    - Categorical D (2, optional): Perceived_AI_Dependency, Anxiety_Level_During_Exams
    - Binary (1): Paid_Subscription

    Stages:
    1. Numerical: StandardScaler -> PolynomialFeatures -> StandardScaler
    2. Categorical: OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    3. Binary: Passthrough
    4. Estimator: Ridge / Lasso / OLS
    """
    num_cols = ['Pre_Semester_GPA', 'Traditional_Study_Hours', 'Weekly_GenAI_Hours', 'Tool_Diversity']
    cat_cols = ['Year_of_Study', 'Major_Category', 'Primary_Use_Case', 'Prompt_Engineering_Skill', 'Institutional_Policy']
    if use_group_d:
        cat_cols = cat_cols + ['Perceived_AI_Dependency', 'Anxiety_Level_During_Exams']
    bin_cols = ['Paid_Subscription']

    if degree > 1:
        num_pipe = Pipeline([
            ('scaler', StandardScaler(with_mean=True, with_std=True)),
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('poly_scaler', StandardScaler(with_mean=True, with_std=True))
        ])
    else:
        num_pipe = Pipeline([
            ('scaler', StandardScaler(with_mean=True, with_std=True))
        ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipe, num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols),
            ('bin', 'passthrough', bin_cols)
        ]
    )

    if model_type == 'Ridge':
        regressor = Ridge(alpha=alpha, random_state=random_state)
    elif model_type == 'Lasso':
        regressor = Lasso(alpha=alpha, random_state=random_state, max_iter=2000, tol=1e-3)
    else:
        regressor = LinearRegression()

    return Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])

