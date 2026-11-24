"""Raw-profile prediction with schema errors, user-visible warnings, and model metadata."""
import warnings
import pandas as pd
import numpy as np


def predict_student(raw_features, model):
    """Predict student GPA from raw features dictionary.

    Args:
        raw_features: dict with 12 raw input feature values.
        model: fitted Pipeline estimator.

    Returns:
        dict: prediction, model identifier, scenario, and warnings.
    """
    if not isinstance(raw_features, dict):
        raise ValueError('raw_features must be a dictionary')

    transformer = model.named_steps.get('features')
    if transformer is None:
        raise ValueError('Model pipeline must contain a step named "features"')

    missing = [name for name in transformer.required_ if name not in raw_features or pd.isna(raw_features[name])]
    if missing:
        raise ValueError('Missing raw inputs: ' + ', '.join(missing))

    for name in ['Weekly_GenAI_Hours', 'Pre_Semester_GPA', 'Traditional_Study_Hours', 'Tool_Diversity']:
        try:
            value = float(raw_features[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{name} must be a finite number') from exc
        if not np.isfinite(value):
            raise ValueError(f'{name} must be a finite number')

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        prediction = float(model.predict(pd.DataFrame([raw_features]))[0])

    messages = [str(w.message) for w in captured]

    for name in ['Weekly_GenAI_Hours', 'Traditional_Study_Hours', 'Tool_Diversity']:
        if float(raw_features[name]) < 0:
            messages.append(f'{name} is below zero; input may be outside the valid domain.')

    if not 0 <= float(raw_features['Pre_Semester_GPA']) <= 4:
        messages.append('Pre_Semester_GPA is outside [0, 4].')

    if not 0 <= prediction <= 4:
        messages.append('Prediction outside GPA range [0, 4]; it has not been clipped.')

    model_name = getattr(transformer, 'model_type', f'I{getattr(transformer, "model", "Unknown")}')
    result = {
        'prediction': prediction,
        'model': model_name,
        'scenario': getattr(transformer, 'scenario', 'full'),
        'warnings': messages
    }

    # Add hyperparameter info if available
    regressor = model.named_steps.get('regressor', model.named_steps.get('elasticnet'))
    if regressor is not None and hasattr(regressor, 'alpha'):
        result['alpha'] = float(regressor.alpha)
        result['l1_ratio'] = float(regressor.l1_ratio)

    return result
