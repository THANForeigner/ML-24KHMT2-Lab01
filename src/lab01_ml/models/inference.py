"""Raw-profile prediction with schema errors and user-visible warnings."""
import warnings
import pandas as pd
import numpy as np

def predict_student(raw_features, model):
    if not isinstance(raw_features, dict):
        raise ValueError('raw_features must be a dictionary')
    transformer = model.named_steps['features']
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
    return {'prediction': prediction, 'model': f'I{transformer.model}',
            'scenario': transformer.scenario, 'warnings': messages}
