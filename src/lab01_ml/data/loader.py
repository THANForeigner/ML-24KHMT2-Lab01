"""Data loading and split integrity validation module."""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import sklearn

TARGET = 'Post_Semester_GPA'
EXCLUDED_COLUMNS = ['Student_ID', 'GPA_Change', 'Skill_Retention_Score', 'Burnout_Risk_Level']


def get_partition_path(root, partition_name):
    """Resolve partition path: checks data/splits/ first, then falls back to data/<partition>/dataset.csv."""
    root = Path(root)
    name_map = {
        'train': [root / 'data/splits/train.csv', root / 'data/train/dataset.csv'],
        'validation': [root / 'data/splits/val.csv', root / 'data/splits/validation.csv', root / 'data/validation/dataset.csv'],
        'val': [root / 'data/splits/val.csv', root / 'data/splits/validation.csv', root / 'data/validation/dataset.csv'],
        'test': [root / 'data/splits/test.csv', root / 'data/test/dataset.csv']
    }
    candidates = name_map.get(partition_name, [root / f'data/{partition_name}/dataset.csv', root / f'data/splits/{partition_name}.csv'])
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Partition '{partition_name}' not found under {root}. Checked: {[str(c) for c in candidates]}")


def load_frozen(root, include_test_target=False):
    """Load frozen splits with SHA-256 verification and holdout protection.
    Automatically falls back from data/splits/ to data/{train,validation,test}/dataset.csv.
    """
    root = Path(root)
    meta_path = root / 'data/metadata.json'
    if not meta_path.exists():
        raise FileNotFoundError(f'Missing metadata file: {meta_path}')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))

    assignment_path = root / 'data/assignments.csv'
    if not assignment_path.exists():
        raise FileNotFoundError(f'Missing assignment file: {assignment_path}')
    assignment_hash = hashlib.sha256(assignment_path.read_bytes()).hexdigest()
    if assignment_hash != meta['assignment_sha256']:
        raise ValueError('Assignment hash differs from data/metadata.json; do not overwrite the split')

    assignment = pd.read_csv(assignment_path, dtype={'Student_ID': str})
    if not assignment['Student_ID'].is_unique:
        raise ValueError('Duplicate Student_ID in data/assignments.csv')

    frames = {}
    seen_ids = set()
    partition_hashes = {}

    for name in ['train', 'validation', 'test']:
        found_path = get_partition_path(root, name)

        # Validate hash using metadata keys
        meta_key = f'{name}/dataset.csv'
        actual_hash = hashlib.sha256(found_path.read_bytes()).hexdigest()
        partition_hashes[name] = actual_hash
        if meta_key in meta.get('dataset_sha256', {}):
            if actual_hash != meta['dataset_sha256'][meta_key]:
                raise ValueError(f'Split hash mismatch: {found_path}; do not regenerate or overwrite')

        # Holdout guard: if test, do not read target unless explicitly approved
        usecols = None
        if name == 'test' and not include_test_target:
            usecols = ['Student_ID']

        frame = pd.read_csv(found_path, dtype={'Student_ID': str}, usecols=usecols)
        ids = frame['Student_ID']
        if not ids.is_unique:
            raise ValueError(f'Duplicate Student_ID in {found_path}')
        if seen_ids.intersection(ids):
            raise ValueError(f'Overlapping Student_ID detected between partitions in {found_path}')

        expected_ids = set(assignment.loc[assignment['split'] == name, 'Student_ID'])
        if set(ids) != expected_ids:
            raise ValueError(f'Assignment does not match partition: {found_path}')

        seen_ids.update(ids)
        frames[name] = frame

    provenance = {
        'source_sha256': meta.get('source_sha256'),
        'assignment_sha256': assignment_hash,
        'partition_sha256': partition_hashes,
        'counts': {k: len(v) for k, v in frames.items()},
        'group_approval': 'unconfirmed',
        'method': meta.get('method'),
        'cv_seed': 42,
        'versions': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'scikit_learn': sklearn.__version__
        }
    }

    return frames, provenance
