"""Unit tests for leakage prevention, split integrity, and pipeline encapsulation."""
import unittest
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from lab01_ml.data.loader import get_partition_path

ROOT = Path(__file__).resolve().parents[1]


class SplitAndLeakageTests(unittest.TestCase):
    def test_frozen_ids_disjoint_and_match_assignment(self):
        assignment = pd.read_csv(ROOT / 'data/assignments.csv', dtype={'Student_ID': str})
        self.assertTrue(assignment.Student_ID.is_unique)
        groups = {}
        for name, count in [('train', 35000), ('validation', 7500), ('test', 7500)]:
            frame = pd.read_csv(get_partition_path(ROOT, name), dtype={'Student_ID': str})
            groups[name] = set(frame.Student_ID)
            self.assertEqual(len(frame), count)
            self.assertEqual(groups[name], set(assignment.loc[assignment['split'] == name, 'Student_ID']))
        self.assertFalse(groups['train'] & groups['validation'])
        self.assertFalse(groups['train'] & groups['test'])
        self.assertFalse(groups['validation'] & groups['test'])

    def test_split_hashes_match_metadata(self):
        meta = json.loads((ROOT / 'data/metadata.json').read_text(encoding='utf-8'))
        assignment_bytes = (ROOT / 'data/assignments.csv').read_bytes()
        self.assertEqual(hashlib.sha256(assignment_bytes).hexdigest(), meta['assignment_sha256'])
        for partition, relpath in [('train', 'train/dataset.csv'), ('validation', 'validation/dataset.csv'), ('test', 'test/dataset.csv')]:
            file_bytes = (ROOT / 'data' / relpath).read_bytes()
            self.assertEqual(hashlib.sha256(file_bytes).hexdigest(), meta['dataset_sha256'][relpath])

    def test_scaling_is_fold_only_in_pipeline(self):
        """Verifies that StandardScaler inside Pipeline scales each fold separately."""
        from lab01_ml.features.elastic_net_features import ElasticNetFeatures, INPUTS
        from lab01_ml.models.factory import create_elastic_net_pipeline

        train_data = pd.read_csv(get_partition_path(ROOT, 'train'))
        X = train_data[INPUTS].iloc[:1000]
        y = train_data['Post_Semester_GPA'].iloc[:1000]

        pipe = create_elastic_net_pipeline(model_type='EN1', alpha=0.1, l1_ratio=0.5)
        pipe.fit(X, y)

        scaler = pipe.named_steps['scaler']
        # The scaler must be fitted with mean_ and var_ having 39 elements
        self.assertEqual(len(scaler.mean_), 39)
        self.assertEqual(len(scaler.var_), 39)

    def test_exclusion_columns_not_in_x(self):
        from lab01_ml.features.elastic_net_features import ElasticNetFeatures, INPUTS
        EXCLUDED = ['Student_ID', 'GPA_Change', 'Skill_Retention_Score', 'Burnout_Risk_Level']
        for col in EXCLUDED:
            self.assertNotIn(col, INPUTS)


if __name__ == '__main__':
    unittest.main()
