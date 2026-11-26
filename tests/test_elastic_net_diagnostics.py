"""Unit tests for Elastic Net specific diagnostics: sparsity, selection frequency, hierarchy, and ceiling checks."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from lab01_ml.data.loader import get_partition_path
from lab01_ml.features.elastic_net_features import INPUTS, TARGET
from lab01_ml.models.factory import create_elastic_net_pipeline
from lab01_ml.models.elastic_net_evaluation import (
    extract_sparsity_metrics,
    calculate_selection_frequency,
    check_hierarchy,
    check_gpa_bounds
)

ROOT = Path(__file__).resolve().parents[1]


class ElasticNetDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = pd.read_csv(get_partition_path(ROOT, 'train')).iloc[:1000]
        X = cls.train[INPUTS]
        y = cls.train[TARGET]
        cls.pipe = create_elastic_net_pipeline(model_type='EN3', alpha=0.05, l1_ratio=0.7)
        cls.pipe.fit(X, y)

    def test_sparsity_metrics(self):
        k, sparsity_pct, coef_df = extract_sparsity_metrics(self.pipe)
        self.assertIsInstance(k, int)
        self.assertTrue(0 <= k <= 54)
        self.assertTrue(0.0 <= sparsity_pct <= 100.0)
        self.assertEqual(len(coef_df), 54)
        self.assertIn('coefficient', coef_df.columns)
        self.assertIn('is_non_zero', coef_df.columns)

    def test_selection_frequency(self):
        # Mock fold coefficients dictionary
        feature_names = ['f1', 'f2', 'f3']
        fold_coefs = [
            np.array([1.0, 0.0, 0.5]),
            np.array([0.8, 0.0, 0.0]),
            np.array([0.9, 0.0, 0.2]),
            np.array([1.1, 0.0, 0.3]),
            np.array([0.95, 0.0, 0.4]),
        ]
        freq_df = calculate_selection_frequency(feature_names, fold_coefs)
        self.assertEqual(len(freq_df), 3)
        self.assertEqual(freq_df.loc[freq_df['feature'] == 'f1', 'frequency'].iloc[0], 1.0) # 5/5
        self.assertEqual(freq_df.loc[freq_df['feature'] == 'f2', 'frequency'].iloc[0], 0.0) # 0/5
        self.assertEqual(freq_df.loc[freq_df['feature'] == 'f3', 'frequency'].iloc[0], 0.8) # 4/5

    def test_hierarchy_check(self):
        feature_names = ['Hc', 'Primary_Use_Case=Research', 'Hc * Primary_Use_Case=Research']
        # Case: interaction is non-zero, but main effect Hc is zero
        coefs = np.array([0.0, 0.5, 0.3])
        violations = check_hierarchy(feature_names, coefs)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]['interaction'], 'Hc * Primary_Use_Case=Research')
        self.assertIn('Hc', violations[0]['missing_main_effects'])

    def test_gpa_bounds_no_clipping(self):
        preds = np.array([3.5, 4.1, 2.8, -0.2, 3.9, 4.05])
        bounds_info = check_gpa_bounds(preds)
        self.assertEqual(bounds_info['below_zero_count'], 1)
        self.assertEqual(bounds_info['above_four_count'], 2)
        self.assertAlmostEqual(bounds_info['out_of_bounds_pct'], 50.0)

    def test_what_if_curves_use_valid_categories_without_warnings(self):
        import warnings
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from lab01_ml.visualization.plots import plot_what_if_curves

        profile = self.train[INPUTS].iloc[0].to_dict()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            fig, axes = plot_what_if_curves(self.pipe, profile, ai_hours_grid=np.linspace(0, 40, 10))
            plt.close(fig)

        unknown_warnings = [w for w in captured if 'unknown categories' in str(w.message).lower()]
        self.assertEqual(len(unknown_warnings), 0, f'Unexpected unknown category warnings: {unknown_warnings}')
        self.assertEqual(len(axes), 3)

        # Invalid category should raise ValueError explicitly
        with self.assertRaises(ValueError):
            plot_what_if_curves(self.pipe, profile, use_cases=['Invalid_Use_Case'])


if __name__ == '__main__':
    unittest.main()
