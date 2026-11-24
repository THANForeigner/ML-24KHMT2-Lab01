"""Unit tests for Elastic Net 5-fold CV tuning, tie-break policy, and ablation."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from lab01_ml.data.loader import get_partition_path
from lab01_ml.features.elastic_net_features import INPUTS, TARGET
from lab01_ml.models.elastic_net_training import (
    tune_elastic_net_cv,
    select_best_elastic_net,
    train_elastic_net_ablation
)

ROOT = Path(__file__).resolve().parents[1]


class ElasticNetTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = pd.read_csv(get_partition_path(ROOT, 'train')).iloc[:1000]
        cls.val = pd.read_csv(get_partition_path(ROOT, 'val')).iloc[:50]

    def test_cv_tuning_produces_valid_metrics(self):
        alphas = [0.01, 0.1]
        l1_ratios = [0.5, 0.7]
        results_df, fold_records = tune_elastic_net_cv(
            self.train,
            model_type='EN1',
            alphas=alphas,
            l1_ratios=l1_ratios,
            n_splits=3,
            random_state=42
        )
        self.assertEqual(len(results_df), len(alphas) * len(l1_ratios))
        for col in ['cv_rmse_mean', 'cv_rmse_std', 'cv_mae_mean', 'cv_r2_mean', 'mean_non_zero_terms']:
            self.assertIn(col, results_df.columns)
            self.assertTrue(np.isfinite(results_df[col]).all())

    def test_tie_break_prefers_sparser_model(self):
        # Create dummy df where model A and model B are within 0.1% RMSE, but B has fewer non-zero terms
        dummy = pd.DataFrame([
            {'model_type': 'EN3', 'alpha': 0.01, 'l1_ratio': 0.5, 'cv_rmse_mean': 0.1000, 'mean_non_zero_terms': 45},
            {'model_type': 'EN3', 'alpha': 0.05, 'l1_ratio': 0.7, 'cv_rmse_mean': 0.10005, 'mean_non_zero_terms': 28}, # within 0.1% (<= 0.1001)
        ])
        best = select_best_elastic_net(dummy, tie_threshold=0.001)
        # Should pick the sparser model (28 terms)
        self.assertEqual(best['alpha'], 0.05)
        self.assertEqual(best['mean_non_zero_terms'], 28)

    def test_ablation_no_anxiety_training(self):
        alphas = [0.1]
        l1_ratios = [0.7]
        results_df, model = train_elastic_net_ablation(
            self.train,
            alphas=alphas,
            l1_ratios=l1_ratios,
            n_splits=3,
            random_state=42
        )
        self.assertIn('cv_rmse_mean', results_df.columns)
        self.assertEqual(len(model.named_steps['features'].get_feature_names_out()), 45)


if __name__ == '__main__':
    unittest.main()
