"""Unit tests for Elastic Net feature engineering and hierarchy."""
import unittest
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from lab01_ml.data.loader import get_partition_path
from lab01_ml.features.elastic_net_features import ElasticNetFeatures, INPUTS, H, P, S


class ElasticNetFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.train = pd.read_csv(get_partition_path(root, 'train'))

    def test_en1_terms_count(self):
        feat = ElasticNetFeatures(model_type='EN1').fit(self.train[INPUTS])
        names = feat.get_feature_names_out()
        self.assertEqual(len(names), 39, f"EN1 should produce 39 terms, got {len(names)}")
        transformed = feat.transform(self.train[INPUTS].iloc[:5])
        self.assertEqual(transformed.shape, (5, 39))

    def test_en2_terms_count_and_no_raw_h(self):
        feat = ElasticNetFeatures(model_type='EN2').fit(self.train[INPUTS])
        names = list(feat.get_feature_names_out())
        self.assertEqual(len(names), 40, f"EN2 should produce 40 terms, got {len(names)}")
        self.assertIn('Hc', names)
        self.assertIn('Hc^2', names)
        # Raw H must NOT be present when Hc is present with intercept
        self.assertNotIn('Weekly_GenAI_Hours', names)
        transformed = feat.transform(self.train[INPUTS].iloc[:5])
        self.assertEqual(transformed.shape, (5, 40))

    def test_en3_terms_count_and_interactions(self):
        feat = ElasticNetFeatures(model_type='EN3').fit(self.train[INPUTS])
        names = list(feat.get_feature_names_out())
        self.assertEqual(len(names), 54, f"EN3 should produce 54 terms, got {len(names)}")
        # Check interaction terms presence
        self.assertTrue(any('Hc * Primary_Use_Case=' in n for n in names))
        self.assertTrue(any('Hc * Prompt_Engineering_Skill=' in n for n in names))
        self.assertTrue(any('Primary_Use_Case=' in n and '* Prompt_Engineering_Skill=' in n for n in names))
        transformed = feat.transform(self.train[INPUTS].iloc[:5])
        self.assertEqual(transformed.shape, (5, 54))

    def test_en3_no_anxiety_ablation_count(self):
        feat = ElasticNetFeatures(model_type='EN3', scenario='no_anxiety').fit(self.train[INPUTS])
        names = list(feat.get_feature_names_out())
        # 54 terms - 9 dummies for Anxiety = 45 terms
        self.assertEqual(len(names), 45, f"EN3 no-anxiety should produce 45 terms, got {len(names)}")
        self.assertFalse(any('Anxiety' in n for n in names))
        transformed = feat.transform(self.train[INPUTS].iloc[:5])
        self.assertEqual(transformed.shape, (5, 45))

    def test_unknown_category_warning_and_zeros(self):
        feat = ElasticNetFeatures(model_type='EN1').fit(self.train[INPUTS].iloc[:100])
        test_row = self.train[INPUTS].iloc[[0]].copy()
        test_row['Major_Category'] = 'Alien_Major_Unknown'
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            transformed = feat.transform(test_row)
        self.assertTrue(any('unknown categories' in str(w.message) for w in captured))
        # Ensure finite values
        self.assertTrue(np.isfinite(transformed).all())

    def test_centering_is_train_only(self):
        sample = self.train[INPUTS].iloc[:50].copy()
        feat = ElasticNetFeatures(model_type='EN3').fit(sample)
        names = list(feat.get_feature_names_out())
        transformed = feat.transform(sample)
        hc_idx = names.index('Hc')
        # Mean of centered H on the training sample should be ~0.0
        self.assertAlmostEqual(transformed[:, hc_idx].mean(), 0.0, places=10)
        self.assertAlmostEqual(feat.means_['Weekly_GenAI_Hours'], sample['Weekly_GenAI_Hours'].mean())


if __name__ == '__main__':
    unittest.main()
