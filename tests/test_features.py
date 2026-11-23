"""Checks for the interaction transform using training rows only."""
import unittest
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lab01_ml.features.engineering import InteractionFeatures, INPUTS


class InteractionTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.read_csv(Path(__file__).resolve().parents[1] / 'data/splits/train.csv')

    def test_counts_and_hierarchy(self):
        for model, count in enumerate((39, 41, 45, 53, 55)):
            transform = InteractionFeatures(model=model).fit(self.train[INPUTS])
            self.assertEqual(len(transform.get_feature_names_out()), count)
            self.assertEqual(transform.transform(self.train[INPUTS].iloc[:1]).shape, (1, count))

    def test_centering_is_train_only(self):
        X = self.train[INPUTS].iloc[:30].copy()
        t = InteractionFeatures(model=4).fit(X)
        names = list(t.get_feature_names_out())
        values = t.transform(X)
        self.assertAlmostEqual(values[:, names.index('Hc')].mean(), 0.0, places=10)
        self.assertAlmostEqual(t.means_['Weekly_GenAI_Hours'], X['Weekly_GenAI_Hours'].mean())
        self.assertTrue(np.isfinite(values).all())

    def test_no_d_and_leak_exclusion(self):
        t = InteractionFeatures(model=3, scenario='no_d').fit(self.train)
        self.assertFalse(any('Anxiety' in n or 'Dependency' in n or 'GPA_Change' in n for n in t.get_feature_names_out()))

if __name__ == '__main__':
    unittest.main()
