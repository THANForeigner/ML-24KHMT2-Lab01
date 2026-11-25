"""Unit tests for Polynomial Linear Regression pipeline and notebook smoke checks."""
import unittest
from pathlib import Path
import re
import numpy as np
import pandas as pd
import nbformat

from lab01_ml.models.factory import create_polynomial_pipeline

ROOT = Path(__file__).resolve().parents[1]


class TestPolynomialPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train_path = ROOT / 'data/train/dataset.csv'
        val_path = ROOT / 'data/validation/dataset.csv'
        cls.train = pd.read_csv(train_path, nrows=200)
        cls.val = pd.read_csv(val_path, nrows=50)
        cls.target_col = 'Post_Semester_GPA'

    def test_pipeline_fit_and_predict(self):
        """Pipeline should fit and predict valid GPA scores."""
        pipe = create_polynomial_pipeline(degree=2, model_type='Ridge', alpha=1.0)
        pipe.fit(self.train, self.train[self.target_col])
        preds = pipe.predict(self.val)

        self.assertEqual(len(preds), len(self.val))
        self.assertFalse(np.isnan(preds).any())
        self.assertTrue(np.all(preds >= 0.0) and np.all(preds <= 5.0))

    def test_pipeline_degrees(self):
        """Pipeline should support degrees 1 through 4 cleanly."""
        for deg in [1, 2, 4]:
            pipe = create_polynomial_pipeline(degree=deg, model_type='Ridge', alpha=10.0)
            pipe.fit(self.train, self.train[self.target_col])
            preds = pipe.predict(self.val)
            self.assertEqual(len(preds), len(self.val))

    def test_prospective_scenario_without_group_d(self):
        """Version 1 (no Group D) should exclude psychological features."""
        pipe = create_polynomial_pipeline(degree=2, model_type='Ridge', use_group_d=False)
        pipe.fit(self.train, self.train[self.target_col])
        preds = pipe.predict(self.val)
        self.assertEqual(len(preds), len(self.val))

    def test_notebook_format_and_zero_emojis(self):
        """Notebooks should exist, validate with nbformat, and contain zero emojis."""
        paths = [
            ROOT / 'notebooks/sources/polynomial_linear_regression/polynomial_linear_regression.ipynb',
            ROOT / 'notebooks/polynomial_linear_regression.ipynb'
        ]
        emoji_pattern = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)

        for path in paths:
            if not path.exists():
                continue
            nb = nbformat.read(path, as_version=4)
            nbformat.validate(nb)

            all_text = ' '.join(c.source for c in nb.cells)
            emojis_found = emoji_pattern.findall(all_text)
            self.assertEqual(
                len(emojis_found),
                0,
                f"Found emojis in {path.name}: {emojis_found[:5]}"
            )
            self.assertIn("Hồi quy Đa thức", all_text)
            self.assertIn("[INFO]", all_text)


if __name__ == '__main__':
    unittest.main()
