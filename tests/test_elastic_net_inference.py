"""Unit tests for Elastic Net inference and pipeline serialization."""
import unittest
from pathlib import Path
import tempfile
import joblib
import pandas as pd
import numpy as np

from lab01_ml.data.loader import get_partition_path
from lab01_ml.features.elastic_net_features import INPUTS, TARGET
from lab01_ml.models.factory import create_elastic_net_pipeline
from lab01_ml.models.inference import predict_student

ROOT = Path(__file__).resolve().parents[1]


class ElasticNetInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train = pd.read_csv(get_partition_path(ROOT, 'train')).iloc[:100]
        cls.pipe = create_elastic_net_pipeline(model_type='EN3', alpha=0.01, l1_ratio=0.5)
        cls.pipe.fit(train[INPUTS], train[TARGET])
        cls.sample = train.iloc[0][INPUTS].to_dict()

    def test_predict_student_success(self):
        result = predict_student(self.sample, self.pipe)
        self.assertIn('prediction', result)
        self.assertIsInstance(result['prediction'], float)
        self.assertIn('model', result)
        self.assertIn('scenario', result)
        self.assertIn('warnings', result)

    def test_predict_student_missing_input_raises(self):
        bad_sample = self.sample.copy()
        del bad_sample['Weekly_GenAI_Hours']
        with self.assertRaises(ValueError):
            predict_student(bad_sample, self.pipe)

    def test_predict_student_invalid_number_raises(self):
        bad_sample = self.sample.copy()
        bad_sample['Weekly_GenAI_Hours'] = 'not_a_number'
        with self.assertRaises(ValueError):
            predict_student(bad_sample, self.pipe)

    def test_predict_student_domain_warning(self):
        out_domain = self.sample.copy()
        out_domain['Weekly_GenAI_Hours'] = -5.0
        result = predict_student(out_domain, self.pipe)
        self.assertTrue(any('below zero' in w for w in result['warnings']))

    def test_joblib_reload_and_predict(self):
        with tempfile.NamedTemporaryFile(suffix='.joblib') as tmp:
            joblib.dump(self.pipe, tmp.name)
            reloaded = joblib.load(tmp.name)
            res1 = predict_student(self.sample, self.pipe)
            res2 = predict_student(self.sample, reloaded)
            self.assertAlmostEqual(res1['prediction'], res2['prediction'], places=9)


if __name__ == '__main__':
    unittest.main()
