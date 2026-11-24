import unittest
from pathlib import Path
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lab01_ml.data.loader import get_partition_path
from lab01_ml.features.engineering import INPUTS
from lab01_ml.models.training import pipeline, metrics
from lab01_ml.models.inference import predict_student

class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.frame = pd.read_csv(get_partition_path(root, 'train')).head(1000)
        cls.model = pipeline(3).fit(cls.frame[INPUTS], cls.frame['Post_Semester_GPA'])
        cls.profile = cls.frame.iloc[0][INPUTS].to_dict()

    def test_prediction_and_required_columns(self):
        self.assertTrue(np.isfinite(predict_student(self.profile, self.model)['prediction']))
        with self.assertRaisesRegex(ValueError, 'Missing raw inputs'):
            predict_student({}, self.model)
        with self.assertRaisesRegex(ValueError, 'finite number'):
            predict_student(self.profile | {'Weekly_GenAI_Hours': 'abc'}, self.model)

    def test_unknown_category_warned(self):
        result = predict_student(self.profile | {'Primary_Use_Case': 'unknown_example'}, self.model)
        self.assertTrue(any('unknown' in warning.lower() for warning in result['warnings']))

    def test_metric_values(self):
        result = metrics([1, 2, 3], [1, 2, 2])
        self.assertAlmostEqual(result['mae'], 1/3)
        self.assertAlmostEqual(result['mse'], 1/3)
        self.assertAlmostEqual(result['rmse'], np.sqrt(1/3))

if __name__ == '__main__':
    unittest.main()
