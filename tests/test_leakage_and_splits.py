import unittest
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

class SplitTests(unittest.TestCase):
    def test_frozen_ids_disjoint_and_match_assignment(self):
        assignment = pd.read_csv(ROOT / 'data/assignments.csv', dtype={'Student_ID': str})
        self.assertTrue(assignment.Student_ID.is_unique)
        groups = {}
        for name, filename, count in [('train', 'train.csv', 35000), ('validation', 'val.csv', 7500), ('test', 'test.csv', 7500)]:
            frame = pd.read_csv(ROOT / 'data/splits' / filename, dtype={'Student_ID': str})
            groups[name] = set(frame.Student_ID)
            self.assertEqual(len(frame), count)
            self.assertEqual(groups[name], set(assignment.loc[assignment['split'] == name, 'Student_ID']))
        self.assertFalse(groups['train'] & groups['validation'])
        self.assertFalse(groups['train'] & groups['test'])
        self.assertFalse(groups['validation'] & groups['test'])

if __name__ == '__main__':
    unittest.main()
