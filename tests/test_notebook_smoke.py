"""Smoke tests for notebook validation and consistency."""
import unittest
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]


class NotebookSmokeTests(unittest.TestCase):
    def test_elastic_net_notebook_format_and_structure(self):
        path = ROOT / 'notebooks/elastic_net_regression.ipynb'
        self.assertTrue(path.exists(), f"Notebook not found at {path}")
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)

        # Check section count
        markdown_cells = [c for c in nb.cells if c.cell_type == 'markdown']
        code_cells = [c for c in nb.cells if c.cell_type == 'code']
        self.assertGreaterEqual(len(markdown_cells), 30)
        self.assertGreaterEqual(len(code_cells), 20)

        # Check key titles present
        all_text = ' '.join(c.source for c in markdown_cells)
        self.assertIn('Elastic Net', all_text)
        self.assertIn('EN1', all_text)
        self.assertIn('EN2', all_text)
        self.assertIn('EN3', all_text)
        self.assertIn('StandardScaler', all_text)
        self.assertIn('Ablation', all_text)
        self.assertIn('What-if', all_text)


if __name__ == '__main__':
    unittest.main()
