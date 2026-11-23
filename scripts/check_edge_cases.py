"""Targeted notebook robustness checks on temporary, clearly labelled CSV fixtures.

Uses actual CSV subsets with injected faults. Fixture outputs never become EDA evidence.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile

os.environ['MPLBACKEND'] = 'Agg'

import nbformat
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = nbformat.read(ROOT / 'notebooks' / 'kaggle' / '01_eda.ipynb', as_version=4)
CODES = [cell.source for cell in NOTEBOOK.cells if cell.cell_type == 'code']
SOURCE = next((ROOT / 'archive').glob('*.csv'))
BASE = pd.read_csv(SOURCE).head(600)


def run(frame, folder, full=True):
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / 'fixture.csv'
    frame.to_csv(csv_path, index=False)
    settings = {
        'LAB_DATA_PATH': str(csv_path),
        'LAB_OUTPUT_DIR': str(folder / 'reports'),
        'LAB_SPLIT_DIR': str(folder / 'splits'),
    }
    old = {key: os.environ.get(key) for key in settings}
    os.environ.update(settings)
    ns = {'__name__': '__fixture_check__'}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for index, source in enumerate(CODES):
                if not full and index > 3:
                    break
                exec(compile(source, f'fixture_cell_{index}', 'exec'), ns)
                if index == 0:
                    ns.update(SAVE_FIGURES=False, SAVE_TABLES=False, FIGURE_DPI=40, PLOT_SAMPLE_SIZE=200)
                if index == 1:
                    ns['display'] = lambda *args, **kwargs: None
                    ns['note'] = lambda *args, **kwargs: None
                    ns['plt'].show = lambda *args, **kwargs: None
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return ns


def main():
    results = []
    cache = ROOT / '.cache'
    cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='notebook-edge-cases-', dir=cache) as tmp:
        root = Path(tmp)

        # Minimal core schema must run all sections and explain optional skips.
        minimal = BASE[['Student_ID', 'Pre_Semester_GPA', 'Post_Semester_GPA']]
        ns = run(minimal, root / 'minimal')
        assert ns['run_metadata']['status'] == 'completed'
        assert ns['manifest']['T11']['status'] == 'skipped'
        assert ns['manifest']['F39']['status'] == 'skipped'
        assert len(ns['models']) == 3
        results.append({'check': 'optional columns absent + all save flags off', 'status': 'PASS'})

        # Inject exact duplicate, connected ID/content groups, missing target, invalid numeric,
        # whitespace/case variants, and an unknown ordinal level.
        faulty = BASE.copy()
        faulty.loc[1, 'Student_ID'] = faulty.loc[0, 'Student_ID']
        faulty.loc[2, faulty.columns != 'Student_ID'] = faulty.loc[3, faulty.columns != 'Student_ID'].to_numpy()
        faulty.loc[4, 'Post_Semester_GPA'] = np.nan
        faulty['Weekly_GenAI_Hours'] = faulty['Weekly_GenAI_Hours'].astype(object)
        faulty.loc[5, 'Weekly_GenAI_Hours'] = 'not-numeric'
        faulty.loc[6, 'Weekly_GenAI_Hours'] = -5
        faulty.loc[7, 'Major_Category'] = ' stem '
        faulty.loc[8, 'Prompt_Engineering_Skill'] = 'Unverified_Level'
        faulty = pd.concat([faulty, faulty.iloc[[10]]], ignore_index=True)
        ns = run(faulty, root / 'quality', full=False)
        assignment = ns['assignments']
        assert assignment.loc[0, 'split'] == assignment.loc[1, 'split']
        assert assignment.loc[2, 'split'] == assignment.loc[3, 'split']
        assert assignment.loc[4, 'exclusion_reason'] == 'missing_or_invalid_target'
        assert assignment.loc[len(faulty)-1, 'exclusion_reason'] == 'exact_duplicate'
        assert pd.isna(ns['clean'].loc[5, 'Weekly_GenAI_Hours'])
        assert pd.isna(ns['clean'].loc[6, 'Weekly_GenAI_Hours'])
        assert ns['clean'].loc[7, 'Major_Category'] == 'STEM'
        assert ns['clean'].loc[8, 'Prompt_Engineering_Skill'] == 'Unverified_Level'
        results.append({'check': 'fault logging, exclusions, transitive groups, no ordinal information loss', 'status': 'PASS'})

        # R²/OLS is not meaningful with a constant target; fail with an actionable reason.
        constant = minimal.copy()
        constant['Post_Semester_GPA'] = 3.0
        try:
            run(constant, root / 'constant', full=False)
        except ValueError as error:
            assert 'target is constant' in str(error)
        else:
            raise AssertionError('Constant target was not rejected')
        results.append({'check': 'constant target fails clearly', 'status': 'PASS'})

        # A mathematical cross-check of the vectorized VIF formula against statsmodels.
        ns = run(BASE, root / 'complete')
        design = ns['X']
        expected = np.array([variance_inflation_factor(design, i) for i in range(1, design.shape[1])])
        np.testing.assert_allclose(ns['vif_table'].VIF.to_numpy(), expected, rtol=1e-8, atol=1e-10)
        for record in ns['bin_log']:
            edges = np.array(json.loads(record['edges']))
            assert np.all(np.diff(edges) > 0)
        results.append({'check': 'Gram VIF matches statsmodels; quantile edges strictly ordered', 'status': 'PASS'})

        # Test-target permutation must leave train and validation statistics/metrics identical.
        # Row keys change with the CSV fingerprint; source positions and actual analytic samples must not.
        test_positions = ns['assignments'].query("split == 'test'").source_row_position.to_numpy()
        poisoned = BASE.copy()
        poisoned.loc[test_positions, 'Post_Semester_GPA'] = np.roll(
            poisoned.loc[test_positions, 'Post_Semester_GPA'].to_numpy(), 1)
        changed = run(poisoned, root / 'test_target_permutation')
        pd.testing.assert_series_equal(ns['assignments'].split, changed['assignments'].split)
        for aid in ['T10', 'T15', 'T19', 'T20', 'T28', 'T29', 'T35']:
            pd.testing.assert_frame_equal(ns['tables'][aid], changed['tables'][aid], check_exact=False, rtol=1e-10, atol=1e-12)
        results.append({'check': 'test-target permutation cannot change train EDA or validation metrics', 'status': 'PASS'})

    destination = ROOT / 'reports' / 'edge_case_verification.json'
    destination.write_text(json.dumps({'fixture_policy': 'Temporary actual-data subsets with injected faults; never EDA evidence',
                                       'checks': results}, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
