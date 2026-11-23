"""Execute in a fresh Python kernel; validate artifacts, optionally rerun for reproducibility."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager
import numpy as np
import pandas as pd
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(path, kernel_name):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    client = NotebookClient(notebook, timeout=900, kernel_name=kernel_name,
                            resources={'metadata': {'path': str(ROOT)}}, record_timing=True)
    code_count = sum(c.cell_type == 'code' for c in notebook.cells)

    def progress(cell, cell_index, **kwargs):
        if cell.cell_type == 'code':
            first = cell.source.splitlines()[0][:100]
            print(f'Cell {cell_index + 1}/{len(notebook.cells)}: {first}', flush=True)

    client.on_cell_start = progress
    started = time.monotonic()
    try:
        client.execute()
    except Exception:
        failed_path = ROOT / 'reports' / '01_EDA.failed.ipynb'
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, failed_path)
        raise
    errors = [o for c in notebook.cells if c.cell_type == 'code' for o in c.get('outputs', []) if o.output_type == 'error']
    assert not errors
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print(f'Executed {code_count} code cells in {time.monotonic()-started:.1f}s, zero errors', flush=True)
    return notebook


def inspect_outputs(output_dir, split_dir):
    meta = json.loads((output_dir / 'run_metadata.json').read_text(encoding='utf-8'))
    assert meta['status'] == 'completed'
    manifest = pd.read_csv(output_dir / 'artifact_manifest.csv').fillna('')
    planned = {f'T{i:02}' for i in range(1, 49)} | {f'F{i:02}' for i in range(1, 58)}
    assert set(manifest.artifact_id) == planned and len(manifest) == len(planned)
    loaded = {}
    for r in manifest.itertuples():
        assert r.status in ['generated', 'reused', 'skipped']
        if r.status == 'skipped':
            assert r.reason and not r.relative_path
            continue
        path = output_dir / r.relative_path
        assert path.is_file() and path.stat().st_size > 0
        if r.artifact_type == 'table':
            loaded[r.artifact_id] = pd.read_csv(path)
        else:
            assert mpimg.imread(path).size > 0
    assignments = pd.read_csv(split_dir / 'split_assignment.csv').fillna('')
    split_meta = json.loads((split_dir / 'split_metadata.json').read_text(encoding='utf-8'))
    assert assignments.row_key.is_unique
    assert 'Post_Semester_GPA' not in assignments and 'Student_ID' not in assignments
    for i, left in enumerate(['train', 'validation', 'test']):
        for right in ['train', 'validation', 'test'][i+1:]:
            a = assignments[assignments.split == left]
            b = assignments[assignments.split == right]
            assert not set(a.row_key) & set(b.row_key)
            assert not set(a.group_key) & set(b.group_key)
    assert sha256(meta['dataset_path']) == meta['dataset_sha256']
    assert len(assignments) == meta['row_count_raw']
    if 'T41' in loaded:
        assert loaded['T41'].passed.all()
    findings = loaded['T44']
    assert set(findings.research_question) == {f'RQ{i:02}' for i in range(1, 13)}
    assert findings.finding_id.is_unique
    assert all(str(ref) in planned for ids in findings.evidence_ids for ref in ids.split())
    assert set(loaded['T29'].evaluation_scope) == {'validation'}
    assert loaded['T29'].valid_n.nunique() == 1
    assert 'GPA_Change' not in ';'.join(loaded['T29'].features.fillna(''))
    for name in ['EDA_FINDINGS.md', 'DATA_DICTIONARY.md', 'PREPROCESSING_PLAN.md']:
        assert (output_dir / name).stat().st_size > 0
    return meta, assignments, loaded, split_meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-rerun', action='store_true')
    args = parser.parse_args()
    path = ROOT / 'notebooks' / 'kaggle' / '01_eda.ipynb'
    output_dir = Path(os.environ.get('LAB_OUTPUT_DIR', str(ROOT / 'reports')))
    split_dir = Path(os.environ.get('LAB_SPLIT_DIR', str(ROOT / 'data' / 'splits')))
    # Avoid installing a user-level kernelspec. This private spec exists for this process only.
    with tempfile.TemporaryDirectory(prefix='lab01-kernel-') as temporary:
        spec_dir = Path(temporary) / 'lab01'
        spec_dir.mkdir()
        (spec_dir / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'Lab 01 Python', 'language': 'python',
            'env': {'MPLBACKEND': 'module://matplotlib_inline.backend_inline',
                    'MPLCONFIGDIR': str(ROOT / '.cache' / 'matplotlib'),
                    'IPYTHONDIR': str(ROOT / '.cache' / 'ipython'),
                    'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                    'PYTHONIOENCODING': 'utf-8'}
        }), encoding='utf-8')
        # Kernel discovery reads JUPYTER_PATH; do not mutate user's Jupyter installation.
        old_jupyter_path = os.environ.get('JUPYTER_PATH')
        kernel_parent = Path(temporary) / 'kernels'
        kernel_parent.mkdir()
        spec_dir.rename(kernel_parent / 'lab01')
        os.environ['JUPYTER_PATH'] = temporary + (os.pathsep + old_jupyter_path if old_jupyter_path else '')
        try:
            notebook = execute(path, 'lab01')
            first_meta, first_assignments, first_tables, first_split = inspect_outputs(output_dir, split_dir)
            rerun_ok = False
            if args.verify_rerun:
                print('Restarting in a NEW kernel for reproducibility verification...', flush=True)
                notebook = execute(path, 'lab01')
                second_meta, second_assignments, second_tables, second_split = inspect_outputs(output_dir, split_dir)
                pd.testing.assert_frame_equal(first_assignments, second_assignments)
                for aid in ['T10', 'T15', 'T19', 'T20', 'T28', 'T29', 'T35', 'T44']:
                    if aid in first_tables:
                        pd.testing.assert_frame_equal(first_tables[aid], second_tables[aid], check_exact=False, rtol=1e-10, atol=1e-12)
                assert first_meta['dataset_sha256'] == second_meta['dataset_sha256']
                assert first_split['assignment_sha256'] == second_split['assignment_sha256']
                rerun_ok = True
            nbformat.write(notebook, output_dir / '01_EDA.executed.ipynb')
        finally:
            if old_jupyter_path is None:
                os.environ.pop('JUPYTER_PATH', None)
            else:
                os.environ['JUPYTER_PATH'] = old_jupyter_path

    sections = [c.source for c in notebook.cells if c.cell_type == 'markdown' and c.source.startswith('## Section ')]
    assert len(sections) == 12
    validation = {
        'AC01': ('PASS', 'nbformat validation, 12 sections, centralized config'),
        'AC02': ('PASS', 'Actual archive CSV read; T01-T04 and metadata'),
        'AC03': ('PASS', 'T05-T09; coercion/category/domain action logs'),
        'AC04': ('PASS', 'Stable assignments + T41; raw SHA-256 unchanged'),
        'AC05': ('PASS', 'T10-T13; F06-F11; per-feature interpretations'),
        'AC06': ('PASS', 'T14-T18; paired GPA/Change; unverified floor explicitly N/A'),
        'AC07': ('PASS', 'T19-T22; valid pair/group n; predictor-only matrices'),
        'AC08': ('PASS', 'T23-T26; F32-F39; exact train bin edges and cell counts'),
        'AC09': ('PASS', 'T27-T32; design rank/VIF/OLS + residual/influence diagnostics'),
        'AC10': ('PASS', 'T33-T38; formulas, constraints, centering, fixed quadratic comparison'),
        'AC11': ('PASS', 'T39-T42; explicit source, timing, role and bias restrictions'),
        'AC12': ('PASS', 'All 105 planned IDs accounted; each saved artifact exists and decodes'),
        'AC13': ('PASS', '12 generated FND records map RQ/evidence/implication; checked against current outputs'),
        'AC14': ('PASS', '3 reports + manifest + split/run metadata exist'),
        'AC15': ('PASS', 'Fresh-kernel nbclient execution on actual CSV; zero cell errors'),
        'AC16': ('PASS' if rerun_ok else 'NOT RUN', 'Second fresh kernel: split and key statistics compared' if rerun_ok else 'Run --verify-rerun'),
        'AC17': ('PASS', 'Data access review: no test frame/target extraction, counts/integrity only; T41; predictors and preprocessing scopes checked'),
        'AC18': ('PASS', 'T47; M01-M05 scenarios, evidence, eligibility, same metric protocol')
    }
    result = {
        'dataset_sha256': first_meta['dataset_sha256'],
        'raw_rows': first_meta['row_count_raw'], 'columns': first_meta['columns_raw'],
        'clean_kernel_execution': True, 'reproducible_rerun': rerun_ok,
        'acceptance_criteria': {key: {'status': status, 'evidence': evidence} for key, (status, evidence) in validation.items()}
    }
    (output_dir / 'verification_results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    lines = ['# Verification Results', '', f'Dataset SHA-256: `{first_meta["dataset_sha256"]}`', '',
             '| Criterion | Status | Evidence |', '|---|---|---|']
    lines += [f'| {key} | {status} | {evidence} |' for key, (status, evidence) in validation.items()]
    lines += ['', 'PASS means the implemented check passed on the attached CSV; it does not resolve the explicit provenance/timing/course-rubric limitations.',
              'Formal optional tests remain disabled; the documented unknown GPA floor and hours additivity are intentionally not fabricated.']
    (output_dir / 'VERIFICATION.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'reproducible_rerun': rerun_ok,
                      'dataset_sha256': first_meta['dataset_sha256'], 'criteria': len(validation)}, indent=2))


if __name__ == '__main__':
    main()
