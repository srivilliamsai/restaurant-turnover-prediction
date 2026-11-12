"""Independently recompute reported scores and inspect final notebook/CSV integrity."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import nbformat

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
STEM = 'submission'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == '__main__':
    manifest = json.loads((ROOT / 'delivery_manifest.json').read_text())
    y = pd.read_csv(ROOT.parent / 'Train_dataset.csv')['Annual Turnover'].to_numpy(float)
    test = pd.read_csv(ROOT.parent / 'Test_dataset.csv')
    for name, expected in manifest['inputs'].items():
        assert sha(ROOT.parent / name) == expected
    for name, expected in manifest['outputs'].items():
        assert sha(ROOT / name) == expected
    for name, expected in manifest['preserved_v5'].items():
        assert sha(ROOT / 'scored_v5_backup' / name) == expected
        assert sha(ROOT.parent / 'improvement_v5' / name) == expected
    for name, expected in manifest['implementation'].items():
        assert sha(ROOT / name) == expected
    screen = pd.read_csv(OUT / 'screening.csv')
    assert len(screen) == 48 and screen['name'].nunique() == 12
    development_scores = 0
    for row in screen.to_dict(orient='records'):
        for scheme in [42, 71]:
            baseline = np.load(OUT / 'cache' / f'v5_{scheme}.npz')['prediction']
            component = np.load(OUT / f'{row["name"]}_{scheme}.npz')['oof']
            prediction = (1 - row['weight']) * baseline + row['weight'] * component
            score = float(np.sqrt(np.mean((y - prediction) ** 2)))
            gain = float(np.sqrt(np.mean((y - baseline) ** 2))) - score
            assert abs(score - row[f'rmse_{scheme}']) < .01
            assert abs(gain - row[f'gain_{scheme}']) < .01
            development_scores += 1
    confirm = pd.read_csv(OUT / 'confirmation.csv')
    for row in confirm.to_dict(orient='records'):
        scheme = str(row['scheme']).replace(':', '_')
        baseline = np.load(OUT / 'cache' / f'v5_{scheme}.npz')['prediction']
        component = np.load(OUT / f'{row["name"]}_{scheme}.npz')['oof']
        p = (1 - row['weight']) * baseline + row['weight'] * component
        score = float(np.sqrt(np.mean((y - p) ** 2)))
        assert abs(score - row['candidate_rmse']) < .01
        assert abs(float(np.sqrt(np.mean((y - baseline) ** 2))) - row['v5_rmse']) < .01
        assert abs(row['v5_rmse'] - row['candidate_rmse'] - row['gain']) < .01
    inspected = []
    v5 = pd.read_csv(ROOT / 'scored_v5_backup' / f'{STEM}.csv')['Annual Turnover'].to_numpy()
    for i, check in enumerate(manifest['checks'], 1):
        directory = ROOT / 'candidates' / f'trial_{i}'
        frame = pd.read_csv(directory / f'{STEM}.csv')
        assert frame.shape == (500, 2)
        assert frame.columns.tolist() == ['Registration Number', 'Annual Turnover']
        assert frame['Registration Number'].equals(test['Registration Number'])
        p = frame['Annual Turnover'].to_numpy(float)
        assert np.isfinite(p).all() and (p >= 0).all()
        assert frame['Registration Number'].is_unique
        independent = np.load(OUT / f'full_fit_trial_{i}.npz')['prediction']
        assert np.max(np.abs(p - independent)) < .01
        assert sha(directory / f'{STEM}.csv') != manifest['preserved_v5'][STEM + '.csv']
        change = float(np.sqrt(np.mean((p - v5) ** 2)))
        assert abs(change - check['prediction_rms_difference_from_v5']) < .01
        assert abs(max(0., 12046722.3691139 - change) - check['conditional_portal_rmse_lower_bound']) < .01
        book = nbformat.read(directory / f'{STEM}.ipynb', as_version=4)
        nbformat.validate(book)
        cells = [c for c in book.cells if c.cell_type == 'code']
        assert all(c.execution_count is not None for c in cells)
        assert not any(o.output_type == 'error' for c in cells for o in c.outputs)
        for cell in cells:
            compile(cell.source, '<candidate notebook>', 'exec')
        execution = json.loads((OUT / f'notebook_execution_trial_{i}.json').read_text())
        assert execution['validation_checkpoint_reads'] == 0
        assert execution['maximum_full_fit_difference'] < .01 and execution['maximum_oof_difference'] < .01
        inspected.append({'trial': f'trial_{i}', 'code_cells': len(cells),
                          'csv_rows': len(frame), 'csv_differs_from_v5': True})
    research = nbformat.read(ROOT / 'V7_Research_Colab.ipynb', as_version=4)
    nbformat.validate(research)
    for cell in research.cells:
        if cell.cell_type == 'code':
            compile(cell.source, '<research notebook>', 'exec')
    lock = json.loads((OUT / 'candidate_lock.json').read_text())
    assert len(lock['shortlist']) == len({c['family'] for c in lock['shortlist']})
    assert manifest['candidate_lock_sha256'] == sha(OUT / 'candidate_lock.json')
    report = {'status': 'PASS', 'independent_development_scores_recomputed': development_scores,
              'independent_confirmation_scores_recomputed': len(confirm),
              'candidate_notebooks_and_csvs': inspected, 'v5_preserved': True,
              'output_and_implementation_hashes_verified': True,
              'research_notebook_syntax_and_format_valid': True,
              'portal_uploads': 0}
    (OUT / 'independent_delivery_checks.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
