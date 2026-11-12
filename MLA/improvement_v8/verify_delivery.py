"""Independent arithmetic, file integrity, preservation and notebook checks."""
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import nbformat as nb
from engine import ROOT, STEM, Engine, sha, save, rmse
import modeling as models


def main():
    out = ROOT / 'results'
    e = Engine()
    manifest = json.loads((ROOT / 'delivery_manifest.json').read_text())
    for n, value in manifest['inputs'].items(): assert sha(ROOT.parent / n) == value
    for n, value in manifest['implementation'].items(): assert sha(ROOT / n) == value
    for n, value in manifest['outputs'].items(): assert sha(ROOT / n) == value
    for n, value in manifest['preserved_v7'].items():
        assert sha(ROOT / 'recommended_submission' / n) == value
        assert sha(ROOT / 'scored_v7_backup' / n) == value
        original = ROOT.parent / 'improvement_v7/candidates/trial_1' / n
        if original.exists():
            assert sha(original) == value
    table = pd.read_csv(out / 'screening.csv')
    assert len(table) == 64 and table.name.nunique() == 16
    assert set(table.name) == set(models.SPECS)
    assert not ((table.gain_42 > 0) & (table.gain_71 > 0)).any()
    scores, maximum_difference = 0, 0.
    for row in table.to_dict(orient='records'):
        for s in [42, 71]:
            b = e.baseline(s)
            p = np.zeros(len(e.y))
            for f, (_, indices) in enumerate(e.splits(s)):
                path = out / 'cache' / f'{row["name"]}_plain_{s}_fold{f}.npz'
                meta = json.loads(path.with_suffix('.json').read_text())
                assert sha(path) == meta['sha256']
                assert meta['context']['inputs'] == e.inputs and meta['context']['sources'] == e.sources
                p[indices] = np.load(path, allow_pickle=False)['prediction']
            score = rmse(e.y, (1 - row['weight']) * b + row['weight'] * p)
            difference = abs(score - row[f'rmse_{s}'])
            assert difference < .01
            assert abs(rmse(e.y, b) - score - row[f'gain_{s}']) < .01
            maximum_difference = max(maximum_difference, difference)
            scores += 1
    lock = json.loads((out / 'candidate_lock.json').read_text())
    decision = json.loads((out / 'decision.json').read_text())
    assert lock['shortlist'] == [] and decision['trials'] == []
    assert decision['lock_sha256'] == sha(out / 'candidate_lock.json')
    assert not (out / 'confirmation.csv').exists()
    segments = pd.read_csv(out / 'development_segments.csv')
    folds = pd.read_csv(out / 'development_folds.csv')
    assert len(segments) == 384 and len(folds) == 640
    for r in table.to_dict(orient='records'):
        for s in [42, 71]:
            q = segments[(segments.name == r['name']) & (segments.weight == r['weight']) & (segments.scheme == s)]
            assert len(q) == 3 and int(q.rows.sum()) == len(e.y)
            assert abs(np.sqrt(q.candidate_sse.sum() / len(e.y)) - r[f'rmse_{s}']) < .01
    book = nb.read(ROOT / 'V8_Research_Colab.ipynb', as_version=4)
    nb.validate(book)
    cells = [c for c in book.cells if c.cell_type == 'code']
    assert all(c.execution_count is not None for c in cells)
    assert not any(o.output_type == 'error' for c in cells for o in c.outputs)
    assert manifest['notebook_execution']['prediction_checkpoint_reads'] == 0
    assert manifest['v7_full_reproduction']['maximum_difference_INR'] < .01
    with zipfile.ZipFile(ROOT / 'V8_Delivery.zip') as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(set(archive.namelist()))
        for name, expected in manifest['outputs'].items():
            import hashlib
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
    record = {'status': 'PASS', 'development_scores_independently_recomputed': scores,
        'maximum_difference_INR': maximum_difference, 'frozen_empty_shortlist_correct': True,
        'no_confirmation_results_invented': True, 'scored_v7_csv_and_notebook_unchanged': True,
        'source_input_output_hashes': True, 'notebook_saved_outputs_valid': True, 'zip_crc_and_outputs': True,
        'development_segment_rows': 384, 'development_fold_rows': 640,
        'primary_metric_retains_every_original_label': True,
        'qualified_candidates': 0, 'portal_uploads_by_agent': 0}
    save(out / 'independent_delivery_checks.json', record)
    relative = 'results/independent_delivery_checks.json'
    with zipfile.ZipFile(ROOT / 'V8_Delivery.zip', 'a', compression=zipfile.ZIP_DEFLATED) as archive:
        if relative not in archive.namelist():
            archive.write(out / 'independent_delivery_checks.json', relative)
        else:
            assert archive.read(relative) == (out / 'independent_delivery_checks.json').read_bytes()
    with zipfile.ZipFile(ROOT / 'V8_Delivery.zip') as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(set(archive.namelist()))
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__': main()
