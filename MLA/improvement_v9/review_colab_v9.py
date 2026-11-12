"""Independently review the downloaded V9 run without fitting any models."""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('archive', type=Path)
parser.add_argument('--data-dir', type=Path, required=True)
args = parser.parse_args()
archive = args.archive.resolve()
destination = archive.parent / 'review_payload'
assert not destination.exists(), 'Use a fresh review directory.'
destination.mkdir()
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for entry in z.infolist():
        path = Path(entry.filename)
        assert not path.is_absolute() and '..' not in path.parts
    z.extractall(destination)

for name in ['engine.py', 'modeling.py', 'baseline.py', 'legacy_reference.py',
             'legacy_models.py', 'workflow.py', 'pilot.py', 'gpu_runner.py',
             'reference_manifest.json', 'v5_contract.json']:
    expected = hashlib.sha256((Path(__file__).resolve().parent / name).read_bytes()).hexdigest()
    assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == expected, name

sys.path.insert(0, str(destination))
import numpy as np
import pandas as pd
from engine import Engine, digest, rmse, sha, validate_csv
import modeling

e = Engine(args.data_dir, destination / 'results')
audit = json.loads((e.out / 'audit.json').read_text())
assert audit['sources'] == e.sources and audit['inputs'] == e.inputs
assert len(e.y) == 3493
pilot = json.loads((e.out / 'pilot.json').read_text())
plan = json.loads((e.out / 'experiment_plan.json').read_text())
lock = json.loads((e.out / 'candidate_lock.json').read_text())
decision = json.loads((e.out / 'decision.json').read_text())
assert lock['sources'] == e.sources and lock['inputs'] == e.inputs
assert lock['plan_sha256'] == sha(e.out / 'experiment_plan.json')
assert decision['lock_sha256'] == sha(e.out / 'candidate_lock.json')

predictions = {}
def component(name, mode, scheme):
    key = (name, mode, str(scheme))
    if key in predictions:
        return predictions[key]
    if name == 'v7' and str(scheme) in e.manifest['snapshots']:
        predictions[key] = e.baseline(scheme)
        return predictions[key]
    result = np.zeros(len(e.y))
    for number, (a, b) in enumerate(e.splits(scheme)):
        extra = {'scheme': str(scheme), 'fold': number}
        if name == 'v7':
            stem = f'v7_{str(scheme).replace(":", "_")}_fold{number}'
            contract = None
        else:
            extra.update(mode=mode, spec=modeling.SPECS[name])
            stem = f'{name}_{mode}_{str(scheme).replace(":", "_")}_fold{number}'
            contract = pilot['families']['tabpfn']['execution_contract'] if name.startswith('tabpfn_') else None
        path = e.cache.root / (stem + '.npz')
        metadata = json.loads(path.with_suffix('.json').read_text())
        assert metadata['sha256'] == sha(path)
        assert metadata['context'] == e.context(name, a, b, extra, contract)
        values = np.load(path, allow_pickle=False)['prediction']
        assert values.shape == (len(b),) and np.isfinite(values).all() and (values >= 0).all()
        result[b] = values
    predictions[key] = result
    return result

development = pd.read_csv(e.out / 'development.csv')
assert len(development) <= 40
score_differences = []
segments, folds = [], []
for row in development.to_dict(orient='records'):
    for scheme in [42, 71]:
        base = component('v7', 'plain', scheme)
        candidate = component(row['name'], row['mode'], scheme)
        blended = (1-row['weight'])*base + row['weight']*candidate
        measured = rmse(e.y, blended)
        score_differences.append(abs(measured-row[f'rmse_{scheme}']))
        assert abs((rmse(e.y, base)-measured)-row[f'gain_{scheme}']) < .01
        common = {k: row[k] for k in ['name', 'mode', 'weight']}
        for label, mask in [('<=40M', e.y <= 40e6), ('40-80M', (e.y > 40e6) & (e.y <= 80e6)), ('>80M', e.y > 80e6)]:
            segments.append({**common, 'scheme': scheme, 'segment': label, 'rows': int(mask.sum()),
                'v7_rmse': rmse(e.y[mask], base[mask]), 'candidate_rmse': rmse(e.y[mask], blended[mask]),
                'sse_reduction': float(np.sum((e.y[mask]-base[mask])**2-(e.y[mask]-blended[mask])**2))})
        for number, (_, index) in enumerate(e.splits(scheme)):
            folds.append({**common, 'scheme': scheme, 'fold': number, 'rows': len(index),
                'v7_rmse': rmse(e.y[index], base[index]), 'candidate_rmse': rmse(e.y[index], blended[index])})
assert max(score_differences, default=0) < .01
pd.DataFrame(segments).to_csv(archive.parent / 'development_segments.csv', index=False)
pd.DataFrame(folds).to_csv(archive.parent / 'development_folds.csv', index=False)

confirmation = []
for choice in decision['candidates']:
    outcomes = {}
    for scheme in [48211, 48212, 48213, 'strat:27421', 'city']:
        base = component('v7', 'plain', scheme)
        prediction = (1-choice['weight'])*base + choice['weight']*component(choice['name'], choice['mode'], scheme)
        old, new = rmse(e.y, base), rmse(e.y, prediction)
        recorded = choice['metrics'][str(scheme)]
        assert abs(new-recorded['candidate_rmse']) < .01
        assert abs(old-recorded['v7_rmse']) < .01
        outcomes[str(scheme)] = {'gain': old-new, 'relative_change': new/old-1}
    qualified = all(outcomes[str(s)]['gain'] > 0 for s in [48211, 48212, 48213]) and all(
        outcomes[s]['relative_change'] <= .01 for s in ['strat:27421', 'city'])
    assert qualified == choice['qualified']
    assert choice['uncertainty']['repetitions'] == 5000
    confirmation.append({'name': choice['name'], 'mode': choice['mode'], 'weight': choice['weight'],
                         'qualified': qualified, 'outcomes': outcomes, 'uncertainty': choice['uncertainty']})

reproduction = json.loads((e.out / 'v7_full_reproduction.json').read_text())
assert reproduction['fresh_full_fit'] and reproduction['maximum_difference_INR'] < .01
for name, expected in e.manifest['preserved_v7'].items():
    assert sha(destination / 'scored_v7_backup' / name) == expected
candidate_checks = json.loads((e.out / 'candidate_notebook_checks.json').read_text())
assert len(candidate_checks) == len(decision['trials'])
exported = []
if decision['trials']:
    for number, choice in enumerate(decision['trials'], 1):
        directory = destination / 'candidates' / f'trial_{number}'
        csv = next(directory.glob('*.csv'))
        validate_csv(pd.read_csv(csv), e.test)
        exported.append({'trial': number, 'name': choice['name'], 'weight': choice['weight'],
                         'csv': str(csv), 'sha256': sha(csv)})
else:
    for name, expected in e.manifest['preserved_v7'].items():
        assert sha(destination / 'recommended_submission' / name) == expected

record = {'status': 'PASS', 'archive_sha256': sha(archive), 'inputs': e.inputs,
    'sources': e.sources, 'core_runtime': e.runtime, 'pilot': pilot['families'],
    'development_configurations': len(development), 'development_scores_recomputed': len(score_differences),
    'maximum_score_difference_INR': max(score_differences, default=0),
    'development_segment_rows': len(segments), 'development_fold_rows': len(folds),
    'all_training_rows_retained': len(e.y), 'confirmation': confirmation,
    'qualified_candidates': len(decision['trials']), 'exports': exported,
    'fresh_v7_reproduction': reproduction, 'fresh_candidate_notebooks': candidate_checks,
    'decision': decision['decision'], 'new_portal_scores_observed': 0,
    'files': {str(p.relative_to(destination)): sha(p) for p in destination.rglob('*') if p.is_file()}}
(archive.parent / 'independent_review.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({k: v for k, v in record.items() if k not in ['files', 'sources', 'pilot']}, indent=2))
