"""Freeze our own V7 implementation and audited OOF snapshots; never modify V7."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
V7 = BASE / 'improvement_v7'
STEM = 'submission'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()


def run():
    manifest = json.loads((V7 / 'delivery_manifest.json').read_text())
    sources = {n: sha(V7 / n) for n in ['reference.py', 'models.py', 'workflow.py', 'scored_reference.json']}
    inputs = {n: sha(BASE / n) for n in manifest['inputs']}
    assert inputs == manifest['inputs']
    contract = json.loads((V7 / 'v5_contract.json').read_text())
    original = (V7 / 'models.py').read_text()
    assert 'import reference as ref' in original
    (ROOT / 'legacy_reference.py').write_bytes((V7 / 'reference.py').read_bytes())
    (ROOT / 'legacy_models.py').write_text(original.replace('import reference as ref', 'import legacy_reference as ref'))
    shutil.copyfile(V7 / 'v5_contract.json', ROOT / 'v5_contract.json')
    backup = ROOT / 'scored_v7_backup'
    backup.mkdir(exist_ok=True)
    preserved = {}
    for ext in ['csv', 'ipynb']:
        relative = f'candidates/trial_1/{STEM}.{ext}'
        assert sha(V7 / relative) == manifest['outputs'][relative]
        shutil.copyfile(V7 / relative, backup / f'{STEM}.{ext}')
        preserved[f'{STEM}.{ext}'] = sha(backup / f'{STEM}.{ext}')
    (ROOT / 'baseline_snapshots').mkdir(exist_ok=True)
    # Import only our frozen code, after the files have been copied.
    import legacy_models as legacy
    train, _, _ = legacy.load_data(BASE)
    snapshots = {}
    for scheme in [42, 71, 'strat:27421', 'city']:
        safe = str(scheme).replace(':', '_')
        old_base = V7 / 'results/cache' / f'v5_{safe}.npz'
        old_component = V7 / 'results' / f'residual_tree_l2_{safe}.npz'
        records = []
        for p in [old_base, old_component]:
            m = json.loads(p.with_suffix('.json').read_text())
            assert m['sha256'] == sha(p)
            assert m['context']['inputs'] == inputs and m['context']['sources'] == sources
            assert m['context']['contract'] == digest(contract)
            assert m['context']['training_rows'] == digest(train['Registration Number'].tolist())
            assert m['context']['extra']['scheme'] == str(scheme)
            records.append(m)
        folds = digest([b.tolist() for _, b in legacy.split_indices(train, scheme)])
        assert records[0]['context']['extra']['folds'] == folds
        assert records[1]['context']['spec'] == legacy.SPECS['residual_tree_l2']
        assert records[1]['context']['seeds'] == legacy.SEEDS
        b = np.load(old_base, allow_pickle=False)['prediction']
        c = np.load(old_component, allow_pickle=False)
        assert np.max(np.abs(c['individual'].mean(axis=1) - c['oof'])) < .01
        prediction = .25 * b + .75 * c['oof']
        path = ROOT / 'baseline_snapshots' / f'v7_{safe}.npz'
        np.savez_compressed(path, prediction=prediction)
        snapshots[str(scheme)] = {'file': str(path.relative_to(ROOT)), 'sha256': sha(path),
            'fold_sha256': folds, 'train_ids': digest(train['Registration Number'].tolist()),
            'inputs': inputs, 'source_metadata': records}
    provenance = {'original_v7_sources': sources, 'inputs': inputs, 'preserved_v7': preserved,
        'import_only_change': 'legacy_models.py changes import reference as ref to import legacy_reference as ref',
        'frozen_files': {n: sha(ROOT / n) for n in ['legacy_reference.py', 'legacy_models.py', 'v5_contract.json']},
        'snapshots': snapshots, 'scored_v7_portal_rmse': 12024316.5355221}
    (ROOT / 'reference_manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('PASS: frozen V7 source, scored files and four verified OOF snapshots.', flush=True)


if __name__ == '__main__':
    run()
