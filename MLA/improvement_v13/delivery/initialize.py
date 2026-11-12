"""Prepare isolated, hash-checked V9 controls without fitting any models."""
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ROOT.mkdir(exist_ok=True)
    authorization = ROOT / 'authorization.json'
    if not authorization.exists():
        authorization.write_text(json.dumps({
            'version': 'V13', 'maximum_hours': 8, 'paid_resources': False,
            'started_epoch': time.time(), 'portal_uploads': 0,
            'settings': ['Q99', 'Q995', 'SQ2'], 'weights': [.25, .5, .75, 1.],
            'user_authorization': 'User explicitly requested implementation of the complete V13 plan.',
        }, indent=2) + '\n')
    parent = PROJECT / 'improvement_v12'
    registry = json.loads((parent / 'baseline_registry.json').read_text())
    files = {}

    def copy(source, relative):
        destination = ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            assert sha(destination) == sha(source), f'Existing immutable file differs: {relative}'
        else:
            shutil.copy2(source, destination)
        files[str(relative)] = sha(destination)

    for name in ['baseline.py', 'legacy_models.py', 'legacy_reference.py',
                 'frozen_v9_modeling.py', 'v5_contract.json', 'city_map.json']:
        copy(parent / name, name)
    for name in ['verified_v7.pkl', 'scored_v7.csv', 'scored_v9.csv',
                 'scored_v9.ipynb', 'v7_original_platform_reproduction.json']:
        copy(parent / 'reference' / name, 'reference/' + name)
    for name, expected in registry['generating_sources'].items():
        source = parent / 'provenance' / name
        assert sha(source) == expected
        copy(source, 'provenance/' + name)
    schemes = registry['schemes']
    for key, record in schemes.items():
        copy(parent / record['file'], record['file'])
        copy(parent / record['v7_witness_file'], record['v7_witness_file'])
        record['v7_witness_kind'] = 'inherited_aggregate'
        record['tabpfn_witness_paths'] = []
        for fold in range(5):
            stem = f"BASE_{key.replace(':', '_')}_fold{fold}"
            for suffix in ['.npz', '.json']:
                relative = 'provenance/baseline_cache/' + stem + suffix
                copy(parent / relative, relative)
            record['tabpfn_witness_paths'].append('provenance/baseline_cache/' + stem + '.npz')
    for seed in [48211, 48212, 48213]:
        key = str(seed)
        source = PROJECT / 'improvement_v10/reference' / f'oof_{seed}.npz'
        copy(source, f'provenance/legacy_oof_{seed}.npz')
        relative = f'reference/v13_control_{seed}.npz'
        record = {'file': relative, 'inputs': registry['inputs'],
                  'row_ids_sha256': registry['row_ids_sha256'],
                  'v7_witness_kind': 'original_mac_folds', 'v7_witness_paths': [],
                  'tabpfn_witness_paths': [], 'control_behavior': schemes['42']['control_behavior']}
        for fold in range(5):
            mac = PROJECT / 'improvement_v9/colab_cli_runs/20261026T160148Z/portability_confirmation/cache'
            for suffix in ['.npz', '.json']:
                name = f'v7_{seed}_fold{fold}' + suffix
                copy(mac / name, 'provenance/mac_controls/' + name)
                name = f'BASE_{seed}_fold{fold}' + suffix
                copy(PROJECT / 'improvement_v10/colab_results/results/cache' / name,
                     'provenance/baseline_cache/' + name)
            record['v7_witness_paths'].append(f'provenance/mac_controls/v7_{seed}_fold{fold}.npz')
            record['tabpfn_witness_paths'].append(f'provenance/baseline_cache/BASE_{seed}_fold{fold}.npz')
        with np.load(source, allow_pickle=False) as old:
            assignments, ids = old['fold'].copy(), old['ids'].copy()
        v7, t9 = np.zeros(len(ids)), np.zeros(len(ids))
        for fold in range(5):
            rows = np.flatnonzero(assignments == fold)
            with np.load(ROOT / record['v7_witness_paths'][fold], allow_pickle=False) as data:
                v7[rows] = data['prediction']
            with np.load(ROOT / record['tabpfn_witness_paths'][fold], allow_pickle=False) as data:
                t9[rows] = data['prediction']
        destination = ROOT / relative
        if destination.exists():
            with np.load(destination, allow_pickle=False) as data:
                np.testing.assert_array_equal(data['v7'], v7)
                np.testing.assert_array_equal(data['t9'], t9)
        else:
            np.savez_compressed(destination, v7=v7, t9=t9, v9=.5*v7+.5*t9, ids=ids, fold=assignments)
        record['sha256'] = sha(destination)
        files[relative] = sha(destination)
        record['reconstruction'] = 'Original-Mac V7 folds plus hash-verified fixed-500-row TabPFN control folds; legacy aggregate is archived separately.'
        schemes[key] = record
    copy(parent / 'reference_manifest.json', 'provenance/parent_reference_manifest.json')
    manifest = json.loads((parent / 'reference_manifest.json').read_text())
    manifest.update(reference_files=files, parent_manifest_sha256=sha(parent / 'reference_manifest.json'))
    (ROOT / 'reference_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (ROOT / 'baseline_registry.json').write_text(json.dumps(registry, indent=2) + '\n')
    print('V13 immutable sources and all seven V9 control assignments prepared.')


if __name__ == '__main__':
    main()
