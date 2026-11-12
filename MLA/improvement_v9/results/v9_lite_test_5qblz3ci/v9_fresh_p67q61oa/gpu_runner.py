"""GPU component execution using identical inputs, folds, models and cache contracts."""
import argparse
import json
import time
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import modeling as models
from engine import Engine, ROOT, sha, save, rmse
import pilot


def run(stage='screen', name=None, mode='plain', data_dir=None):
    e = Engine(data_dir or ROOT / 'data', ROOT / 'results')
    started = time.monotonic()
    if stage == 'screen':
        e.cache.stage = 'pilot'
        pilot.run(e)
        passed = json.loads((e.out / 'pilot.json').read_text())['families']['tabpfn']['eligible']
        todo = [(n, 'plain', s) for n, spec in models.SPECS.items() if spec['family'] == 'tabpfn' for s in [42, 71]] if passed else []
    elif stage == 'refine':
        assert name in models.SPECS and models.SPECS[name]['family'] == 'tabpfn'
        todo = [(name, 'ensemble16', s) for s in [42, 71]]
    elif stage == 'confirm':
        assert name in models.SPECS and models.SPECS[name]['family'] == 'tabpfn'
        todo = [(name, mode, s) for s in [48211, 48212, 48213, 'strat:27421', 'city']]
    else:
        raise ValueError(stage)
    e.cache.stage = 'screen' if stage == 'screen' else stage
    rows = []
    for n, m, scheme in todo:
        p = e.component(n, m, scheme)
        rows.append({'name': n, 'mode': m, 'scheme': str(scheme), 'rmse': rmse(e.y, p)})
        pd.DataFrame(rows).to_csv(e.out / f'gpu_{stage}.csv', index=False)
    import importlib.metadata, torch
    checkpoint_path = ROOT / 'tabpfn_checkpoint.json'
    record = {'stage': stage, 'inputs': e.inputs, 'sources': e.sources, 'core_runtime': e.runtime,
        'gpu_runtime': {p: importlib.metadata.version(p) for p in ['torch', 'tabpfn']},
        'gpu_name': torch.cuda.get_device_name(0), 'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
        'checkpoint': json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else None,
        'seconds': time.monotonic() - started, 'completed_components': rows,
        'files': {}, 'training_targets_retained': True, 'portal_uploads': 0}
    paths = list(e.cache.root.glob('tabpfn_*.npz')) + list(e.cache.root.glob('tabpfn_*.json'))
    paths += [e.out / 'pilot.json', e.out / 'compute_budget.json']
    if checkpoint_path.exists(): paths.append(checkpoint_path)
    if (e.out / f'gpu_{stage}.csv').exists(): paths.append(e.out / f'gpu_{stage}.csv')
    for p in paths: record['files'][str(p.relative_to(ROOT))] = sha(p)
    meta = e.out / f'gpu_{stage}_manifest.json'; save(meta, record)
    archive = ROOT / f'V9_GPU_{stage}.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in paths + [meta]: z.write(p, p.relative_to(ROOT))
    print('GPU_RESULT', json.dumps(record), flush=True)
    try:
        from google.colab import files
    except ImportError: pass
    else: files.download(str(archive))
    return archive


def import_result(e, archive):
    """Verify and import native fold predictions; reject differing source or folds."""
    with zipfile.ZipFile(archive) as z:
        metas = [n for n in z.namelist() if n.startswith('results/gpu_') and n.endswith('_manifest.json')]
        assert len(metas) == 1
        m = json.loads(z.read(metas[0]))
        assert m['inputs'] == e.inputs and m['sources'] == e.sources and m['core_runtime'] == e.runtime
        import hashlib
        for relative, expected in m['files'].items():
            assert hashlib.sha256(z.read(relative)).hexdigest() == expected, relative
        # Keep all remote evidence separate from the local budget and feasibility files.
        remote = e.out / 'gpu_evidence' / m['stage']; remote.mkdir(parents=True, exist_ok=True)
        for relative in m['files']:
            p = remote / relative; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(z.read(relative))
        save(remote / 'manifest.json', m)
        for relative in m['files']:
            if relative.startswith('results/cache/tabpfn_'):
                name = Path(relative).name
                if name.endswith('.npz'):
                    metadata = json.loads(z.read(str(Path(relative).with_suffix('.json'))))
                    ctx = metadata['context']; extra = ctx['extra']
                    a, b = e.splits(extra['scheme'])[extra['fold']]
                    assert ctx == e.context(ctx['name'], a, b, extra, ctx.get('execution_contract'))
                    pred = np.load(__import__('io').BytesIO(z.read(relative)), allow_pickle=False)['prediction']
                    assert pred.shape == (len(b),) and np.isfinite(pred).all() and (pred >= 0).all()
                destination = e.cache.root / name
                if destination.exists(): assert destination.read_bytes() == z.read(relative)
                else: destination.write_bytes(z.read(relative))
        if m['checkpoint']:
            save(ROOT / 'tabpfn_checkpoint.json', m['checkpoint'])
        local_pilot = json.loads((e.out / 'pilot.json').read_text())
        remote_pilot = json.loads(z.read('results/pilot.json'))
        if remote_pilot['families']['tabpfn']['eligible']:
            local_pilot['families']['tabpfn'] = {**remote_pilot['families']['tabpfn'], 'execution_location': 'free Colab T4 GPU', 'provenance': str(remote / 'manifest.json')}
        else:
            local_pilot['families']['tabpfn']['remote_pilot'] = remote_pilot['families']['tabpfn']
        save(e.out / 'pilot.json', local_pilot)
        if m['stage'] == 'screen':
            state = e.cache.state(); state['gpu_elapsed_seconds'] = m['seconds']
            state['seconds'] += m['seconds']; save(e.cache.budget, state)
    return m


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['screen','refine','confirm','import'])
    parser.add_argument('--name'); parser.add_argument('--mode', default='plain'); parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    if args.stage == 'import': import_result(Engine(), args.archive)
    else: run(args.stage, args.name, args.mode)
