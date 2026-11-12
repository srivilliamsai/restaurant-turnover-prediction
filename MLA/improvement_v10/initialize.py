"""Freeze verified V9 evidence without changing any earlier workflow or model."""
import ast
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
V9 = ROOT.parent/'improvement_v9'
RUN = V9/'colab_cli_runs/20261026T160148Z'
sys.path.insert(0, str(V9))
import numpy as np
from engine import Engine, sha, digest
import modeling as old_modeling


def copy_checked(source, destination, expected=None):
    actual = sha(source)
    if expected is not None:
        assert actual == expected, str(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        assert sha(destination) == actual, f'Existing frozen file changed: {destination}'
    else:
        shutil.copy2(source, destination)
    return actual


def main():
    ROOT.mkdir(exist_ok=True)
    delivery = json.loads((RUN/'portable_delivery/manifest.json').read_text())
    for name in ['legacy_reference.py', 'legacy_models.py', 'baseline.py', 'v5_contract.json']:
        copy_checked(V9/name, ROOT/name, delivery['source_hashes'][name])
    copy_checked(V9/'modeling.py', ROOT/'frozen_v9_modeling.py', delivery['source_hashes']['modeling.py'])
    review = json.loads((ROOT.parent/'improvement_v10_review/source_and_data_audit.json').read_text())
    map_text = json.dumps(review['proposed_city_map'], indent=2, sort_keys=True)+'\n'
    if (ROOT/'city_map.json').exists():
        assert (ROOT/'city_map.json').read_text() == map_text
    else:
        (ROOT/'city_map.json').write_text(map_text)
    ref = ROOT/'reference'; ref.mkdir(exist_ok=True)
    for suffix in ['csv','ipynb']:
        name = f'submission.{suffix}'
        expected = delivery['files']['submission/'+name]
        copy_checked(V9/'scored_v9_backup'/name, ref/('scored_v9.'+suffix), expected)
    copy_checked(V9/'scored_v7_backup/submission.csv', ref/'scored_v7.csv',
                 '5f20ea8d610e325c0eae4727b7ed8c737ce2d881239792c9a6af5ce4e3dec1ba')
    copy_checked(RUN/'export_diagnostics/local_v7_fit.pkl', ref/'verified_v7.pkl', delivery['v7_checkpoint_sha256'])
    for source, name in [(RUN/'export_diagnostics/local_comparison.json','v7_original_platform_reproduction.json'),
                         (RUN/'portability_confirmation/decision.json','v9_corrected_confirmation.json'),
                         (RUN/'portable_delivery/manifest.json','v9_delivery_manifest.json'),
                         (V9/'portal_result_2026-10-27.json','v9_portal_result.json')]:
        copy_checked(source, ref/name)
    e = Engine(ROOT.parent, ROOT/'preparation_context')
    source_cache = RUN/'partial_backup/results/cache'
    execution = delivery['tabpfn_execution_contract']
    manifest = {'inputs': e.inputs, 'execution_contract': execution,
        'v9_portal_rmse': 11904515.9864996, 'screenshot_rank': 9, 'oof': {},
        'original_v9_sources': e.sources, 'v7_checkpoint_sha256': delivery['v7_checkpoint_sha256'],
        'reference_files': {}, 'city_map_sha256': sha(ROOT/'city_map.json')}
    for scheme in [42,71,48211,48212,48213,'strat:27421']:
        folds = e.splits(scheme); b = np.zeros(len(e.y)); t = np.zeros(len(e.y)); evidence = []
        if str(scheme) in e.manifest['snapshots']:
            b = e.baseline(scheme)
            snapshot = e.manifest['snapshots'][str(scheme)]
            evidence.append({'kind':'original-platform V7 snapshot',**snapshot})
        for f,(a,v) in enumerate(folds):
            if str(scheme) not in e.manifest['snapshots']:
                p = RUN/'portability_confirmation/cache'/f'v7_{scheme}_fold{f}.npz'
                m = json.loads(p.with_suffix('.json').read_text())
                assert m['sha256'] == sha(p)
                assert m['context'] == e.context('v7',a,v,{'scheme':str(scheme),'fold':f})
                b[v] = np.load(p,allow_pickle=False)['prediction']
                evidence.append({'kind':'fresh original-platform V7 fold','path':str(p),
                                 'sha256':sha(p),'metadata_sha256':sha(p.with_suffix('.json'))})
            p = source_cache/f'tabpfn_raw_v7_ensemble16_{str(scheme).replace(":","_")}_fold{f}.npz'
            m = json.loads(p.with_suffix('.json').read_text())
            assert sha(p) == m['sha256']
            expected = e.context('tabpfn_raw_v7',a,v,{'scheme':str(scheme),'fold':f,'mode':'ensemble16',
                                                  'spec':old_modeling.SPECS['tabpfn_raw_v7']},execution)
            assert m['context'] == expected, str(p)
            t[v] = np.load(p,allow_pickle=False)['prediction']
            evidence.append({'kind':'verified original V9 TabPFN fold','path':str(p),
                             'sha256':sha(p),'metadata_sha256':sha(p.with_suffix('.json'))})
        assert np.isfinite(b).all() and np.isfinite(t).all()
        pred = .5*b+.5*t
        if scheme not in [42,71]:
            p = RUN/'portability_confirmation'/(str(scheme).replace(':','_')+'.npz')
            z = np.load(p,allow_pickle=False)
            assert np.array_equal(z['baseline'],b) and np.array_equal(z['component'],t)
            assert np.array_equal(z['prediction'],pred)
        name = str(scheme).replace(':','_'); destination = ref/f'oof_{name}.npz'
        assignment = np.zeros(len(e.y),int)
        for f,(_,v) in enumerate(folds):assignment[v]=f
        np.savez_compressed(destination,v7=b,t9=t,v9=pred,fold=assignment,
                            ids=e.train['Registration Number'].to_numpy())
        manifest['oof'][name] = {'file':str(destination.relative_to(ROOT)), 'sha256':sha(destination),
            'fold_sha256':digest([v.tolist() for _,v in folds]),
            'row_ids_sha256':digest(e.train['Registration Number'].tolist()), 'verified_from':evidence,
            'v9_rmse':float(np.sqrt(np.mean((e.y-pred)**2)))}
    for p in sorted(ref.iterdir()):
        if p.is_file(): manifest['reference_files'][str(p.relative_to(ROOT))]=sha(p)
    manifest_path = ROOT/'reference_manifest.json'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    zone=ZoneInfo('Asia/Kolkata');inventory=[]
    for path in sorted(Path('~/Downloads').glob('gemini-code-*.py')):
        text=path.read_text();lines=text.splitlines();colab=bool(lines and lines[0].startswith('!pip '))
        parse_text='\n'+'\n'.join(lines[1:]) if colab else text
        try:ast.parse(parse_text);syntax='parseable after notebook setup handling'
        except SyntaxError as error:syntax=f'{error.msg}; line {error.lineno}'
        inventory.append({'path':str(path),'sha256':sha(path),'lines':len(lines),
                          'downloaded_local':datetime.fromtimestamp(path.stat().st_birthtime,zone).isoformat(),
                          'colab_setup_command':colab,'syntax':syntax,'executed':False})
    (ROOT/'source_inventory.json').write_text(json.dumps({'scope':'Gemini source inventory; scripts not executed.',
        'count':len(inventory),'files':inventory},indent=2)+'\n')
    print('PASS: preserved scored V9 CSV/notebook, verified own V7 checkpoint and six complete V9 OOF assignments.')
    print('Original-platform baseline reproduction boundary preserved; revised city folds require fresh comparison.')


if __name__=='__main__':main()
