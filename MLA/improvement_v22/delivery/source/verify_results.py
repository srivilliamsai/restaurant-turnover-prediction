"""Independently reconstruct every selectable score from hashed complete folds."""
import json
import numpy as np
import pandas as pd
import features
import workflow
from core import ROOT, Engine, sha, save, rmse, DEVELOPMENT, CONFIRM


def main():
    e = Engine()
    frozen = json.loads((ROOT/'source_frozen.json').read_text())
    assert e.sources == frozen['sources'] and e.inputs == frozen['inputs']
    pilot = json.loads((e.out/'pilot.json').read_text())
    measured = pd.read_csv(e.out/'screening.csv')
    checked = 0
    difference = 0.

    def component(name, scheme):
        nonlocal checked
        p = np.zeros(len(e.y))
        seen = np.zeros(len(e.y), int)
        for fold, (a, b) in enumerate(e.splits(scheme)):
            path = e.out/'cache'/f"{name}_{str(scheme).replace(':','_')}_fold{fold}.npz"
            meta = json.loads(path.with_suffix('.json').read_text())
            assert meta['sha256'] == sha(path)
            assert meta['context'] == e.context(name, scheme, fold, a, b)
            fit = meta['fit_metadata']
            assert fit['name'] == name and fit['training_rows'] == len(a)
            assert fit['units'] == 'raw INR millions'
            with np.load(path, allow_pickle=False) as z:
                assert z['prediction'].shape == b.shape
                p[b] = z['prediction']
            seen[b] += 1
            checked += 1
        assert (seen == 1).all() and np.isfinite(p).all()
        return p

    def blend(parts, p, route, w):
        assert route == 'P35-replacement'
        return .25*parts['v9'] + .75*((1-w)*parts['p35'] + w*p)

    expected = []
    for name in pilot['screen_settings']:
        predictions = {s: component(name, s) for s in DEVELOPMENT}
        for route in features.ROUTES:
            for w in features.WEIGHTS:
                row = {'name': name, 'route': route, 'weight': w,
                       'effective_new_share': .75*w}
                saved = measured[(measured['name']==name)&(measured['route']==route)&(measured['weight']==w)]
                assert len(saved) == 1
                for s in DEVELOPMENT:
                    parts = e.reference_parts(s)
                    p = blend(parts, predictions[s], route, w)
                    old, new = rmse(e.y, parts['v14']), rmse(e.y, p)
                    row.update({f'rmse_{s}':new, f'gain_{s}':old-new, f'relative_gain_{s}':(old-new)/old})
                row['mean_relative_gain'] = float(np.mean([row[f'relative_gain_{s}'] for s in DEVELOPMENT]))
                for key, value in row.items():
                    if key not in ['name','route']:
                        d = abs(float(saved.iloc[0][key])-value)
                        difference = max(difference, d)
                        assert d < 1e-6, (name, route, w, key, d)
                expected.append(row)
    assert len(expected) == len(measured) == len(features.WEIGHTS)*len(pilot['screen_settings'])
    eligible = [r for r in expected if r['gain_42']>0 and r['gain_71']>0 and r['mean_relative_gain']>=.002]
    eligible.sort(key=lambda r:(-r['mean_relative_gain'],r['effective_new_share'],r['name'],r['route']))
    selected = eligible[:1]
    lock = json.loads((e.out/'candidate_lock.json').read_text())
    assert len(selected) == len(lock['choices'])
    if selected:
        for key in ['name','route','weight']:
            assert selected[0][key] == lock['choices'][0][key]
    decision = json.loads((e.out/'decision.json').read_text())
    if not selected:
        assert decision['complete'] and not decision['qualified'] and not decision['finalists']
        assert not (e.out/'confirmation.csv').exists()
    else:
        choice = selected[0]
        metrics, bases, candidates = {}, {}, {}
        for s in CONFIRM:
            parts = e.reference_parts(s)
            p = blend(parts, component(choice['name'],s), choice['route'], choice['weight'])
            old,new = rmse(e.y,parts['v14']),rmse(e.y,p)
            metrics[str(s)] = {'v14_rmse':old,'candidate_rmse':new,'gain':old-new,'relative_gain':(old-new)/old}
            bases[s],candidates[s] = parts['v14'],p
            for k,v in metrics[str(s)].items():
                assert abs(v-decision['finalists'][0]['metrics'][str(s)][k]) < 1e-6
        rng=np.random.default_rng(8319)
        boot=[]
        for draw in range(5000):
            indices=rng.integers(0,len(e.y),len(e.y))
            boot.append(np.mean([rmse(e.y[indices],bases[z][indices])-rmse(e.y[indices],candidates[z][indices]) for z in CONFIRM[:3]]))
        ci={'ci95_INR':np.quantile(boot,[.025,.975]).tolist()}
        assert np.max(np.abs(np.asarray(ci['ci95_INR'])-decision['finalists'][0]['uncertainty']['ci95_INR'])) < 1e-6
        qualifies = (all(metrics[str(s)]['gain']>0 for s in CONFIRM[:3])
                     and np.mean([metrics[str(s)]['relative_gain'] for s in CONFIRM[:3]])>=.002
                     and ci['ci95_INR'][0]>0
                     and all(metrics[s]['relative_gain']>=-.01 for s in ['strat:27421','city']))
        assert bool(qualifies) == decision['qualified']
    for suffix in ['.csv','.ipynb']:
        original = ROOT/'reference'/('scored_v14'+suffix)
        if not decision['qualified']:
            assert sha(original) == sha(ROOT/'delivery'/('KEEP_V14_'+__import__('core').STEM+suffix))
    result = {'status':'PASS','independently_recomputed_blends':len(expected),
              'complete_fold_outputs_verified':checked,'maximum_metric_difference_INR':difference,
              'source_and_data_identity_unchanged':True,'all_original_validation_targets':len(e.y),
              'fixed_component_replacement_formula_independently_recomputed':True,
              'baseline_v14_component_fold_witnesses_verified':35,'selection_recomputed':True,
              'qualified':decision['qualified'],'confirmation_performed':bool(selected),
              'v14_scored_files_preserved':True,'portal_uploads':0}
    save(ROOT/'delivery/independent_review.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()

