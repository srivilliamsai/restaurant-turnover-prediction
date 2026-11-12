"""Independent reconstruction of all exploratory follow-up evidence; no fits."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rank_check import RankEngine, identities
from core import ROOT, STEM, CONFIRM, sha, save, rmse


def main():
    e=RankEngine()
    frozen=json.loads((ROOT/'rank_frozen.json').read_text())
    assert e.sources==frozen['original_sources'] and e.inputs==frozen['inputs']
    assert all(sha(ROOT/n)==v for n,v in frozen['supplement_sources'].items())
    decision=json.loads((e.out/'decision.json').read_text())
    assert decision['choice']['weight']==1. and not decision['original_V22_development_gate_passed']
    bases,candidates,metrics={},{},{}
    count=0
    for scheme in CONFIRM:
        p=np.zeros(len(e.y));seen=np.zeros(len(e.y),int)
        for fold,(a,b) in enumerate(e.splits(scheme)):
            path=e.out/'cache'/f"P35-SC38_{str(scheme).replace(':','_')}_fold{fold}.npz"
            meta=json.loads(path.with_suffix('.json').read_text())
            assert sha(path)==meta['sha256'] and meta['context']==e.context('P35-SC38',scheme,fold,a,b)
            assert meta['fit_metadata']['input_features']==38
            assert meta['fit_metadata']['training_rows']==len(a) and meta['fit_metadata']['units']=='raw INR millions'
            with np.load(path,allow_pickle=False) as z:
                assert z['prediction'].shape==b.shape and np.isfinite(z['prediction']).all()
                p[b]=z['prediction']
            seen[b]+=1;count+=1
        assert (seen==1).all()
        ref=e.reference_parts(scheme);pred=.25*ref['v9']+.75*p
        old,new=rmse(e.y,ref['v14']),rmse(e.y,pred)
        m={'v14_rmse':old,'candidate_rmse':new,'gain':old-new,'relative_gain':(old-new)/old}
        assert all(abs(v-decision['metrics'][str(scheme)][k])<1e-6 for k,v in m.items())
        metrics[str(scheme)]=m;bases[scheme]=ref['v14'];candidates[scheme]=pred
    rng=np.random.default_rng(8319);boot=[]
    for _ in range(5000):
        rows=rng.integers(0,len(e.y),len(e.y))
        boot.append(np.mean([rmse(e.y[rows],bases[s][rows])-rmse(e.y[rows],candidates[s][rows]) for s in CONFIRM[:3]]))
    ci=np.quantile(boot,[.025,.975])
    assert np.max(np.abs(ci-np.array(decision['uncertainty']['ci95_INR'])))<1e-6
    mean=float(np.mean([metrics[str(s)]['relative_gain'] for s in CONFIRM[:3]]))
    positive=bool(all(metrics[str(s)]['gain']>0 for s in CONFIRM[:3]) and ci[0]>0 and
                  all(metrics[s]['relative_gain']>=-.01 for s in ['city','strat:27421']))
    original=bool(positive and mean>=.002)
    assert original==decision['original_confirmation_gate_passed'] and positive==decision['descriptive_positive_robustness']
    target=ROOT/'delivery'
    initial=json.loads((target/'initial_export.json').read_text())
    assert sha(target/'full_predictions.npz')==initial['full_predictions_sha256']
    with np.load(target/'full_predictions.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(.5*z['v7']+.5*z['base'],z['v9'])
        np.testing.assert_array_equal(.25*z['v9']+.75*z['p35'],z['v14'])
        pred=.25*z['v9']+.75*z['sc38']
        np.testing.assert_array_equal(pred,z['candidate'])
        for version in ['v9','v14']:
            assert np.max(np.abs(z[version]-pd.read_csv(ROOT/f'reference/scored_{version}.csv')['Annual Turnover']))<.01
    fresh=json.loads((target/'fresh_execution/fresh_execution_checks.json').read_text())
    assert fresh['status']=='PASS' and fresh['maximum_final_difference_INR']<.01
    for path in [target/('EXPLORATORY_'+STEM+'.csv'),target/'fresh_execution'/(STEM+'.csv')]:
        frame=pd.read_csv(path)
        assert frame.shape==(500,2) and frame.columns.tolist()==['Registration Number','Annual Turnover']
        assert frame['Registration Number'].equals(e.test['Registration Number'])
        assert np.isfinite(frame['Annual Turnover']).all() and (frame['Annual Turnover']>=0).all()
        assert np.max(np.abs(frame['Annual Turnover']-pred))<.01
    import nbformat
    book=nbformat.read(target/('EXPLORATORY_'+STEM+'.ipynb'),as_version=4)
    assert 'exploratory' in book.cells[0].source.lower()
    assert all(c.execution_count is not None for c in book.cells if c.cell_type=='code')
    assert not [o for c in book.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.get('output_type')=='error']
    for ext in ['csv','ipynb']:assert sha(target/f'KEEP_V14_{STEM}.{ext}')==sha(ROOT/f'reference/scored_v14.{ext}')
    result={'status':'PASS','complete_candidate_confirmation_folds':count,'locked_candidate_weight':1.,
        'all_five_schemes_independently_reconstructed':True,'bootstrap_independently_reconstructed':True,
        'original_V22_gate_unchanged_and_failed':True,'original_confirmation_gate_passed':original,
        'descriptive_positive_robustness':positive,'mean_random_relative_gain':mean,
        'conditional_gain_CI95_INR':ci.tolist(),'fresh_submission_notebook_verified':True,
        'CSV_round_trip_and_formula_verified':True,'scored_V14_artifacts_preserved':True,
        'hidden_test_score_known':False,'rank_change_established':False,'portal_uploads':0}
    save(target/'independent_review.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
