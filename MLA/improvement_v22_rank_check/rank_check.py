"""One locked exploratory follow-up; original V22 qualification is unchanged."""
import argparse
import hashlib
import json
import os
import pickle
import shutil
import threading
import time
from pathlib import Path
import numpy as np
import pandas as pd
import baseline
import delivery
import modeling
import verify
import workflow
from core import ROOT, STEM, Engine, sha, save, digest, rmse, CONFIRM, validate_csv


class RankEngine(Engine):
    def context(self, name, scheme, fold, a, b):
        return {**super().context(name, scheme, fold, a, b),
                'exploratory_rank_scope_sha256': sha(ROOT/'rank_frozen.json')}


def identities(e):
    frozen = json.loads((ROOT/'source_frozen.json').read_text())
    assert e.sources == frozen['sources'] and e.inputs == frozen['inputs']
    rank = json.loads((ROOT/'rank_frozen.json').read_text())
    assert all(sha(ROOT/name) == want for name,want in rank['supplement_sources'].items())
    assert rank['original_sources'] == e.sources and rank['inputs'] == e.inputs
    assert json.loads((ROOT/'rank_protocol.json').read_text())['candidate'] == {
        'name':'P35-SC38','route':'P35-replacement','weight':1.,'effective_new_share':.75}
    assert modeling.runtime() == json.loads((ROOT/'resolved_models.json').read_text())['runtime']


def confirm(e):
    verify.reference_checks(e)
    verify.feature_checks(e)
    a,b = e.splits(42)[0]
    refs = e.reference_parts(42)
    candidate_path = ROOT/'original_results/cache/P35-SC38_42_fold0.npz'
    candidate_meta = json.loads(candidate_path.with_suffix('.json').read_text())
    assert sha(candidate_path) == candidate_meta['sha256']
    assert candidate_meta['context'] == Engine.context(e,'P35-SC38',42,0,a,b)
    with np.load(candidate_path,allow_pickle=False) as z: original_candidate = z['prediction'].copy()
    receipt = {}
    for name, expected in [('BASE',refs['t9'][b]),('P35_8',refs['p35'][b]),('P35-SC38',original_candidate)]:
        e.budget.require(60)
        fitted = modeling.fit(e.train.iloc[a],e.y[a],name)
        try:
            pred = modeling.predict(fitted,e.train.iloc[b])
            difference = float(np.max(np.abs(pred-expected)))
            assert difference < .01, (name,difference)
            receipt[name] = {'maximum_difference_INR':difference,'metadata':fitted['target_metadata']}
        finally: modeling.release(fitted)
    save(e.out/'control_pilot.json',{'status':'PASS','original_engineering_admission_reverified':True,
         'current_runtime':modeling.runtime(),'controls':receipt,'rank_scope_sha256':sha(ROOT/'rank_frozen.json')})
    print('LIVE CONTROL AND LOCKED CANDIDATE REPRODUCTION PASS',flush=True)
    metrics,bases,predictions = {},{},{}
    segments,folds,concentrations = [],[],[]
    for scheme in CONFIRM:
        parts = e.reference_parts(scheme)
        p,_,fit = e.component('P35-SC38',scheme)
        candidate = .25*parts['v9']+.75*p
        old,new = rmse(e.y,parts['v14']),rmse(e.y,candidate)
        metrics[str(scheme)] = {'v14_rmse':old,'candidate_rmse':new,'gain':old-new,'relative_gain':(old-new)/old}
        bases[scheme],predictions[scheme] = parts['v14'],candidate
        seg,fld,con = workflow.diagnostics(e,scheme,parts['v14'],candidate,'LOCKED_P35_SC38@1')
        segments.extend(seg);folds.extend(fld);concentrations.append(con)
        save(e.out/'confirmation_progress.json',{'metrics':metrics,'fixed_choice':True,
             'rank_scope_sha256':sha(ROOT/'rank_frozen.json')})
        print(json.dumps({'CONFIRMATION':str(scheme),**metrics[str(scheme)]}),flush=True)
    interval = workflow.uncertainty(e.y,bases,predictions)
    original_confirm_gate = workflow.qualifies(metrics,interval)
    positive_robustness = bool(all(metrics[str(s)]['gain']>0 for s in CONFIRM[:3])
        and interval['ci95_INR'][0]>0 and all(metrics[s]['relative_gain']>=-.01 for s in ['city','strat:27421']))
    decision = {'complete':True,'qualified':False,'original_V22_development_gate_passed':False,
        'original_confirmation_gate_passed':original_confirm_gate,
        'descriptive_positive_robustness':positive_robustness,
        'choice':json.loads((ROOT/'rank_protocol.json').read_text())['candidate'],
        'metrics':metrics,'uncertainty':interval,
        'mean_random_relative_gain':float(np.mean([metrics[str(s)]['relative_gain'] for s in CONFIRM[:3]])),
        'decision':'Exploratory evidence only. Original V22 remains unqualified; V14 remains best observed until a lower actual portal score is returned.',
        'hidden_test_score_known':False,'rank_change_established':False,'portal_uploads':0}
    save(e.out/'decision.json',decision)
    pd.DataFrame([{'scheme':s,**m} for s,m in metrics.items()]).to_csv(e.out/'confirmation.csv',index=False)
    pd.DataFrame(segments).to_csv(e.out/'confirmation_segments.csv',index=False)
    pd.DataFrame(folds).to_csv(e.out/'confirmation_folds.csv',index=False)
    save(e.out/'confirmation_concentration.json',concentrations)
    print(json.dumps({'EXPLORATORY_CONFIRMATION_COMPLETE':True,'descriptive_positive_robustness':positive_robustness,
                      'mean_random_relative_gain':decision['mean_random_relative_gain']}),flush=True)


def export(e):
    from nbclient import NotebookClient
    import nbformat
    decision = json.loads((e.out/'decision.json').read_text())
    assert decision['complete'] and decision['choice']['weight'] == 1.
    target = ROOT/'delivery';target.mkdir(exist_ok=True)
    with (ROOT/'reference/verified_v7.pkl').open('rb') as handle: v7 = baseline.predict(pickle.load(handle),e.test)
    assert np.max(np.abs(v7-pd.read_csv(ROOT/'reference/scored_v7.csv')['Annual Turnover'])) < .01
    parts,metadata = {},{}
    for name in ['BASE','P35_8','P35-SC38']:
        e.budget.require(60)
        fitted = modeling.fit(e.train,e.y,name)
        try: parts[name],metadata[name] = modeling.predict(fitted,e.test),fitted['target_metadata']
        finally: modeling.release(fitted)
    v9 = .5*v7+.5*parts['BASE']
    v14 = .25*v9+.75*parts['P35_8']
    for version,p in [('v9',v9),('v14',v14)]:
        expected = pd.read_csv(ROOT/f'reference/scored_{version}.csv')['Annual Turnover'].to_numpy(float)
        assert float(np.max(np.abs(p-expected))) < .01
    prediction = .25*v9+.75*parts['P35-SC38']
    np.savez_compressed(target/'full_predictions.npz',v7=v7,base=parts['BASE'],p35=parts['P35_8'],
                        sc38=parts['P35-SC38'],v9=v9,v14=v14,candidate=prediction)
    frame = pd.DataFrame({'Registration Number':e.test['Registration Number'],'Annual Turnover':prediction})
    validate_csv(frame,e.test);verify.csv_checks(e,prediction)
    frame.to_csv(target/(STEM+'.csv'),index=False)
    save(target/'initial_export.json',{'status':'PASS','exploratory':True,'qualified_under_original_plan':False,
         'metadata':metadata,'inputs':e.inputs,'full_predictions_sha256':sha(target/'full_predictions.npz'),'portal_uploads':0})
    book,path = delivery.build_notebook(e,{'choices':[decision['choice']]},target)
    book.cells[0].source = book.cells[0].source.replace('# V22 qualified turnover model','# V22 exploratory rank-check candidate')
    book.cells[0].source += ('\n\n**Exploratory candidate. The original V22 development gate was not met. '
        'This notebook verifies reproducibility, not a leaderboard gain. No actual portal score exists for this candidate. '
        'Keep the scored V14 CSV and matching notebook.**')
    book.metadata['colab']['name'] = 'EXPLORATORY_'+STEM+'.ipynb'
    fresh = target/'fresh_execution';fresh.mkdir(exist_ok=True)
    keys = ['MLA_DATA_DIR','MLA_V22_OUTPUT_DIR','MLA_AUTOMATED_RUN']
    previous = {k:os.environ.get(k) for k in keys}
    os.environ.update(MLA_DATA_DIR=str(e.data),MLA_V22_OUTPUT_DIR=str(fresh),MLA_AUTOMATED_RUN='1')
    try:
        NotebookClient(book,timeout=max(1,int(e.budget.remaining('export'))-120),kernel_name='python3',
                       resources={'metadata':{'path':str(ROOT)}}).execute()
    finally:
        for k,v in previous.items():
            if v is None: os.environ.pop(k,None)
            else: os.environ[k] = v
    nbformat.write(book,path)
    checks = json.loads((fresh/'fresh_execution_checks.json').read_text())
    assert checks['status']=='PASS' and checks['maximum_final_difference_INR'] < .01
    assert all(c.execution_count is not None for c in book.cells if c.cell_type=='code')
    assert not [o for c in book.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.get('output_type')=='error']
    for suffix in ['.csv','.ipynb']:
        (target/(STEM+suffix)).rename(target/('EXPLORATORY_'+STEM+suffix))
        shutil.copy2(ROOT/'reference'/('scored_v14'+suffix),target/('KEEP_V14_'+STEM+suffix))
    for p in fresh.glob('v22_fresh_*'):
        assert p.is_dir() and p.parent==fresh
        shutil.rmtree(p)
    save(target/'exploratory_export.json',{'status':'PASS','fresh_notebook_executed':True,
         'maximum_CSV_difference_INR':checks['maximum_final_difference_INR'],
         'original_V22_qualification':False,'portal_score':None,'portal_uploads':0})
    print('EXPLORATORY FRESH NOTEBOOK AND CSV REPRODUCTION PASS',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['confirm','export'])
    p.add_argument('--data-dir');p.add_argument('--output-dir');args=p.parse_args()
    e=RankEngine(args.data_dir,args.output_dir)
    stop=threading.Event()
    def heartbeat():
        while not stop.wait(15): save(e.out/'runtime_heartbeat.json',{'stage':args.stage,'epoch':time.time(),'status':'running'})
    worker=threading.Thread(target=heartbeat,daemon=True);worker.start()
    save(e.out/'runtime_heartbeat.json',{'stage':args.stage,'epoch':time.time(),'status':'running'})
    try:
        with e.budget.charge(args.stage):
            identities(e)
            globals()[args.stage](e)
        save(e.out/'stage_status.json',{'stage':args.stage,'status':'complete'})
    except Exception as error:
        save(e.out/'stage_status.json',{'stage':args.stage,'status':'failed','error_type':type(error).__name__,'reason':str(error)[:1000]})
        raise
    finally:
        stop.set();worker.join(timeout=2)
        save(e.out/'runtime_heartbeat.json',{'stage':args.stage,'epoch':time.time(),'status':'finished'})


if __name__=='__main__':main()
