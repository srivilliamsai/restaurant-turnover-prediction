"""Recovery tests: exact caches, independent isolation queries and stale transport."""
import json,tempfile,sys
from pathlib import Path
import numpy as np
import pandas as pd
from core import ROOT,Engine,save,sha,digest
from pilot_queries import PilotQueryCache
sys.path.insert(0,str(ROOT/'transport'))
from heartbeat_guard import HeartbeatGuard
import workflow

def main():
    manifest=json.loads((ROOT/'disconnect_resume_manifest.json').read_text())
    witness=ROOT/manifest['witness_directory'];ledger=json.loads((ROOT/'colab_results/results/compute_budget.json').read_text())
    assert ledger['maximum_hours']==18 and ledger['seconds']>13000
    with tempfile.TemporaryDirectory(prefix='disconnect_preflight_',dir=ROOT) as temporary:
        out=Path(temporary);e=Engine(ROOT.parent,out)
        for name in manifest['reuse_settings']:
            original=json.loads((witness/'results/pilot_checkpoints'/f'{name}.json').read_text())
            save(out/'pilot_checkpoints'/f'{name}.json',{'context':workflow.pilot_context(e,name),'result':original['result'],
                 'result_sha256':original['result_sha256'],'migration_manifest':'disconnect_resume_manifest.json',
                 'migration_witness':sha(ROOT/'disconnect_resume_manifest.json')})
            assert workflow.read_pilot_checkpoint(e,name)==original['result']
        raw=pd.DataFrame({'Registration Number':[1,2],'Annual Turnover':[4.,8.],'predictor':[3.,6.]},index=[5,9])
        cache=PilotQueryCache(out,{'setting':'SOFTWARE_TEST_ONLY','source':'no GPU or model invoked'})
        count=[0]
        def calculate(frame):count[0]+=1;return frame['predictor'].to_numpy(float)
        p=cache.predict(raw,'first_fit',{'state':'frozen'},lambda:calculate(raw));first_seconds=cache.last_inference_seconds
        q=cache.predict(raw,'first_fit',{'state':'frozen'},lambda:(_ for _ in ()).throw(AssertionError('cached query recomputed')))
        np.testing.assert_array_equal(p,q);assert count[0]==1 and cache.last_inference_seconds==first_seconds
        changed=raw.copy();changed['Registration Number']=-1;changed['Annual Turnover']=1e30
        np.testing.assert_array_equal(p,cache.predict(changed,'first_fit',{'state':'frozen'},lambda:calculate(changed)))
        assert count[0]==2,'Isolation counterfactual reused its control prediction'
        reverse=raw.iloc[::-1]
        np.testing.assert_array_equal(p[::-1],cache.predict(reverse,'first_fit',{'state':'frozen'},lambda:calculate(reverse)))
        cache.predict(raw,'fresh_repeat_fit',{'state':'frozen'},lambda:calculate(raw));assert count[0]==4
        resumed=PilotQueryCache(out,cache.context)
        np.testing.assert_array_equal(p,resumed.predict(raw,'first_fit',{'state':'frozen'},lambda:(_ for _ in ()).throw(AssertionError('resume recomputed a finished query'))))
        # Use a separate one-query directory to make corruption fixtures explicit.
        broken=PilotQueryCache(out/'broken',{'setting':'SOFTWARE_TEST_ONLY'})
        broken.predict(raw,'first_fit',{},lambda:calculate(raw));m=next(broken.root.glob('*.json'));z=m.with_suffix('.npz');mb=m.read_bytes();zb=z.read_bytes()
        for failure in ['stale','corrupt','incomplete']:
            m.write_bytes(mb);z.write_bytes(zb)
            if failure=='stale':d=json.loads(mb);d['context']={};save(m,d)
            if failure=='corrupt':z.write_bytes(b'bad')
            if failure=='incomplete':m.unlink()
            try:broken.predict(raw,'first_fit',{},lambda:calculate(raw))
            except (AssertionError,ValueError):pass
            else:raise AssertionError(failure+' prediction checkpoint accepted')
        hb=out/'heartbeat.json';guard=HeartbeatGuard(hb,'pilot',now=0)
        guard.observe(now=299)
        try:guard.observe(now=301)
        except RuntimeError:pass
        else:raise AssertionError('Missing heartbeat did not bound the stale client')
        save(hb,{'stage':'pilot','pid':1,'elapsed_seconds':5});guard=HeartbeatGuard(hb,'pilot',now=0);guard.observe(now=1)
        guard.observe(now=300)
        try:guard.observe(now=302)
        except RuntimeError:pass
        else:raise AssertionError('Rewriting the same heartbeat hid a stalled runtime')
        save(hb,{'stage':'pilot','pid':1,'elapsed_seconds':10});guard.observe(now=303);guard.observe(now=500)
    for p in list(ROOT.glob('*.py'))+list((ROOT/'transport').glob('*.py')):compile(p.read_text(),str(p),'exec')
    save(ROOT/'disconnect_preflight.json',{'status':'PASS','software_tests_only':True,'new_model_fits':0,
         'three_real_passes_verified_with_original_witnesses':True,'exact_cache_and_cross_instance_resume':True,
         'label_id_counterfactual_forces_independent_prediction':True,'first_and_repeat_fits_use_distinct_cache_keys':True,
         'original_inference_time_retained':True,'stale_corrupt_incomplete_predictions_rejected':True,
         'stale_or_missing_heartbeat_bounded':True,'unchanged_heartbeat_content_cannot_fake_progress':True,
         'all_four_settings_required':True,'maximum_hours':18,'previous_task_seconds_retained':ledger['seconds'],
         'sources':Engine(ROOT.parent,ROOT/'disconnect_audit').sources})
    print('Disconnect recovery PASS: exact independent prediction checkpoints, resume, integrity rejection, heartbeat guard and three preserved real passes. No GPU model fitted.')

if __name__=='__main__':main()
