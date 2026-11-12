"""Exercise pilot resume and timeout branches without fitting any model."""
import json,subprocess,tempfile
from pathlib import Path
from unittest.mock import patch
from core import ROOT,Engine,save,sha,digest
import workflow,features,verify

def main():
    migration=json.loads((ROOT/'pilot_reuse_manifest.json').read_text())
    witness=ROOT/migration['witness_directory']
    original=json.loads((witness/'pilot_progress.json').read_text())['T32']
    for name,want in migration['original_sources'].items():
        if name not in ['core.py','workflow.py']:assert sha(ROOT/name)==want,name
    ledger=json.loads((ROOT/'colab_results/results/compute_budget.json').read_text())
    profile=json.loads((ROOT/'resolved_models.json').read_text())
    assert ledger['seconds']>3600 and ledger['maximum_hours']==12
    with tempfile.TemporaryDirectory(prefix='resume_preflight_',dir=ROOT) as temporary:
        e=Engine(ROOT.parent,Path(temporary));verify.feature_checks(e);workflow.audit(e)
        def checkpoint(name,result,migrate=False):
            value={'context':workflow.pilot_context(e,name),'result':result,'result_sha256':digest(result)}
            if migrate:value['migration_witness']=sha(ROOT/'pilot_reuse_manifest.json')
            save(e.out/'pilot_checkpoints'/f'{name}.json',value);return value
        value=checkpoint('T32',original,True)
        assert workflow.read_pilot_checkpoint(e,'T32')==original
        bad=dict(value);bad['result_sha256']='bad';save(e.out/'pilot_checkpoints/T32.json',bad)
        try:workflow.read_pilot_checkpoint(e,'T32')
        except AssertionError:pass
        else:raise AssertionError('Corrupt pilot pass accepted')
        bad=dict(value);bad['context']={};save(e.out/'pilot_checkpoints/T32.json',bad)
        try:workflow.read_pilot_checkpoint(e,'T32')
        except AssertionError:pass
        else:raise AssertionError('Stale pilot pass accepted')
        checkpoint('T32',original,True)
        for name in ['R1','R4','T4x16']:
            checkpoint(name,{'status':'unavailable','reason':{'type':'SOFTWARE_TEST_ONLY','message':'No GPU model fitted in this preflight.'}})
        save(e.cache.ledger,ledger)
        with patch.object(workflow.modeling,'runtime',return_value=profile['runtime']),patch.object(workflow.subprocess,'Popen',side_effect=AssertionError('Completed pilot checkpoint was refitted')):
            workflow.pilot(e)
        result=json.loads((e.out/'pilot.json').read_text())
        assert result['status']=='NO_FEASIBLE_FAMILY' and result['screen_settings']==[]
        assert e.cache.state()['seconds']>=ledger['seconds']
        for name in ['R1','R4','T4x16']:(e.out/'pilot_checkpoints'/f'{name}.json').unlink()
        calls=[]
        class TimedOutWorker:
            def __init__(self,argv,**kwargs):self.stopped=False;calls.append(argv[-3])
            def wait(self,timeout=None):
                if not self.stopped:raise subprocess.TimeoutExpired('synthetic pilot worker',timeout)
                return -15
            def terminate(self):self.stopped=True
            def kill(self):self.stopped=True
            def poll(self):return -15 if self.stopped else None
        with patch.object(workflow.modeling,'runtime',return_value=profile['runtime']),patch.object(workflow.subprocess,'Popen',TimedOutWorker):
            workflow.pilot(e)
        result=json.loads((e.out/'pilot.json').read_text())
        assert calls==['R1','R4','T4x16'] and result['settings']['T32']==original
        assert all(result['settings'][n]['unfinished'] for n in calls)
        assert result['screen_settings']==[] and not result['selection_uses_validation_rmse']
        workflow.screen(e);workflow.confirm(e)
        decision=json.loads((e.out/'decision.json').read_text());assert decision['complete'] and not decision['qualified']
        assert 'feasibility' in decision['decision']
    for p in list(ROOT.glob('*.py'))+list((ROOT/'transport').glob('*.py')):compile(p.read_text(),str(p),'exec')
    record={'status':'PASS','software_tests_only':True,'new_model_fits':0,'development_scores_generated':False,
            'completed_T32_original_witness_verified':True,'numerical_sources_unchanged':True,
            'stale_and_corrupt_pilot_pass_rejected':True,'completed_settings_never_refitted':True,
            'actual_timeout_handler_and_partial_family_fallback':True,'actual_no_feasible_family_branch':True,
            'previous_task_seconds_retained':ledger['seconds'],'maximum_hours':12,'delivery_reserve_seconds':10800,
            'sources':Engine(ROOT.parent,ROOT/'resume_audit').sources}
    save(ROOT/'resume_preflight.json',record)
    print('Resume, timeout, unchanged numerical contracts and retained budget: PASS; no new model fits.')

if __name__=='__main__':main()
