"""Software-only checks for the user-approved all-four, eighteen-hour resume."""
import json,subprocess,tempfile
from pathlib import Path
from unittest.mock import patch
from core import ROOT,Engine,save,sha,digest
import workflow,features,modeling

def main():
    manifest=json.loads((ROOT/'all_settings_resume_manifest.json').read_text())
    witness=ROOT/manifest['witness_directory']
    profile=json.loads((ROOT/'resolved_models.json').read_text())
    state=json.loads((ROOT/'colab_results/results/compute_budget.json').read_text())
    assert state['seconds']>=7170 and json.loads((ROOT/'authorization.json').read_text())['maximum_hours']==18
    for name,want in manifest['original_sources'].items():
        if name not in ['core.py','workflow.py']:assert sha(ROOT/name)==want,name
    with tempfile.TemporaryDirectory(prefix='all_four_preflight_',dir=ROOT) as temporary:
        e=Engine(ROOT.parent,Path(temporary));workflow.audit(e)
        for name in manifest['reuse_settings']:
            original=json.loads((witness/'results/pilot_checkpoints'/f'{name}.json').read_text())
            value={'context':workflow.pilot_context(e,name),'result':original['result'],'result_sha256':original['result_sha256'],
                   'migration_manifest':'all_settings_resume_manifest.json','migration_witness':sha(ROOT/'all_settings_resume_manifest.json')}
            save(e.out/'pilot_checkpoints'/f'{name}.json',value)
            assert workflow.read_pilot_checkpoint(e,name)==original['result']
        # A timed-out fourth worker must block development, keeping all three
        # true passes. No model or GPU resource is invoked by this test.
        class TimedOutWorker:
            def __init__(self,argv,**kwargs):self.stopped=False;assert argv[-3]=='T4x16'
            def wait(self,timeout=None):
                if not self.stopped:raise subprocess.TimeoutExpired('synthetic pilot worker',timeout)
                return -15
            def terminate(self):self.stopped=True
            def kill(self):self.stopped=True
            def poll(self):return -15 if self.stopped else None
        state['maximum_hours']=18;save(e.cache.ledger,state)
        with patch.object(modeling,'runtime',return_value=profile['runtime']),patch.object(workflow.subprocess,'Popen',TimedOutWorker):
            try:workflow.pilot(e)
            except RuntimeError as error:assert 'All four settings are required' in str(error)
            else:raise AssertionError('Incomplete fourth setting silently omitted')
        partial=json.loads((e.out/'pilot.json').read_text())
        assert partial['status']=='INCOMPLETE_ALL_SETTINGS' and partial['unfinished_settings']==['T4x16']
        try:workflow.screen(e)
        except AssertionError:pass
        else:raise AssertionError('Development admitted an incomplete pilot')
        # All-complete branch is synthetic software coverage only.
        fake={'status':'PASS','fold_seconds':300.,'software_test_only':True}
        save(e.out/'pilot_checkpoints/T4x16.json',{'context':workflow.pilot_context(e,'T4x16'),'result':fake,'result_sha256':digest(fake)})
        with patch.object(modeling,'runtime',return_value=profile['runtime']),patch.object(workflow.subprocess,'Popen',side_effect=AssertionError('Verified pass was refitted')):
            workflow.pilot(e)
        pilot=json.loads((e.out/'pilot.json').read_text())
        assert pilot['status']=='PASS' and pilot['screen_settings']==list(features.SPECS) and not pilot['budget_skipped_families']
        def control_prediction(name,scheme):
            v7,t9=e.reference_parts(scheme)
            return t9 if name.startswith('T') else .5*v7+.5*t9
        e.component=control_prediction;workflow.screen(e);workflow.confirm(e)
        lock=json.loads((e.out/'candidate_lock.json').read_text())
        assert lock['settings']==4 and lock['configurations']==16 and lock['choices']==[]
        decision=json.loads((e.out/'decision.json').read_text());assert decision['complete'] and not decision['qualified']
        assert e.cache.state()['seconds']>=state['seconds']
    for p in list(ROOT.glob('*.py'))+list((ROOT/'transport').glob('*.py')):compile(p.read_text(),str(p),'exec')
    save(ROOT/'all_settings_preflight.json',{'status':'PASS','software_tests_only':True,'new_model_fits':0,
         'three_real_passes_verified_with_original_witnesses':True,'numerical_sources_unchanged':True,
         'incomplete_fourth_setting_blocks_development':True,'all_four_and_sixteen_required':True,
         'completed_checks_never_refitted':True,'actual_timeout_and_no_finalist_paths':True,
         'previous_task_seconds_retained':state['seconds'],'maximum_hours':18,'delivery_reserve_seconds':10800,
         'sources':Engine(ROOT.parent,ROOT/'all_four_audit').sources})
    print('PASS: three preserved passes, no silent omission, all four/16 configurations, eighteen-hour retained budget; no new model fits.')

if __name__=='__main__':main()
