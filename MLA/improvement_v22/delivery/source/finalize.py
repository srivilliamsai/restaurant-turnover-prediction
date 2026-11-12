"""Verify collected artifacts and create the final local delivery without fitting models."""
import argparse,json,shutil,time,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
from core import ROOT,STEM,Engine,sha,save,validate_csv
import delivery

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--allow-incomplete',action='store_true')
    args=parser.parse_args()
    collected=ROOT/'colab_results'
    status=json.loads((ROOT/'transport_run/status.json').read_text())
    complete=status['phase']=='complete'
    if not complete and not args.allow_incomplete:raise RuntimeError('Remote run is incomplete; inspect logs or explicitly package incomplete evidence.')
    for folder in ['results','delivery']:
        source=collected/folder
        if source.exists():shutil.copytree(source,ROOT/folder,dirs_exist_ok=True)
    profile=collected/'resolved_models.json'
    if profile.exists():shutil.copy2(profile,ROOT/profile.name)
    e=Engine()
    original=(ROOT/'reference/scored_v14.csv')
    assert sha(original)=='39d49f79beb8549d04ffc78d134aad2407dc5e73914dc3c30e8bb162a3c0ba05'
    decision_path=e.out/'decision.json'
    decision=json.loads(decision_path.read_text()) if decision_path.exists() else {
        'complete':False,'qualified':False,'choices':[],'finalists':[],
        'decision':'Retain V14: V22 stopped before a complete development/confirmation decision.','portal_uploads':0}
    target=ROOT/'delivery'
    target.mkdir(exist_ok=True)
    if not (e.out/'pilot.json').exists():
        progress=e.out/'pilot_progress.json'
        save(e.out/'pilot.json',{'status':'INCOMPLETE','settings':json.loads(progress.read_text()) if progress.exists() else {},
                                 'screen_settings':[],'sources':e.sources,'inputs':e.inputs})
    if decision['qualified']:
        checks_path=target/'fresh_execution/fresh_execution_checks.json'
        checks=json.loads(checks_path.read_text()) if checks_path.exists() else {}
        verified=complete and checks.get('status')=='PASS' and checks.get('maximum_final_difference_INR',1e99)<.01
        if verified:
            import nbformat
            for path in [target/(STEM+'.csv'),target/'fresh_execution'/(STEM+'.csv')]:validate_csv(pd.read_csv(path),e.test)
            a=pd.read_csv(target/(STEM+'.csv'))['Annual Turnover'].to_numpy(float)
            b=pd.read_csv(target/'fresh_execution'/(STEM+'.csv'))['Annual Turnover'].to_numpy(float)
            assert float(np.max(np.abs(a-b)))<.01
            book=nbformat.read(target/(STEM+'.ipynb'),as_version=4)
            assert all(c.execution_count is not None for c in book.cells if c.cell_type=='code')
            assert not [o for c in book.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.get('output_type')=='error']
        else:
            save(e.out/'confirmation_decision_before_incomplete_export.json',decision)
            decision.update(complete=False,qualified=False,choices=[],decision='Retain V14: confirmation finalist did not complete independent delivery verification.')
            for suffix in ['.csv','.ipynb']:
                path=target/(STEM+suffix)
                if path.exists():path.rename(target/('UNVERIFIED_'+STEM+suffix))
    if not decision['qualified']:
        for suffix in ['.csv','.ipynb']:
            shutil.copy2(ROOT/'reference'/('scored_v14'+suffix),target/('KEEP_V14_'+STEM+suffix))
            assert sha(ROOT/'reference'/('scored_v14'+suffix))==sha(target/('KEEP_V14_'+STEM+suffix))
    save(decision_path,decision)
    ledger=e.budget.state()
    ledger['run_complete']=complete
    ledger['finalized_epoch']=time.time()
    save(e.budget.path,ledger)
    release=json.loads((ROOT/'transport_run/runtime_release.json').read_text())
    if complete:
        assert release['release_command_succeeded'], 'Owned runtime release is unverified'
        account=json.loads((ROOT/'account_release_check.json').read_text())
        assert account['status']=='PASS' and account['live_active_assignments']==0
        save(target/'account_release_check.json',account)
    save(target/'runtime_release.json',release)
    save(target/'execution_summary.json',{'status':'COMPLETE' if complete else 'INCOMPLETE',
                                          'experiment_complete':complete,'qualified':decision['qualified'],
                                          'source_identity_unchanged':e.sources==json.loads((ROOT/'source_frozen.json').read_text())['sources'],
                                          'owned_runtime_released':release['release_command_succeeded'],
                                          'scored_v14_csv_sha256':sha(original),'portal_uploads':0})
    assert json.loads((target/'execution_summary.json').read_text())['source_identity_unchanged']
    # Keep an executable research layout beside the readable source inventory.
    # The original model sources and their frozen identities are not changed.
    for path in ROOT.glob('*.py'):shutil.copy2(path,target/path.name)
    for name in ['city_map.json','v5_contract.json','source_frozen.json']:
        shutil.copy2(ROOT/name,target/name)
    for name in ['run.log','setup.log']:
        if (collected/name).exists():shutil.copy2(collected/name,target/name)
    if complete:
        subprocess.run([sys.executable,str(ROOT/'verify_results.py')],check=True)
        review=json.loads((target/'independent_review.json').read_text())
        measured=pd.read_csv(e.out/'screening.csv')
        best=measured.sort_values(['mean_relative_gain','effective_new_share'],ascending=[False,True]).iloc[0].to_dict() if len(measured) else None
        summary={'experiment_complete':True,'qualified':decision['qualified'],'best_development':best,
                 'verified_candidate_fold_outputs':review['complete_fold_outputs_verified'],
                 'verified_blend_comparisons':review['independently_recomputed_blends'],
                 'portal_uploads':0,'best_observed_portal_version':'V14',
                 'best_observed_portal_rmse_INR':11736110.7877709,'owned_runtime_released':True,
                 'decision':decision['decision']}
        save(target/'result_summary.json',summary)
        lines=['# V22 result','',decision['decision'],'',
               'Best observed portal RMSE remains **11,736,110.7877709 (V14)**. No V22 portal upload occurred.','',
               f"{review['complete_fold_outputs_verified']} candidate fold outputs and {review['independently_recomputed_blends']} fixed comparisons were independently reconstructed from hashed predictions. All 3,493 original labels remain in evaluation.",'']
        if best:
            lines += [f"Best development setting: **{best['name']} at {best['weight']:.0%} replacement weight**; mean relative local gain **{best['mean_relative_gain']:.6%}**, versus the unchanged **0.20%** floor.",
                      f"Gains on assignments 42/71: INR {best['gain_42']:,.2f} / {best['gain_71']:,.2f}.",'']
        lines += ['P35-SC38 removes exactly Service and Comfortablility from the original 40-column V14 view. It uses the original TabPFN-3.5 checkpoint, eight estimators, seed 10242 and default model settings. V9 retains 25% of every blend. Categorical vocabularies fit on training folds only.','',
                  'The owned free T4 was released and the selected account reported zero active assignments. The four-hour ledger includes implementation and setup; no prior ledger was reset. Local validation is not a portal score. See V22_Report.md and independent_review.json.']
        (target/'V22_Result_Summary.md').write_text('\n'.join(lines)+'\n')
        save(ROOT/'execution_status.json',summary)
    delivery.package(e,decision,target)
    if not complete:
        path=target/'V22_Report.md'
        path.write_text(path.read_text()+'\n## Incomplete execution\n\nThis run stopped before every required stage completed. See execution_summary.json, stage_status.json and preserved run.log. No unmeasured comparison is claimed.\n')
        # Repackage with the explicit incomplete-execution paragraph included.
        # package() regenerates the report, so update archive and manifest directly.
        manifest=json.loads((target/'manifest.json').read_text())
        manifest['files']['V22_Report.md']=sha(path)
        save(target/'manifest.json',manifest)
        import zipfile
        archive=ROOT/'V22_Delivery.zip'
        replacement=archive.with_suffix('.replacement.zip')
        with zipfile.ZipFile(archive) as old,zipfile.ZipFile(replacement,'w',zipfile.ZIP_DEFLATED) as new:
            for entry in old.infolist():
                if entry.filename not in ['V22_Report.md','manifest.json']:new.writestr(entry,old.read(entry.filename))
            new.write(path,'V22_Report.md')
            new.write(target/'manifest.json','manifest.json')
        replacement.replace(archive)
    print(json.dumps({'experiment_complete':complete,'qualified':decision['qualified'],'owned_runtime_released':release['release_command_succeeded'],
                      'delivery':str(ROOT/'V22_Delivery.zip'),'report':str(target/'V22_Report.md')}),flush=True)

if __name__=='__main__':main()


