"""Verify collected artifacts and create the final local delivery without fitting models."""
import argparse,json,shutil,time
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
    original=(ROOT/'reference/scored_v9.csv')
    assert sha(original)=='5a384dc3d8bf58c4e29f0d6cac646aa33afe22bdcdd186cccd9758342769bbd4'
    decision_path=e.out/'decision.json'
    decision=json.loads(decision_path.read_text()) if decision_path.exists() else {
        'complete':False,'qualified':False,'choices':[],'finalists':[],
        'decision':'Retain V9: V13 stopped before a complete development/confirmation decision.','portal_uploads':0}
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
            decision.update(complete=False,qualified=False,choices=[],decision='Retain V9: confirmation finalist did not complete independent delivery verification.')
            for suffix in ['.csv','.ipynb']:
                path=target/(STEM+suffix)
                if path.exists():path.rename(target/('UNVERIFIED_'+STEM+suffix))
    if not decision['qualified']:
        for suffix in ['.csv','.ipynb']:
            shutil.copy2(ROOT/'reference'/('scored_v9'+suffix),target/('KEEP_V9_'+STEM+suffix))
            assert sha(ROOT/'reference'/('scored_v9'+suffix))==sha(target/('KEEP_V9_'+STEM+suffix))
    save(decision_path,decision)
    ledger=e.budget.state()
    ledger['run_complete']=complete
    ledger['finalized_epoch']=time.time()
    save(e.budget.path,ledger)
    release=json.loads((ROOT/'transport_run/runtime_release.json').read_text())
    save(target/'runtime_release.json',release)
    save(target/'execution_summary.json',{'status':'COMPLETE' if complete else 'INCOMPLETE',
                                          'experiment_complete':complete,'qualified':decision['qualified'],
                                          'source_identity_unchanged':e.sources==json.loads((ROOT/'source_frozen.json').read_text())['sources'],
                                          'owned_runtime_released':release['release_command_succeeded'],
                                          'scored_v9_csv_sha256':sha(original),'portal_uploads':0})
    assert json.loads((target/'execution_summary.json').read_text())['source_identity_unchanged']
    # Keep an executable research layout beside the readable source inventory.
    # The original model sources and their frozen identities are not changed.
    for path in ROOT.glob('*.py'):shutil.copy2(path,target/path.name)
    for name in ['city_map.json','v5_contract.json','source_frozen.json']:
        shutil.copy2(ROOT/name,target/name)
    for name in ['run.log','setup.log']:
        if (collected/name).exists():shutil.copy2(collected/name,target/name)
    delivery.package(e,decision,target)
    if not complete:
        path=target/'V13_Report.md'
        path.write_text(path.read_text()+'\n## Incomplete execution\n\nThis run stopped before every required stage completed. See execution_summary.json, stage_status.json and preserved run.log. No unmeasured comparison is claimed.\n')
        # Repackage with the explicit incomplete-execution paragraph included.
        # package() regenerates the report, so update archive and manifest directly.
        manifest=json.loads((target/'manifest.json').read_text())
        manifest['files']['V13_Report.md']=sha(path)
        save(target/'manifest.json',manifest)
        import zipfile
        archive=ROOT/'V13_Delivery.zip'
        replacement=archive.with_suffix('.replacement.zip')
        with zipfile.ZipFile(archive) as old,zipfile.ZipFile(replacement,'w',zipfile.ZIP_DEFLATED) as new:
            for entry in old.infolist():
                if entry.filename not in ['V13_Report.md','manifest.json']:new.writestr(entry,old.read(entry.filename))
            new.write(path,'V13_Report.md')
            new.write(target/'manifest.json','manifest.json')
        replacement.replace(archive)
    print(json.dumps({'experiment_complete':complete,'qualified':decision['qualified'],'owned_runtime_released':release['release_command_succeeded'],
                      'delivery':str(ROOT/'V13_Delivery.zip'),'report':str(target/'V13_Report.md')}),flush=True)

if __name__=='__main__':main()

