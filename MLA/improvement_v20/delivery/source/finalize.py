"""Reconstruct measured V20 results, verify release, and package exact evidence."""
import json,re,shutil,time,zipfile
from pathlib import Path
import pandas as pd
from core import ROOT,STEM,Engine,save,sha

def main():
    collected=ROOT/'colab_results'
    for folder in ['results','delivery']:shutil.copytree(collected/folder,ROOT/folder,dirs_exist_ok=True)
    for name in ['resolved_models.json','setup.log','run.log']:shutil.copy2(collected/name,ROOT/name)
    e=Engine();frozen=json.loads((ROOT/'source_frozen.json').read_text())
    assert e.sources==frozen['sources'] and e.inputs==frozen['inputs']
    import verify_results
    verify_results.main()
    decision=json.loads((e.out/'decision.json').read_text())
    pilot=json.loads((e.out/'pilot.json').read_text())
    rows=pd.read_csv(e.out/'screening.csv').to_dict('records')
    best=max(rows,key=lambda r:r['mean_relative_gain']) if rows else None
    release=json.loads((ROOT/'transport_run/runtime_release.json').read_text())
    assert release['release_command_succeeded'] and release['owned_session']=='v20-free-t4'
    allocation=json.loads((ROOT/'transport_run/allocations.json').read_text())[-1]
    assert release['epoch']>=allocation['requested_epoch']
    account=json.loads((ROOT/'account_release_check.json').read_text())
    assert account['status']=='PASS' and account['live_active_assignments']==0 and account['checked_epoch']>=release['epoch']
    if decision['qualified']:
        import nbformat
        checks=json.loads((ROOT/'delivery/fresh_execution/fresh_execution_checks.json').read_text())
        assert checks['status']=='PASS' and checks['prediction_agreement_passed']
        book=nbformat.read(ROOT/'delivery'/(STEM+'.ipynb'),as_version=4)
        assert all(c.execution_count is not None for c in book.cells if c.cell_type=='code')
        assert not [o for c in book.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.get('output_type')=='error']
    else:
        for suffix in ['.csv','.ipynb']:
            assert sha(ROOT/'delivery'/('KEEP_V14_'+STEM+suffix))==sha(ROOT/'reference'/('scored_v14'+suffix))
    fold_records=[json.loads(f.read_text()) for f in (e.out/'cache').glob('*.json')]
    development_count=sum(m['context']['name'] in __import__('features').SPECS and m['context']['scheme'] in ['42','71'] for m in fold_records)
    status={'status':'COMPLETE','completed_epoch':time.time(),'development_scores_generated':bool(rows),
        'completed_candidate_development_folds':development_count,'fixed_blend_comparisons':len(rows),
        'confirmation_performed':bool(decision.get('finalists')),'qualified':decision['qualified'],
        'maximum_hours':6,'original_started_epoch':json.loads((ROOT/'authorization.json').read_text())['started_epoch'],
        'paid_resources':False,'portal_uploads':0,'independent_reconstruction':'PASS','owned_runtime_released':True,
        'active_assignments_after_release':0}
    save(ROOT/'execution_status.json',status)
    lines=['# V20 result','', '**'+('One candidate qualified for a manual portal trial.' if decision['qualified'] else 'Retain V14.')+'**','',
        'Best observed portal RMSE remains **11,736,110.7877709 (V14)**. No V20 portal upload occurred.','',
        f'{development_count} complete candidate development fold outputs and {len(rows)} fixed blends were independently reconstructed from hashed predictions. All 3,493 original labels remain in evaluation.','']
    if best:
        lines += [f"Best development setting: **{best['name']} at {best['weight']:.0%} new-model weight**, mean relative local gain **{best['mean_relative_gain']:.6%}**. Qualification floor: **0.20%**.",
            f"Gains on assignments 42/71: INR {best['gain_42']:,.2f} / {best['gain_71']:,.2f}.",'']
    lines += ['| Setting | Engineering result | Detail |','|---|---|---|']
    for name,item in pilot['settings'].items():
        detail=item.get('reason','Full-fold, full-data, fresh-fit, reload and query checks completed.').replace('|','/')
        lines.append(f"| {name} | {item['status']} | {detail} |")
    lines += ['',decision['decision'],'',
        'Free T4 released; a fresh selected-account check found zero active assignments. The six-hour elapsed ledger includes setup failures, the pre-accuracy budget-guard repair, interruptions and recovery. Those records are preserved in engineering_repair_evidence/.','',
        'Local validation is not a portal score. Confirmation partitions reuse historically examined labels. See V20_Report.md and the hashed evidence for exact measurements and limitations.']
    (ROOT/'delivery/V20_Result_Summary.md').write_text('\n'.join(lines)+'\n')
    for name in ['execution_status.json','source_frozen.json','authorization.json','deadline_check.json','setup_status.json',
                 'candidate_dependency_locks.json','engineering_setup_history.json','IMPLEMENTATION_PROGRESS.md',
                 'REPRODUCTION.md','planning_manifest.json','evidence_audit.json','tabicl_source_snapshot.zip','local_tests.json']:
        if (ROOT/name).exists():shutil.copy2(ROOT/name,ROOT/'delivery'/name)
    shutil.copy2(ROOT/'account_release_check.json',ROOT/'delivery/account_release_check.json')
    shutil.copy2(ROOT/'transport_run/runtime_release.json',ROOT/'delivery/runtime_release.json')
    if (ROOT/'engineering_repair_evidence').exists():shutil.copytree(ROOT/'engineering_repair_evidence',ROOT/'delivery/engineering_repair_evidence',dirs_exist_ok=True)
    if (ROOT/'planning_original').exists():shutil.copytree(ROOT/'planning_original',ROOT/'delivery/planning_original',dirs_exist_ok=True)
    import delivery
    delivery.package(e,decision,ROOT/'delivery')
    current=ROOT.parent/'CURRENT_BEST.md'
    text=current.read_text()
    text=re.sub(r'\[Latest completed experiment result\]\([^\n]+\)',
                '[Latest completed experiment result]('+str(ROOT/'delivery/V20_Result_Summary.md')+')',text)
    paragraph=f"V20 completed {development_count} candidate development folds and {len(rows)} fixed blends. "
    if best:paragraph+=f"Best mean relative local gain: {best['mean_relative_gain']:.6%} ({best['name']} at {best['weight']:.0%}). "
    paragraph+=decision['decision']+' No V20 portal upload occurred. Independent reconstruction passed and the owned free T4 was released.'
    text+='\n'+paragraph+'\n';current.write_text(text)
    print(json.dumps({'status':'PASS','decision':decision['decision'],'development_folds':development_count,'comparisons':len(rows),'best':best,'artifact':str(ROOT/'V20_Delivery.zip')},indent=2))

if __name__=='__main__':main()
