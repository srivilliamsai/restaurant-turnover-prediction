"""Verify collected completion and build a portable evidence archive; no fits."""
import json
import shutil
import time
import zipfile
from pathlib import Path
from core import ROOT, Engine, sha, save


def main():
    e = Engine()
    summary = json.loads((ROOT/'delivery/result_summary.json').read_text())
    review = json.loads((ROOT/'delivery/independent_review.json').read_text())
    assert summary['experiment_complete'] and review['status'] == 'PASS'
    assert summary['verified_blend_comparisons'] == 3
    assert summary['verified_candidate_fold_outputs'] in [10, 35]
    assert json.loads((ROOT/'transport_run/runtime_release.json').read_text())['release_command_succeeded']
    release = json.loads((ROOT/'account_release_check.json').read_text())
    assert release['status'] == 'PASS' and release['live_active_assignments'] == 0
    assert release['account_identity_sha256'] == e.budget.authorization['account_identity_sha256']
    frozen = json.loads((ROOT/'source_frozen.json').read_text())
    assert e.sources == frozen['sources'] and e.inputs == frozen['inputs']
    original = json.loads((ROOT/'account_change_evidence/previous_account_attempt/authorization.json').read_text())
    for key in ['started_epoch','hard_stop_epoch','training_stop_epoch','engineering_stop_epoch',
                'maximum_hours','delivery_reserve_hours','settings','weights']:
        assert original[key] == e.budget.authorization[key], key
    planning = ROOT/'planning_original'
    planning.mkdir(exist_ok=True)
    planning_manifest = json.loads((ROOT/'planning_manifest.json').read_text())
    historical = dict(planning_manifest)
    historical.pop('relative_base_directory',None)
    for name,want in historical['files_sha256'].items():
        source = ROOT/({'PLAN.md':'PLAN_PROPOSED.md','README.md':'README_RESEARCH.md'}.get(name,name))
        assert sha(source) == want, name
        shutil.copy2(source,planning/name)
    save(planning/'planning_manifest.json',historical)
    planning_manifest['relative_base_directory'] = 'planning_original'
    save(ROOT/'planning_manifest.json',planning_manifest)
    ledger = e.budget.state()
    ledger.update(run_complete=True, finalized_epoch=time.time())
    save(e.budget.path, ledger)
    save(ROOT/'delivery/compute_budget.json', ledger)
    best = summary['best_development']
    lines = ['# V22 complete', '', summary['decision'], '',
             'Best observed portal RMSE remains **11,736,110.7877709 (V14)**. No V22 portal upload occurred.', '',
             f"P35-SC38 passed engineering and completed {review['complete_fold_outputs_verified']} candidate fold outputs and three fixed blend comparisons.",
             f"Best development weight: {best['weight']:.0%} replacement of the P35 share; mean relative gain **{best['mean_relative_gain']:.6%}**, against the unchanged **0.20%** floor.",
             f"Development gains on assignments 42/71: INR {best['gain_42']:,.2f} / {best['gain_71']:,.2f}.", '',
             'The new component removes only Service and Comfortablility from the exact original V14 view. All 3,493 labels are retained; the V9 quarter remains unchanged.', '',
             'The user-requested account change and repeated engineering remain charged to the original four-hour clock. Both the old runtime release and the final selected-account check are recorded. The selected account has zero active assignments.', '',
             '- [Result summary](delivery/V22_Result_Summary.md)',
             '- [Measured report](delivery/V22_Report.md)',
             '- [Independent reconstruction](delivery/independent_review.json)',
             '- [Reproduction instructions](REPRODUCTION.md)',
             '- [Accepted plan](PLAN.md)',
             '- [Original research proposal](PLAN_PROPOSED.md)',
             '- [Internet source review](RESEARCH_SOURCES.md)', '']
    (ROOT/'README.md').write_text('\n'.join(lines))
    progress = ROOT/'IMPLEMENTATION_PROGRESS.md'
    final_progress = ('\nExecution completed under the original clock. '+summary['decision']+
                     f" Best development relative gain {best['mean_relative_gain']:.6%}; all three fixed comparisons and complete folds independently reconstructed. The owned runtime was released and final selected-account verification returned zero active assignments. Final portable archive preserves original inputs, references, complete results and the account-change evidence.\n")
    if final_progress not in progress.read_text():
        with progress.open('a') as handle: handle.write(final_progress)
    target = ROOT/'delivery'
    for name in ['README.md','REPRODUCTION.md','IMPLEMENTATION_PROGRESS.md','model_identity_verification.json',
                 'account_switch.json','execution_status.json','execution_source_manifest.json','planning_manifest.json']:
        shutil.copy2(ROOT/name,target/name)
    for folder in ['account_change_evidence','transport_run','planning_original']:
        shutil.copytree(ROOT/folder,target/folder,dirs_exist_ok=True)
    manifest = json.loads((target/'manifest.json').read_text())
    manifest['compute_budget'] = ledger
    manifest['files'] = {str(p.relative_to(target)):sha(p) for p in target.rglob('*')
                         if p.is_file() and p.name != 'manifest.json'
                         and '__pycache__' not in p.parts and '.partial' not in p.name}
    save(target/'manifest.json',manifest)

    members = {}
    for p in ROOT.iterdir():
        if p.is_file() and p.suffix in ['.py','.json','.md','.npz'] and p.name not in ['archive_manifest.json','archive_verification.json']:
            members['improvement_v22/'+p.name] = p
    for folder in ['results','reference','provenance','delivery','account_change_evidence','transport_run','planning_original']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and 'model_weights' not in p.parts and '.partial' not in p.name and p.suffix != '.safetensors':
                members['improvement_v22/'+str(p.relative_to(ROOT))] = p
    for name in e.inputs:
        members[name] = e.data/name
    hashes = {name:sha(path) for name,path in sorted(members.items())}
    save(ROOT/'archive_manifest.json',{'layout':'Original CSVs beside improvement_v22/; all paths relative to extraction root.',
         'files_sha256':hashes,'source_identity_unchanged':True,'portal_uploads':0})
    members['improvement_v22/archive_manifest.json'] = ROOT/'archive_manifest.json'
    archive = ROOT/'V22_Delivery.zip'
    replacement = archive.with_suffix('.complete.zip')
    with zipfile.ZipFile(replacement,'w',zipfile.ZIP_DEFLATED) as bundle:
        for name,path in sorted(members.items()):
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            bundle.write(path,name)
    with zipfile.ZipFile(replacement) as bundle:
        assert bundle.testzip() is None
        assert set(bundle.namelist()) == set(members)
        import hashlib
        for name,path in members.items():
            assert hashlib.sha256(bundle.read(name)).hexdigest() == sha(path), name
    replacement.replace(archive)
    save(ROOT/'archive_verification.json',{'status':'PASS','archive':archive.name,'sha256':sha(archive),
         'bytes':archive.stat().st_size,'verified_files':len(members),'all_archive_entries_hashed':True,
         'original_input_and_computational_source_identity_unchanged':True,
         'original_execution_clock_retained':True,'old_and_new_accounts_released':True,'portal_uploads':0})
    current = ROOT.parent/'CURRENT_BEST.md'
    text = current.read_text().replace('improvement_v21/delivery/V21_Result_Summary.md',
                                      'improvement_v22/delivery/V22_Result_Summary.md')
    current_paragraph = (f"\nV22 completed one exact P35-SC38 pair-removal setting, ten development folds and three fixed replacements. Best mean relative development gain: {best['mean_relative_gain']:.6%}. "+
             summary['decision']+' All three comparisons were independently reconstructed. No portal upload occurred; original V14 artifacts remain the observed best. The owned T4 was released; final selected-account verification found zero active assignments. The requested account change and repeated engineering retained the original four-hour clock. See improvement_v22/delivery/V22_Result_Summary.md.\n')
    if current_paragraph not in text: text += current_paragraph
    current.write_text(text)
    print(json.dumps({'status':'PASS','decision':summary['decision'],'archive':str(archive),
                      'archive_files':len(members),'archive_sha256':sha(archive),
                      'elapsed_hours':ledger['seconds']/3600,'portal_uploads':0},indent=2))


if __name__ == '__main__': main()
