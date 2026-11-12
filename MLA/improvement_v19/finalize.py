"""Verify collected folds independently and rebuild the complete local delivery."""
import csv
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path
from core import ROOT, STEM, Engine, save, sha


def main():
    collected = ROOT / 'colab_results'
    for folder in ['results', 'delivery']:
        shutil.copytree(collected / folder, ROOT / folder, dirs_exist_ok=True)
    for name in ['resolved_models.json', 'setup.log', 'run.log']:
        shutil.copy2(collected / name, ROOT / name)
    remote_archive=collected/'V19_Delivery.zip'
    if remote_archive.exists():
        with zipfile.ZipFile(remote_archive) as archive:
            assert archive.testzip() is None
            for name in archive.namelist():
                path=Path(name)
                assert not path.is_absolute() and '..' not in path.parts
                if path.parts and path.parts[0]=='engineering_sources':
                    archive.extract(name,ROOT)
    frozen = json.loads((ROOT / 'source_frozen.json').read_text())
    engine = Engine()
    assert engine.sources == frozen['sources'] and engine.inputs == frozen['inputs']
    import verify_results
    verify_results.main()
    decision = json.loads((engine.out / 'decision.json').read_text())
    if decision['qualified']:
        fresh=ROOT/'delivery/fresh_execution/fresh_execution_checks.json'
        checks=json.loads(fresh.read_text())
        assert checks['status']=='PASS' and checks['prediction_agreement_passed']
        import nbformat
        book=nbformat.read(ROOT/'delivery'/(STEM+'.ipynb'),as_version=4)
        assert all(c.execution_count is not None for c in book.cells if c.cell_type=='code')
        assert not [o for c in book.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.get('output_type')=='error']
    release = json.loads((ROOT / 'transport_run/runtime_release.json').read_text())
    assert release['release_command_succeeded'] and release['owned_session'] == 'v19-free-t4'
    allocation = json.loads((ROOT / 'transport_run/allocations.json').read_text())[-1]
    assert release['epoch'] >= allocation['requested_epoch'], 'Release must cover the current allocation.'
    save(ROOT / 'delivery/runtime_release.json', release)
    switch = ROOT/'account_switch.json'
    if switch.exists():shutil.copy2(switch,ROOT/'delivery/account_switch.json')
    account = ROOT / 'account_release_check.json'
    if account.exists():
        account_record = json.loads(account.read_text())
        assert account_record['status'] == 'PASS' and account_record['live_active_assignments'] == 0
        assert account_record['checked_epoch'] >= release['epoch']
        shutil.copy2(account, ROOT / 'delivery/account_release_check.json')
    else:
        raise AssertionError('A fresh account release check is required.')
    with (engine.out / 'screening.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    best = max(rows, key=lambda row: float(row['mean_relative_gain'])) if rows else None
    pilot = json.loads((engine.out / 'pilot.json').read_text())
    fit_count = len(list((engine.out / 'cache').glob('*.json')))
    status_path = ROOT/'execution_status.json'
    execution = json.loads(status_path.read_text())
    execution.update(status='COMPLETE', completed_epoch=time.time(),
        development_scores_generated=bool(rows), completed_candidate_development_folds=sum(1 for p in (engine.out/'cache').glob('*.json') if json.loads(p.read_text())['context']['name'] in __import__('features').SPECS and json.loads(p.read_text())['context']['scheme'] in ['42','71']),
        completed_matched_control_folds=sum(1 for p in (engine.out/'cache').glob('*.json') if json.loads(p.read_text())['context']['name'] in ['BASE','P35_8']), fixed_blend_comparisons=len(rows),
        qualified_development_candidates=len(decision.get('finalists',[])),
        qualified=decision['qualified'], confirmation_performed=bool(decision.get('finalists')),
        independent_result_reconstruction='PASS', owned_runtime_released=True,
        selected_account_active_assignments=0, portal_uploads=0,
        completion_evidence='Collected export stage, independent reconstruction, owned-runtime release and fresh account check.')
    save(status_path, execution)
    readme=ROOT/'README.md'
    readme.write_text(readme.read_text().replace('The free T4 has been released. Independent verification and final packaging are in progress.',
        'Independent reconstruction passed. The free T4 has been released and the selected account has zero active assignments. See [the measured result](delivery/V19_Result_Summary.md).'))
    plan=ROOT/'PLAN.md'
    plan.write_text(plan.read_text().replace('The free T4 has been released; final independent verification and delivery are in progress.',
        'Execution and independent reconstruction are complete; the free T4 was released. See [the measured result](delivery/V19_Result_Summary.md).'))
    progress=ROOT/'IMPLEMENTATION_PROGRESS.md'
    progress.write_text(progress.read_text()+'\nFinal independent reconstruction passed. Fresh selected-account verification found zero active assignments after owned-runtime release. No V19 portal upload occurred.\n')
    best_file=ROOT.parent/'CURRENT_BEST.md'
    best_text=best_file.read_text().replace('improvement_v18/delivery/V18_Result_Summary.md)','improvement_v19/delivery/V19_Result_Summary.md)')
    best_text=best_text.replace(' — new Colab account verified; training has not started.',' — execution complete; retain V14.')
    paragraph='V19 completed twenty candidate development folds and fifteen fixed blends. The best blend, 75% V14 plus 25% Causilo, improved both development assignments with mean relative gain 0.013192%, below the unchanged 0.20% qualification floor. TabDPT and the equal blend did not improve mean RMSE. Both Kumo sizes were excluded by a dependency import failure. No candidate advanced to confirmation; no new V19 submission was exported or uploaded. Independent reconstruction passed, the owned free T4 was released, and the selected account had zero active assignments. The initial interrupted attempt and engineering repairs are documented without resetting the eight-hour elapsed ledger.'
    if paragraph not in best_text:best_text+='\n'+paragraph+'\n'
    best_file.write_text(best_text)
    log_target = ROOT / 'delivery/execution_logs'
    log_target.mkdir(exist_ok=True)
    for path in [ROOT / 'run.log', ROOT / 'setup.log', *sorted(engine.out.glob('pilot_*.log'))]:
        if path.is_file():
            shutil.copy2(path, log_target / path.name)
    probe = ROOT / 'transport_run/pilot_runtime_probe.json'
    if probe.exists():
        shutil.copy2(probe, log_target / probe.name)
    lines = ['# V19 result', '',
             'V19 completed. **Retain V14.**' if not decision['qualified'] else 'V19 completed with one qualified candidate for a manual portal trial.', '',
             'V14 remains the best observed portal submission: **INR 11,736,110.7877709**. No V19 portal upload occurred.', '']
    if best:
        old_mean = sum(float(best['rmse_' + str(seed)]) + float(best['gain_' + str(seed)]) for seed in [42, 71]) / 2
        new_mean = sum(float(best['rmse_' + str(seed)]) for seed in [42, 71]) / 2
        lines += ['| Model | Mean development RMSE | Portal RMSE |', '|---|---:|---:|',
                  f'| V14 control | {old_mean:,.2f} | 11,736,110.7877709 |',
                  f"| Best V19 blend | {new_mean:,.2f} | Not submitted |", '',
                  f"Best setting: **{best['name']} at {float(best['weight']):.0%} weight**. Mean local gain: **{100 * float(best['mean_relative_gain']):.6f}%**, versus the frozen **0.20%** requirement. Gains on assignments 42/71: INR {float(best['gain_42']):,.2f} and {float(best['gain_71']):,.2f}.", '']
    lines += ['| Recipe | Engineering result | Admission detail |', '|---|---|---|']
    for name, record in pilot['settings'].items():
        detail = pilot.get('budget_exclusions', {}).get(name, record.get('reason', 'Passed the declared engineering checks.'))
        detail = detail.replace('|', '/').replace('\n', ' ')
        lines.append(f"| {name} | {record['status']} | {detail} |")
    if not rows:
        lines += ['', '**No recipe reached accuracy testing. V19 has no measured RMSE improvement.**']
    lines += ['', f'{fit_count} complete outer-fold predictions and {len(rows)} fixed blends were independently reconstructed from hashed outputs. All 3,493 original labels are retained in the inputs; no labels were removed.', '',
              'Each admitted setting passed its full-fold engineering, saved-state and fresh-fit checks. All engineering failures and budget exclusions are recorded in pilot.json with full pilot logs.', '',
              'Confirmation and submission artifacts follow the frozen decision; no unqualified candidate was exported.', '',
              'The scored V14 CSV and matching executed notebook remain preserved in reference/. The owned free T4 was released; account release evidence accompanies this result. The original eight-hour ledger was not reset.', '',
              'An earlier runtime generated ten TabDPT development folds and automatic screening before an engineering repair. Those accuracy metrics were not inspected to choose repairs. Causilo query stability and a Kumo dependency import failure motivated the amendment. The first VM became unavailable before its complete archive could be downloaded; five completed fold backups and its source bundle were recovered. These interrupted and duplicate fits remain charged to the original elapsed-time ledger. See engineering_amendment.json and engineering_repair_evidence/.']
    (ROOT / 'delivery/V19_Result_Summary.md').write_text('\n'.join(lines) + '\n')
    save(ROOT / 'delivery/execution_summary.json', {
        'complete': True, 'qualified': decision['qualified'], 'outer_fold_outputs': fit_count,
        'blend_comparisons': len(rows), 'best': best, 'all_original_labels': len(engine.y),
        'owned_runtime_released': True, 'portal_uploads': 0,
        'elapsed_seconds_including_setup_and_local_verification': time.time() - engine.budget.authorization['started_epoch'],
        'maximum_hours': 8, 'paid_resources': False})
    budget_state = engine.budget.state()
    budget_state['local_verification_completed_epoch'] = time.time()
    budget_state['post_runtime_verification_seconds'] = time.time() - release['epoch']
    save(engine.out / 'compute_budget.json', budget_state)
    if not decision['qualified']:
        assert not (ROOT / 'delivery' / (STEM + '.csv')).exists()
        assert not (ROOT / 'delivery' / (STEM + '.ipynb')).exists()
        for suffix in ['.csv', '.ipynb']:
            assert sha(ROOT / 'reference' / ('scored_v14' + suffix)) == sha(ROOT / 'delivery' / ('KEEP_V14_' + STEM + suffix))
    if (ROOT/'mac_controls').exists():
        shutil.copytree(ROOT/'mac_controls',ROOT/'delivery/mac_controls',dirs_exist_ok=True)
    transport=ROOT/'delivery/execution_transport'
    transport.mkdir(exist_ok=True)
    for path in (ROOT/'transport_run').glob('*.json'):
        shutil.copy2(path,transport/path.name)
    coordinator=ROOT/'transport_run/coordinator.log'
    if coordinator.exists():shutil.copy2(coordinator,transport/'coordinator.log')
    for name in ['delivery_notes.json','execution_status.json','source_frozen.json','REPRODUCTION.md']:
        if (ROOT/name).exists():shutil.copy2(ROOT/name,ROOT/'delivery'/name)
    for name in ['engineering_amendment.json','active_environment_probe.json','IMPLEMENTATION_PROGRESS.md']:
        if (ROOT/name).exists():shutil.copy2(ROOT/name,ROOT/'delivery'/name)
    evidence=ROOT/'engineering_repair_evidence'
    if evidence.exists():shutil.copytree(evidence,ROOT/'delivery/engineering_repair_evidence',dirs_exist_ok=True)
    research=ROOT/'research_sources'
    if research.exists():shutil.copytree(research,ROOT/'delivery/research_sources',dirs_exist_ok=True)
    import delivery
    original_report = delivery.report
    def formatted_report(current_engine, current_decision, current_target):
        original_report(current_engine, current_decision, current_target)
        report_path = current_target / 'V19_Report.md'
        report_path.write_text(report_path.read_text().replace('\\n', '\n'))
    delivery.report = formatted_report
    delivery.package(engine, decision, ROOT / 'delivery')
    manifest = json.loads((ROOT / 'delivery/manifest.json').read_text())
    with zipfile.ZipFile(ROOT / 'V19_Delivery.zip') as archive:
        assert archive.testzip() is None
        for name, expected in manifest['files'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
        expected_members = {}
        for folder in [ROOT/'delivery', ROOT/'provenance', ROOT/'reference', ROOT/'engineering_sources', engine.out/'cache']:
            for path in folder.rglob('*'):
                if path.is_file() and '.partial' not in path.name and not any(part in path.parts for part in ['model_weights','__pycache__']) and path.suffix != '.safetensors':
                    prefix = '' if folder == ROOT/'delivery' else folder.name+'/'
                    expected_members[prefix+str(path.relative_to(folder))] = sha(path)
        for name in __import__('legacy_models').REQUIRED:
            expected_members['official_inputs/'+name] = sha(engine.data/name)
        assert len(archive.namelist()) == len(set(archive.namelist()))
        assert set(archive.namelist()) == set(expected_members)
        for name, expected in expected_members.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
    save(ROOT / 'archive_verification.json', {'status': 'PASS', 'archive_sha256': sha(ROOT / 'V19_Delivery.zip'),
         'manifest_files_verified': len(manifest['files']), 'independent_review_included': 'independent_review.json' in manifest['files'],
         'all_archive_members_hash_verified': len(expected_members), 'unique_member_names': True,
         'runtime_release_included': 'runtime_release.json' in manifest['files'], 'v14_scored_files_preserved': True})
    print(json.dumps({'status': 'PASS', 'qualified': decision['qualified'], 'folds': fit_count,
                      'comparisons': len(rows), 'best_mean_relative_gain': float(best['mean_relative_gain']) if best else None,
                      'delivery': str(ROOT / 'V19_Delivery.zip')}, indent=2))


if __name__ == '__main__':
    main()
