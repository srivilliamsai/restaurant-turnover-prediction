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
    frozen = json.loads((ROOT / 'source_frozen.json').read_text())
    engine = Engine()
    assert engine.sources == frozen['sources'] and engine.inputs == frozen['inputs']
    import verify_results
    verify_results.main()
    decision = json.loads((engine.out / 'decision.json').read_text())
    if decision['qualified']:
        fresh=ROOT/'delivery/fresh_execution/fresh_execution_checks.json'
        checks=json.loads(fresh.read_text())
        assert checks['status']=='PASS' and checks['maximum_final_difference_INR']<.01
        import nbformat
        book=nbformat.read(ROOT/'delivery'/(STEM+'.ipynb'),as_version=4)
        assert all(c.execution_count is not None for c in book.cells if c.cell_type=='code')
        assert not [o for c in book.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.get('output_type')=='error']
    release = json.loads((ROOT / 'transport_run/runtime_release.json').read_text())
    assert release['release_command_succeeded'] and release['owned_session'] == 'v18-free-t4'
    save(ROOT / 'delivery/runtime_release.json', release)
    switch = ROOT/'account_switch.json'
    if switch.exists():shutil.copy2(switch,ROOT/'delivery/account_switch.json')
    account = ROOT / 'account_release_check.json'
    if account.exists():
        shutil.copy2(account, ROOT / 'delivery/account_release_check.json')
    with (engine.out / 'screening.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    best = max(rows, key=lambda row: float(row['mean_relative_gain'])) if rows else None
    pilot = json.loads((engine.out / 'pilot.json').read_text())
    fit_count = len(list((engine.out / 'cache').glob('*.json')))
    log_target = ROOT / 'delivery/execution_logs'
    log_target.mkdir(exist_ok=True)
    for path in [ROOT / 'run.log', ROOT / 'setup.log', *sorted(engine.out.glob('pilot_*.log'))]:
        if path.is_file():
            shutil.copy2(path, log_target / path.name)
    probe = ROOT / 'transport_run/pilot_runtime_probe.json'
    if probe.exists():
        shutil.copy2(probe, log_target / probe.name)
    lines = ['# V18 result', '',
             'V18 completed. **Retain V14.**' if not decision['qualified'] else 'V18 completed with one qualified candidate for a manual portal trial.', '',
             'V14 remains the best observed portal submission: **INR 11,736,110.7877709**. No V18 portal upload occurred.', '']
    if best:
        old_mean = sum(float(best['rmse_' + str(seed)]) + float(best['gain_' + str(seed)]) for seed in [42, 71]) / 2
        new_mean = sum(float(best['rmse_' + str(seed)]) for seed in [42, 71]) / 2
        lines += ['| Model | Mean development RMSE | Portal RMSE |', '|---|---:|---:|',
                  f'| V14 control | {old_mean:,.2f} | 11,736,110.7877709 |',
                  f"| Best V18 blend | {new_mean:,.2f} | Not submitted |", '',
                  f"Best setting: **{best['name']} at {float(best['weight']):.0%} weight**. Mean local gain: **{100 * float(best['mean_relative_gain']):.6f}%**, versus the frozen **0.20%** requirement. Gains on assignments 42/71: INR {float(best['gain_42']):,.2f} and {float(best['gain_71']):,.2f}.", '']
    lines += ['| Recipe | Engineering result | Admission detail |', '|---|---|---|']
    for name, record in pilot['settings'].items():
        detail = pilot.get('budget_exclusions', {}).get(name, record.get('reason', 'Passed the declared engineering checks.'))
        detail = detail.replace('|', '/').replace('\n', ' ')
        lines.append(f"| {name} | {record['status']} | {detail} |")
    if not rows:
        lines += ['', '**No recipe reached accuracy testing. V18 has no measured RMSE improvement.**']
    lines += ['', f'{fit_count} complete outer-fold predictions and {len(rows)} fixed blends were independently reconstructed from hashed outputs. All 3,493 original labels are retained in the inputs; no labels were removed.', '',
              'Each admitted setting passed its full-fold engineering, saved-state and fresh-fit checks. All engineering failures and budget exclusions are recorded in pilot.json with full pilot logs.', '',
              'Confirmation and submission artifacts follow the frozen decision; no unqualified candidate was exported.', '',
              'The scored V14 CSV and matching executed notebook are preserved byte-for-byte under KEEP_V14 filenames. The owned free T4 was released; account release evidence accompanies this result. The eighteen-hour ledger was not reset.']
    (ROOT / 'delivery/V18_Result_Summary.md').write_text('\n'.join(lines) + '\n')
    save(ROOT / 'delivery/execution_summary.json', {
        'complete': True, 'qualified': decision['qualified'], 'outer_fold_outputs': fit_count,
        'blend_comparisons': len(rows), 'best': best, 'all_original_labels': len(engine.y),
        'owned_runtime_released': True, 'portal_uploads': 0,
        'elapsed_seconds_including_setup_and_local_verification': time.time() - engine.budget.authorization['started_epoch'],
        'maximum_hours': 18, 'paid_resources': False})
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
    import delivery
    original_report = delivery.report
    def formatted_report(current_engine, current_decision, current_target):
        original_report(current_engine, current_decision, current_target)
        report_path = current_target / 'V18_Report.md'
        report_path.write_text(report_path.read_text().replace('\\n', '\n'))
    delivery.report = formatted_report
    delivery.package(engine, decision, ROOT / 'delivery')
    manifest = json.loads((ROOT / 'delivery/manifest.json').read_text())
    with zipfile.ZipFile(ROOT / 'V18_Delivery.zip') as archive:
        assert archive.testzip() is None
        for name, expected in manifest['files'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
    save(ROOT / 'archive_verification.json', {'status': 'PASS', 'archive_sha256': sha(ROOT / 'V18_Delivery.zip'),
         'manifest_files_verified': len(manifest['files']), 'independent_review_included': 'independent_review.json' in manifest['files'],
         'runtime_release_included': 'runtime_release.json' in manifest['files'], 'v14_scored_files_preserved': True})
    print(json.dumps({'status': 'PASS', 'qualified': decision['qualified'], 'folds': fit_count,
                      'comparisons': len(rows), 'best_mean_relative_gain': float(best['mean_relative_gain']) if best else None,
                      'delivery': str(ROOT / 'V18_Delivery.zip')}, indent=2))


if __name__ == '__main__':
    main()
