"""Package only independently verified Colab results and reviewable artifacts."""
import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
args = parser.parse_args()
run = args.run.resolve()
root = Path(__file__).resolve().parent
review = json.loads((run / 'independent_review.json').read_text())
assert review['status'] == 'PASS'
payload = run / 'review_payload'
decision = json.loads((payload / 'results/decision.json').read_text())
driver = json.loads((run / 'execution.json').read_text())
assert driver['status'] == 'completed'
book_path = run / 'V9_Research_Colab_CLI_executed.ipynb'
book = json.loads(book_path.read_text())
original = json.loads((root / 'V9_Research_Colab_Lite.ipynb').read_text())
codes = [c for c in book['cells'] if c['cell_type'] == 'code']
original_codes = [c for c in original['cells'] if c['cell_type'] == 'code']
assert len(codes) == 10 and len(codes) == len(original_codes)
assert all(c['execution_count'] is not None for c in codes)
assert all(''.join(a['source']) == ''.join(b['source']) for a,b in zip(codes,original_codes))
assert all(o.get('output_type') != 'error' for c in codes for o in c.get('outputs',[]))

submission = run / 'submission'
submission.mkdir(exist_ok=True)
for number, choice in enumerate(decision['trials'], 1):
    target = submission / f'trial_{number}'
    target.mkdir(exist_ok=True)
    source = payload / 'candidates' / f'trial_{number}'
    for path in list(source.glob('*.csv')) + list(source.glob('*.ipynb')):
        shutil.copyfile(path, target / path.name)
if not decision['trials']:
    shutil.copytree(payload / 'recommended_submission', submission / 'keep_V7', dirs_exist_ok=True)

with (payload / 'results/development.csv').open() as f:
    development = list(csv.DictReader(f))
best = {}
for row in development:
    key = (row['name'],row['mode'])
    if key not in best or float(row['mean_rmse']) < float(best[key]['mean_rmse']):
        best[key] = row
selection = json.loads((payload / 'results/candidate_lock.json').read_text())['shortlist']
report = ['# V9 — measured Colab CLI results', '',
    '**Current observed portal best: V7, RMSE 12,024,316.5355221.**', '',
    ('A validation-qualified V9 trial is ready for portal testing.' if decision['trials'] else
     '**Retain V7. No V9 candidate passed every required confirmation gate; no new upload is recommended.**'), '',
    'Google sign-in and free standard Tesla T4 allocation succeeded through the official Colab CLI. '
    'The frozen notebook executed all ten code cells. No paid resources or portal uploads were requested by this run.', '',
    f"Screening and refinement evaluated {review['development_configurations']} blend configurations. "
    'All 3,493 original training targets were retained. Metrics below are validation RMSE in INR; '
    'they are not new portal scores.', '',
    '## Development comparisons', '',
    'Each row shows the best development blend for that setting. Positive gain means lower RMSE than V7.', '',
    '| Setting | Mode | Candidate weight | Gain 42 | Gain 71 | Improves both |',
    '|---|---|---:|---:|---:|---|']
for row in sorted(best.values(), key=lambda r: float(r['mean_rmse'])):
    a,b = float(row['gain_42']),float(row['gain_71'])
    report.append(f"| {row['name']} | {row['mode']} | {float(row['weight']):.0%} | {a:,.2f} | {b:,.2f} | {'yes' if a>0 and b>0 else 'no'} |")
report += ['', '## Frozen confirmation', '',
    'Features, model parameters, estimator count and blend weight were locked before confirmation. '
    'These assignments reuse historically examined labels and provide robustness evidence, not an untouched external test.', '']
for choice in decision['candidates']:
    report += [f"`{choice['name']}` / `{choice['mode']}` / candidate weight {choice['weight']:.0%}.", '',
        '| Assignment | V7 RMSE | Candidate RMSE | Gain | Relative change |',
        '|---|---:|---:|---:|---:|']
    for scheme,values in choice['metrics'].items():
        report.append(f"| {scheme} | {values['v7_rmse']:,.2f} | {values['candidate_rmse']:,.2f} | {values['gain']:,.2f} | {values['relative_change']:.3%} |")
    failures = [s for s,v in choice['metrics'].items() if
        (s in ['48211','48212','48213'] and v['gain']<=0) or
        (s in ['city','strat:27421'] and v['relative_change']>.01)]
    low,high=choice['uncertainty']['ci95']
    report += ['', f"Qualification: **{'passed' if choice['qualified'] else 'failed'}**. "
        + ('Failed comparisons: '+', '.join(failures)+'.' if failures else 'Every required gate passed.'), '',
        f'Conditional paired-row bootstrap, 5,000 resamples: mean random-assignment gain '
        f"{choice['mean_random_gain']:,.2f}; 95% interval [{low:,.2f}, {high:,.2f}]. "
        'This excludes uncertainty from model search, retraining and distribution shift.', '']
concentration = run / 'interim_development_concentration.json'
if concentration.exists():
    report += ['## Error concentration', '',
        'For the frozen finalist on development assignments:']
    for entry in json.loads(concentration.read_text()):
        report.append(f"- Assignment {entry['scheme']}: ten restaurants account for "
            f"{entry['top10_fraction_of_sse_reduction']:.1%} of net squared-error reduction. "
            f"The >80M group contributes {entry['greater80M_fraction_of_sse_reduction']:.1%} "
            'of that net reduction; a negative contribution means this group worsened.')
    report += ['', 'Full ≤40M, 40–80M and >80M development errors are in `development_segments.csv`; '
               'confirmation segment and fold errors are in the research archive. The primary metric always uses every original target.', '']
report += ['## Reproduction and delivery', '',
    f"Independent review recomputed {review['development_scores_recomputed']} development scores; "
    f"maximum difference was INR {review['maximum_score_difference_INR']:.12f}. "
    'It checked source, data, fold, prediction and CSV hashes and recomputed every confirmation decision.', '',
    f"Fresh full V7 reproduction differed by at most INR {review['fresh_v7_reproduction']['maximum_difference_INR']:.12f}. "
    f"Fresh matching candidate notebook checks completed for {len(review['fresh_candidate_notebooks'])} trial(s). "
    'CSV checks require exactly 500 rows, original unique IDs and order, exact headers, finite nonnegative values, '
    'and reproduction within INR 0.01 in the recorded environment.', '',
    'Use only qualified trial CSVs for a new portal test. Keep the lowest observed portal score and record its exact CSV hash. '
    'Preserve the one-time coding submission until the final model choice. First place has not been established.', '',
    'TabPFN uses the official general-purpose v2 checkpoint, not competitor predictions or competition-trained weights. '
    'Checkpoint attribution: Prior Labs; Hollmann et al., Nature 2025, doi:10.1038/s41586-024-08328-6. '
    'Exact checkpoint hash and execution dependencies are recorded in the environment and research manifests.', '']
report_path = run / 'V9_Colab_Report.md'
report_path.write_text('\n'.join(report))

files = [report_path, book_path, run/'V9_research_results.zip', run/'independent_review.json',
         run/'execution.json', run/'development_segments.csv',run/'development_folds.csv']
files += [p for p in submission.rglob('*') if p.is_file()]
environment = run / 'colab_environment_manifest.json'
if environment.exists(): files.append(environment)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = {'inputs': review['inputs'], 'sources': review['sources'], 'execution': driver,
    'decision': decision, 'qualified_candidates': len(decision['trials']),
    'observed_portal_best': {'version':'V7','rmse':12024316.5355221},
    'new_portal_scores_observed':0, 'paid_resources_requested':False,
    'files': {str(p.relative_to(run)):sha(p) for p in files},
    'official_inputs_zip_sha256':sha(root/'V9_Official_Inputs.zip')}
manifest_path = run / 'V9_CLI_manifest.json'
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
archive_path = root / 'V9_CLI_Delivery.zip'
with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as z:
    for path in files+[manifest_path]: z.write(path,path.relative_to(run))
    z.write(root/'V9_Official_Inputs.zip','V9_Official_Inputs.zip')
with zipfile.ZipFile(archive_path) as z: assert z.testzip() is None
(root/'CLI_PROGRESS.md').write_text('# V9 CLI execution complete\n\n'+decision['decision']+'.\n\n'
    'Latest report: `colab_cli_runs/20261026T160148Z/V9_Colab_Report.md`.\n\n'
    'Latest package: `V9_CLI_Delivery.zip`. The earlier CPU/manual handoff remains historical.\n')
print('Packaged:', archive_path)
print('Qualified trials:',len(decision['trials']))
