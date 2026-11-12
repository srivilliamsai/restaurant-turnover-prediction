"""Build the standalone, executed V8 research delivery and measured report."""
import argparse
import base64
import json
import zipfile
import nbformat as nb
import pandas as pd
from engine import ROOT, STEM, Engine, sha, save

OUT = ROOT / 'results'


def build():
    book = nb.v4.new_notebook(metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12'}, 'colab': {'name': 'V8_Research_Colab.ipynb', 'provenance': []}})
    def md(s): book.cells.append(nb.v4.new_markdown_cell(s))
    def code(s): book.cells.append(nb.v4.new_code_cell(s))
    md('''# Restaurant Turnover — V8 measured research

submission · submission · Team submission

The completed local round found **no qualifying V8 candidate**. Keep V7, whose user-reported portal RMSE is **12,024,316.5355221**. This notebook independently repeats all 16 fixed settings and 64 blend configurations, then freshly reproduces the scored V7 predictor. The unchanged V7 CSV/notebook are included as the fallback; uploading that CSV again does not create a new model.

All source is embedded. Only the four official data CSVs are required; previous project directories are not needed. No competitor code, predictions, weights or leaderboard-derived adjustments enter the predictor. All validation targets remain in INR scoring.''')
    code('''import os, sys, json, base64, hashlib, subprocess, importlib.metadata, tempfile
from pathlib import Path
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'LOKY_MAX_CPU_COUNT']:
    os.environ[key] = '2'
PACKAGES = {'numpy':'2.5.4', 'pandas':'3.0.6', 'scikit-learn':'1.9.1', 'scipy':'1.18.1', 'catboost':'1.2.10'}
needed = []
for package, version in PACKAGES.items():
    try:
        installed = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed != version:
        needed.append(f'{package}=={version}')
if needed:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *needed])
import numpy as np
import pandas as pd
print('Python:', sys.version.split()[0])
print({p: importlib.metadata.version(p) for p in PACKAGES})
''')
    md('''## Inputs and storage

Set local MLA_DATA_DIR or upload the four original CSVs when prompted. Default execution is fresh. Optional Google Drive storage requires your authorization; free Colab resources vary. Set REUSE_CHECKPOINTS=True to resume matching checkpoints.''')
    code('''USE_GOOGLE_DRIVE = False
REUSE_CHECKPOINTS = False
if USE_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    os.environ['MLA_OUTPUT_DIR'] = '/content/drive/MyDrive/MLA_v8'
DATA_DIR = Path(os.environ.get('MLA_DATA_DIR', '.')).resolve()
OUTPUT_DIR = Path(os.environ.get('MLA_OUTPUT_DIR', '.')).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REQUIRED = ['Train_dataset.csv', 'Test_dataset.csv', 'Data_Dictionary.csv', 'Sample_Solution.csv']
if not all((DATA_DIR / n).is_file() for n in REQUIRED):
    try:
        from google.colab import files
    except ImportError:
        raise FileNotFoundError('Place four official CSVs in DATA_DIR or set MLA_DATA_DIR.')
    files.upload()
    DATA_DIR = Path.cwd()
WORK_ROOT = OUTPUT_DIR / 'v8_resumable' if REUSE_CHECKPOINTS else Path(tempfile.mkdtemp(prefix='v8_fresh_', dir=OUTPUT_DIR))
WORK_ROOT.mkdir(parents=True, exist_ok=True)
print('Research root:', WORK_ROOT)
''')
    names = ['legacy_reference.py', 'legacy_models.py', 'baseline.py', 'modeling.py', 'engine.py',
             'workflow.py', 'reference_manifest.json', 'v5_contract.json']
    paths = [ROOT / n for n in names] + sorted((ROOT / 'baseline_snapshots').glob('*.npz')) + sorted((ROOT / 'scored_v7_backup').glob('*'))
    bundle = {str(p.relative_to(ROOT)): base64.b64encode(p.read_bytes()).decode() for p in paths if p.is_file()}
    hashes = {n: sha(ROOT / n) for n in bundle}
    code('BUNDLE = json.loads(' + repr(json.dumps(bundle)) + ')\nFILE_HASHES = ' + repr(hashes) + '''
for relative, encoded in BUNDLE.items():
    path = WORK_ROOT / relative
    value = base64.b64decode(encoded)
    assert hashlib.sha256(value).hexdigest() == FILE_HASHES[relative]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == value, 'Runtime source changed; choose fresh output storage.'
    else:
        path.write_bytes(value)
sys.path.insert(0, str(WORK_ROOT))
from engine import Engine
import workflow
e = Engine(DATA_DIR, WORK_ROOT / 'results', budget_hours=18)
e.cache.stage = 'audit'
workflow.audit(e)
print('PASS: input, implementation, snapshot and scored-file hashes.')
''')
    for stage, text in [('screen', 'Freshly fit sixteen settings on five folds of assignments 42 and 71.'),
        ('refine', 'Refine only settings improving on both assignments, then freeze.'),
        ('confirm', 'Confirm only frozen finalists. No finalist means no fabricated confirmation score.'),
        ('export', 'Freshly reproduce scored V7, then export qualified candidates or its exact fallback.')]:
        md('## ' + stage.capitalize() + '\n\n' + text)
        code(f'e.cache.stage = {stage!r}\nworkflow.{stage}(e)\n')
    md('''## Measured results

Positive gains mean lower RMSE than V7. These figures are recomputed from this execution. Cross-validation scores and portal scores describe different datasets.''')
    code('''table = pd.read_csv(e.out / 'screening.csv')
best = table.groupby('family', sort=False).head(1)
print(best[['name', 'weight', 'gain_42', 'gain_71', 'mean_rmse']].to_string(index=False))
decision = json.loads((e.out / 'decision.json').read_text())
print('Decision:', decision['decision'])
reproduction = json.loads((e.out / 'v7_full_reproduction.json').read_text())
assert reproduction['maximum_difference_INR'] < .01
if not REUSE_CHECKPOINTS:
    assert e.cache.reads == 0
print('V7 maximum reproduction difference (INR):', reproduction['maximum_difference_INR'])
print('Checkpoint reads:', e.cache.reads)
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass
else:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    pos = np.arange(len(best))
    ax.barh(pos - .17, best['gain_42']/1000, height=.3, label='42', color='#2563eb')
    ax.barh(pos + .17, best['gain_71']/1000, height=.3, label='71', color='#0f766e')
    ax.set_yticks(pos, best['family'])
    ax.axvline(0, color='#334155', linewidth=1)
    ax.set(xlabel='RMSE improvement over V7 (INR thousands)', title='Best screened blend in each family')
    ax.legend(title='Assignment')
    fig.tight_layout()
    plt.show()
''')
    md('''## Files

The fallback is byte-identical to the already scored V7. Preserve its matching notebook. The research archive includes original sources, provenance and measured results. It is not a portal CSV. No Olympus or one-time coding upload is performed.''')
    code('''import zipfile
archive = OUTPUT_DIR / 'V8_research_results.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for path in WORK_ROOT.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            z.write(path, path.relative_to(WORK_ROOT))
print('Saved:', archive)
if not decision['trials']:
    csv_path = WORK_ROOT / 'recommended_submission' / 'submission.csv'
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == e.manifest['preserved_v7'][csv_path.name]
    print('Keep V7; CSV hash:', hashlib.sha256(csv_path.read_bytes()).hexdigest())
try:
    from google.colab import files
except ImportError:
    pass
else:
    files.download(str(archive))
''')
    for cell in book.cells:
        if cell.cell_type == 'code': compile(cell.source, '<V8 notebook>', 'exec')
    nb.validate(book)
    nb.write(book, ROOT / 'V8_Research_Colab.ipynb')
    print('Built standalone research notebook.', flush=True)


def finalize():
    e = Engine()
    table = pd.read_csv(OUT / 'screening.csv')
    d = json.loads((OUT / 'decision.json').read_text())
    execution = json.loads((OUT / 'research_notebook_execution.json').read_text())
    reproduction = json.loads((OUT / 'v7_full_reproduction.json').read_text())
    assert not d['trials'], 'Rewrite measured delivery if a candidate qualifies.'
    lines = ['| Family | Best component | Weight | Gain on 42 (INR) | Gain on 71 (INR) |', '| --- | --- | ---: | ---: | ---: |']
    for r in table.groupby('family', sort=False).head(1).to_dict(orient='records'):
        lines.append(f"| {r['family']} | {r['name']} | {r['weight']:.0%} | {r['gain_42']:,.2f} | {r['gain_71']:,.2f} |")
    report = f'''# V8 result — retain scored V7

**Decision:** no new V8 candidate qualifies. Keep V7 Trial 1: user-reported portal RMSE **12,024,316.5355221**, rank 24 in the supplied screenshot. No V8 CSV should be uploaded.

## Measured comparisons

Completed all **16 settings**, **160 V8 outer-fold component fits**, and **64 blends** on identical five-fold assignments 42/71. Recomputed 128 original-scale development scores. No blend improved V7 on either assignment. Positive gain would mean improvement; the gains below are negative.

{chr(10).join(lines)}

The closest approach, target-encoded square-root HGB with smoothing 200 at 25% weight, worsened RMSE by about INR 3,134 and 8,519. The article-inspired square-root Ridge hypothesis also failed promotion. Public-repository scores were not substituted for our measurements.

## Gates and limits

No setting passed development. Refinement and fresh confirmation assignments were **not run**, as specified by the plan. No confirmation confidence intervals were calculated. The fixed screen finished before the 18-hour ceiling; the experiment list was kept unchanged.

All 3,493 original target values remain in evaluation. IDs and targets are excluded from predictors. No external data, public predictions, copied model weights or leaderboard-derived row adjustments enter the models. Every assignment reuses historically examined labels; development results are not untouched external evidence or portal scores.

## Reproduction

- Fresh full V7 refit matched the scored CSV within **INR {reproduction['maximum_difference_INR']:.12f}**, below INR .01.
- All {execution['code_cells']} code cells of the standalone notebook executed in a fresh process and isolated directory with no earlier project directories or V8 component checkpoints. It repeated all V8 fits and the full V7 refit; no saved errors. Hash-verified prior V7 OOF snapshots were explicitly reused for development baselines.
- All {execution['development_scores_recomputed']} notebook scores matched the independent workflow within INR .01; maximum difference **INR {execution['maximum_score_difference']:.12f}**.
- Target-encoding own-label exclusion, training-only frequency maps, categorical limits, missing/unseen values, permutation, target/ID invariance and stale/corrupt/interrupted checkpoints passed.
- All 384 turnover-bin diagnostics and 640 fold comparisons are saved in results/development_segments.csv and results/development_folds.csv. No targets were filtered from the primary metric.
- The 500-row fallback CSV and matching notebook are byte-identical to scored V7. Original IDs/order, exact headers, finite nonnegative values and twelve malformed-file rejections passed.
- Ran locally on CPU. Colab compatibility is provided; no cloud execution or paid compute is claimed. No portal or one-time coding upload was performed.

## Repository review before training

Reviewed revision `2ce8c0657b788198a2d8b06066b351320881ac0d` of the author's repository. Square-root Ridge is documented. Its saved manifests report S05 at 12.115M and the leaderboard-adjusted S08 at 12.088M; these are author-reported records, not independently reproduced portal measurements. The article's later 2.893M result is not a demonstrated training recipe in this review and does not identify your current leader's method.

Sources: [Ridge source](https://github.com/RoloMagnus/restaurant-turnover/blob/master/src/winner_replication_study.py), [saved model manifest](https://github.com/RoloMagnus/restaurant-turnover/blob/master/experiments/champion_v2_manifest.json), [article](https://medium.com/@rolandolbouloyfascitelli/how-playing-volleyball-with-ai-agents-helped-me-predict-restaurant-turnover-and-win-a-hackathon-a3c696d6e900).

## Files to use

Keep the existing V7 CSV. `recommended_submission` contains the same CSV and matching coding notebook for convenience. `V8_Research_Colab.ipynb` reproduces the research; it is not a new winning submission. Preserve the one-time coding upload until your final choice. The best observed portal score remains **12,024,316.5355221**.
'''
    (ROOT / 'REPORT.md').write_text(report)
    (ROOT / 'README.md').write_text('''# V8 restaurant-turnover research

Read REPORT.md first. All sixteen settings and sixty-four blend configurations failed the development gate; retain scored V7. The recommended_submission files are unchanged V7 backups, not a new CSV to upload.

V8_Research_Colab.ipynb is standalone and executed, with measured outputs. Its default fresh execution repeats the bounded screening and full V7 fit. Optional resumable checkpoints use strict hashes. Supply the four official CSVs; earlier project directories are not needed.

Local command: python improvement_v8/run_local.py all --data-dir /path/to/official/csvs
Stages: audit, screen, refine, confirm, export. --out-dir names the results directory; candidate/recommendation folders are created alongside it. --budget-hours defaults to 18, the maximum. No paid resource is required. No portal action was taken.

Refinement and confirmation were skipped intentionally because no setting qualified. The source repository was reviewed before training; competitor code and predictions are not embedded in the predictor. The 18-hour ceiling is not a requirement to repeat rejected experiments.
''')
    sources = ['modeling.py', 'baseline.py', 'engine.py', 'workflow.py', 'legacy_models.py', 'legacy_reference.py',
               'v5_contract.json', 'reference_manifest.json', 'verify_core.py', 'prepare_reference.py',
               'run_local.py', 'deliver.py', 'execute_notebook.py', 'verify_delivery.py', 'analyze.py']
    paths = [ROOT / n for n in ['REPORT.md', 'README.md', 'PLAN.md', 'V8_Research_Colab.ipynb',
        'rolo_repository_review.json', 'portal_trials.csv']]
    paths += sorted((ROOT / 'recommended_submission').glob('*'))
    paths += [OUT / n for n in ['audit.json', 'screening.csv', 'development.csv', 'experiment_plan.json',
        'candidate_lock.json', 'decision.json', 'core_verification.json', 'v7_full_reproduction.json', 'research_notebook_execution.json',
        'development_segments.csv', 'development_folds.csv']]
    save(ROOT / 'delivery_manifest.json', {'inputs': e.inputs, 'runtime': e.runtime,
        'implementation': {n: sha(ROOT / n) for n in sources},
        'outputs': {str(p.relative_to(ROOT)): sha(p) for p in paths}, 'decision': d,
        'preserved_v7': e.manifest['preserved_v7'], 'qualified_candidates': 0,
        'v8_outer_fold_fits': 160, 'development_configurations': 64,
        'confirmation_assignments_run': 0, 'current_best_portal_rmse': 12024316.5355221,
        'notebook_execution': execution, 'v7_full_reproduction': reproduction, 'portal_uploads_by_agent': 0})
    archive = ROOT / 'V8_Delivery.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for path in ROOT.rglob('*'):
            if path.is_file() and path != archive and path.name != 'record_review.py' and '__pycache__' not in path.parts and 'notebook_execution' not in path.parts and not path.name.endswith('.log'):
                z.write(path, path.relative_to(ROOT))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist()))
    print('Finalized report, manifest and delivery ZIP.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['build', 'finalize'])
    args = p.parse_args()
    globals()[args.stage]()
