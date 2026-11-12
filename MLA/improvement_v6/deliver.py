"""Build the standalone notebook and measured report after the locked workflow."""
import argparse
import ast
import base64
import hashlib
import json
import textwrap
import zlib
from pathlib import Path
import numpy as np
import pandas as pd
import nbformat as nb
from workflow import ROOT, BASE, RESULTS, STEM, Run, sha, save_json, behavior_checks
from modeling import fit_final, splits, SPECS


def array_literal(values):
    payload = base64.b64encode(zlib.compress(np.asarray(values, dtype='<f8').tobytes())).decode()
    return "np.frombuffer(zlib.decompress(base64.b64decode(" + repr(payload) + ")), dtype='<f8').copy()"


def table(frame):
    frame = frame.copy()
    for column in frame.select_dtypes('number'):
        frame[column] = frame[column].map(lambda v: f'{v:,.2f}')
    return '| ' + ' | '.join(frame.columns) + ' |\n| ' + ' | '.join(['---'] * len(frame.columns)) + ' |\n' + '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in frame.itertuples(index=False, name=None))


def build_notebook():
    decision = json.loads((RESULTS / 'decision.json').read_text())
    audit = json.loads((RESULTS / 'audit.json').read_text())
    choice = decision['choice']
    contract = json.loads((ROOT / 'v5_contract.json').read_text())
    screening = pd.read_csv(RESULTS / 'screening.csv')
    confirm_path = RESULTS / 'confirmation.csv'
    confirmation = pd.read_csv(confirm_path) if confirm_path.exists() else pd.DataFrame()
    full_expected = np.load(RESULTS / 'full_fit.npz')['prediction']
    baseline_oof = np.load(RESULTS / 'v5_18421.npz')['oof'] if (RESULTS / 'v5_18421.npz').exists() else None
    if baseline_oof is None:
        baseline_oof = Run(BASE).baseline(18421)
    new_oof = np.load(RESULTS / f"{choice['name']}_18421.npz")['oof'] if choice['weight'] else baseline_oof
    expected_oof = (1 - choice['weight']) * baseline_oof + choice['weight'] * new_oof
    cells = []

    def md(source):
        cells.append(nb.v4.new_markdown_cell(textwrap.dedent(source).strip()))

    def code(source):
        cells.append(nb.v4.new_code_cell(textwrap.dedent(source).strip()))

    md(f'''# Restaurant Annual Turnover — Improvement 6

**Student:** submission · **Register number:** submission · **Team:** submission

Prepared 26 October 2026. In Colab select a CPU runtime and run all cells. Upload
the four official CSVs when prompted. Save the executed notebook using File → Download.

**Measured decision: {decision['decision']}.** Selected predictor: `{choice['name']}`;
new-model weight: {choice['weight']:.0%}. The frozen v5 portal score is
**12,046,722.3691139**, as confirmed by the user. The shown leader is **4,796,740.5600053**.
This notebook does not claim an unobserved portal score or rank.
''')
    md('''## Runtime and input files

This notebook embeds all predictor source code. It does not require older improvement
folders, prediction caches, local helper scripts or a Downloads directory. It installs
the measured package versions when necessary. Fresh fitting is performed below.
''')
    versions = audit['runtime']
    packages = {'numpy': versions['numpy'], 'pandas': versions['pandas'],
                'scikit-learn': versions['sklearn'], 'catboost': versions['catboost'],
                'lightgbm': versions['lightgbm'], 'scipy': versions['scipy']}
    code('''
import os, sys, json, hashlib, base64, zlib, types, subprocess, importlib.metadata
from pathlib import Path
''' + '\nPACKAGES = ' + repr(packages) + '''
to_install = []
for package, version in PACKAGES.items():
    try:
        installed = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed != version:
        to_install.append(f'{package}=={version}')
if to_install:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *to_install])
try:
    import matplotlib.pyplot as plt
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'matplotlib'])
    import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
print('Python:', sys.version.split()[0])
print('Packages:', {name: importlib.metadata.version(name) for name in PACKAGES})
''')
    code('''
REQUIRED = ['Train_dataset.csv', 'Test_dataset.csv', 'Data_Dictionary.csv', 'Sample_Solution.csv']
DATA_DIR = Path(os.environ.get('MLA_DATA_DIR', '.')).resolve()
OUTPUT_DIR = Path(os.environ.get('MLA_OUTPUT_DIR', '.')).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if not all((DATA_DIR / name).is_file() for name in REQUIRED):
    try:
        from google.colab import files
    except ImportError:
        raise FileNotFoundError('Place the four official CSVs in the current folder or set MLA_DATA_DIR.')
    files.upload()
    DATA_DIR = Path.cwd()
assert all((DATA_DIR / name).is_file() for name in REQUIRED)
INPUT_HASHES = {name: hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest() for name in REQUIRED}
''' + '\nEXPECTED_INPUTS = ' + repr(audit['inputs']) + '''
assert INPUT_HASHES == EXPECTED_INPUTS, 'These are not the four measured competition inputs.'
print('PASS: all four input SHA256 hashes match the measured experiment.')
''')
    md('''## Frozen reference and six bounded hypotheses

The following source is embedded from the audited implementation. The v5 module is
isolated to keep its legacy functions separate from new predictor functions.
The six settings use native-categorical LightGBM (L2/Huber, 7/15 leaves), a square-root
CatBoost model corrected using three inner OOF folds, and bounded feature-based
distribution weights for Huber CatBoost. All scores retain original INR targets.
''')
    source = (ROOT / 'reference.py').read_text()
    code('REFERENCE_SOURCE = ' + repr(source) + '''
ref = types.ModuleType('reference')
exec(compile(REFERENCE_SOURCE, '<embedded frozen v5>', 'exec'), ref.__dict__)
sys.modules['reference'] = ref
print('Frozen reference source SHA256:', hashlib.sha256(REFERENCE_SOURCE.encode()).hexdigest())
''')
    code((ROOT / 'modeling.py').read_text())
    code('CONTRACT = json.loads(' + repr(json.dumps(contract)) + ')\n'
         + 'CHOICE = json.loads(' + repr(json.dumps(choice)) + ')\n' + '''
train, test, y = load_data(DATA_DIR)
ref.get_splits = splits
assert np.isclose(ref.rmse([0, 3], [0, 0]), np.sqrt(4.5))
changed = train.copy()
changed['Annual Turnover'] = 0
changed['Registration Number'] = -999
pd.testing.assert_frame_equal(feature_frame(train), feature_frame(changed))
assert feature_frame(train).columns.equals(feature_frame(test).columns)
print('Train:', train.shape, 'Test:', test.shape)
print('PASS: metric, target/ID exclusion, matching feature schema and input alignment.')
print('Chosen configuration:', json.dumps(CHOICE, indent=2))
''')
    md('''## Recorded development and locked confirmation

These are the actual completed experiment results, embedded as records for review.
The next section independently retrains one complete five-fold comparison without
using prediction caches. The development assignments are 42/71; confirmation uses
18421/18422 and cleaned city groups. All assignments reuse previously examined labels.
''')
    code('development_table = pd.DataFrame(json.loads(' + repr(screening.to_json(orient='records', double_precision=15)) + '))\n'
         + 'confirmation_table = pd.DataFrame(json.loads(' + repr(confirmation.to_json(orient='records', double_precision=15)) + '))\n'
         + 'DECISION = json.loads(' + repr(json.dumps(decision)) + ')\n' + '''
best_by_family = development_table.sort_values('mean_rmse').groupby('family', sort=False).head(1)
print('Best development setting per family (positive gain is better):')
print(best_by_family[['name', 'weight', 'gain_42', 'gain_71']].to_string(index=False))
print()
print('Locked confirmation:')
print(confirmation_table.to_string(index=False) if len(confirmation_table) else 'No candidate passed development.')
print()
print('Decision:', DECISION['decision'])
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
positions = np.arange(len(best_by_family))
axes[0].barh(positions - .16, best_by_family['gain_42'] / 1000, height=.3, label='Seed 42', color='#2563eb')
axes[0].barh(positions + .16, best_by_family['gain_71'] / 1000, height=.3, label='Seed 71', color='#0f766e')
axes[0].set_yticks(positions, best_by_family['name'])
axes[0].axvline(0, color='#334155')
axes[0].set(xlabel='RMSE improvement over v5 (INR thousands)', title='Recorded development comparisons')
axes[0].legend()
axes[1].hist(y / 1e6, bins=45, color='#2563eb', edgecolor='white')
axes[1].set(xlabel='Annual turnover (INR millions)', ylabel='Restaurants', yscale='log', title='All original labels remain in scoring')
fig.tight_layout()
plt.show()
''')
    md('''## Fresh five-fold reproduction

This cell trains every fold of assignment 18421 again, including the frozen v5
reference and the selected new component when applicable. The compressed reference
arrays below are used **only after fitting to check reproduction**; they never supply
model predictions. The assertion requires maximum differences below INR 0.01.
''')
    code('EXPECTED_BASELINE_OOF = ' + array_literal(baseline_oof) + '\n'
         + 'EXPECTED_SELECTED_OOF = ' + array_literal(expected_oof) + '\n' + '''
fresh_baseline = ref.evaluate(train, y, 18421, CONTRACT)['candidate']
assert np.max(np.abs(fresh_baseline - EXPECTED_BASELINE_OOF)) < .01
if CHOICE['weight']:
    fresh_component, _, fresh_diagnostics = component_oof(
        train, test, y, SPECS[CHOICE['name']], 18421, print)
    fresh_selected = (1 - CHOICE['weight']) * fresh_baseline + CHOICE['weight'] * fresh_component
else:
    fresh_selected = fresh_baseline.copy()
assert np.max(np.abs(fresh_selected - EXPECTED_SELECTED_OOF)) < .01
print('PASS: freshly trained complete five-fold predictions reproduce the independent reference within INR 0.01.')
fresh_validation_rows = []
for segment, mask in [('all', np.ones(len(y), bool)), ('<=40M', y <= 40e6),
                      ('40-80M', (y > 40e6) & (y <= 80e6)), ('>80M', y > 80e6)]:
    fresh_validation_rows.append({'segment': segment, 'rows': int(mask.sum()),
        'v5_RMSE': ref.rmse(y[mask], fresh_baseline[mask]),
        'selected_RMSE': ref.rmse(y[mask], fresh_selected[mask])})
print(pd.DataFrame(fresh_validation_rows).to_string(index=False))
''')
    md('''## Fresh full fit, behavioral checks and CSV export

All selected models are fitted again on the official training data. Expected test
predictions are numerical reproduction references, not hidden target labels.
The final file keeps the original 500 restaurant IDs in their original order.
''')
    tree = ast.parse((ROOT / 'workflow.py').read_text())
    behavior_source = next(ast.get_source_segment((ROOT / 'workflow.py').read_text(), node)
                           for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'behavior_checks')
    code(behavior_source)
    code('EXPECTED_FULL_FIT = ' + array_literal(full_expected) + '\n' + '''
test_prediction, fitted = fit_final(train, test, y, CHOICE, CONTRACT)
maximum_difference = float(np.max(np.abs(test_prediction - EXPECTED_FULL_FIT)))
assert maximum_difference < .01, maximum_difference
checks = behavior_checks(fitted, test, test_prediction)
output_path = OUTPUT_DIR / 'submission.csv'
submission = pd.DataFrame({'Registration Number': test['Registration Number'], 'Annual Turnover': test_prediction})
submission.to_csv(output_path, index=False)
roundtrip = pd.read_csv(output_path)
ref.check_submission(roundtrip, test)
assert np.max(np.abs(roundtrip['Annual Turnover'].to_numpy() - test_prediction)) < .01
assert len(output_path.read_text().splitlines()) == 501
print('PASS: fresh full fit reproduces the independent export; maximum difference INR', maximum_difference)
print('Behavior checks:', json.dumps(checks, indent=2))
print('CSV:', output_path.name)
print('SHA256:', hashlib.sha256(output_path.read_bytes()).hexdigest())
print(submission.head().to_string(index=False))
print(submission['Annual Turnover'].describe().to_string())
''')
    md('''## Interpretation and portal use

Lower RMSE is better. These are robustness comparisons on reused training labels,
not an estimate of a new portal score. The portal's scoring subset, remaining
submission allowance and deadline timezone still require checking in the portal.
No account access or upload is performed by this notebook.

Use at most two qualified portal trials. Record the exact CSV hash and returned
RMSE, retaining the lowest observed score. If the decision is **retain portal-scored
v5**, the exported model reproduces the existing scored solution; do not spend a
portal trial uploading it again. A large prediction change does not establish accuracy.

Native categories: https://lightgbm.readthedocs.io/en/stable/Advanced-Topics.html

Validation: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
''')
    code('''
try:
    from google.colab import files
except ImportError:
    print('Local output saved:', output_path)
else:
    files.download(str(output_path))
''')
    book = nb.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': audit['runtime']['python']},
        'colab': {'name': STEM + '.ipynb', 'provenance': []}})
    nb.validate(book)
    nb.write(book, ROOT / f'{STEM}.ipynb')


def finalize():
    decision = json.loads((RESULTS / 'decision.json').read_text())
    audit = json.loads((RESULTS / 'audit.json').read_text())
    execution = json.loads((RESULTS / 'notebook_execution.json').read_text())
    checks = json.loads((RESULTS / 'submission_checks.json').read_text())
    screening = pd.read_csv(RESULTS / 'screening.csv')
    best = screening.sort_values('mean_rmse').groupby('name', sort=False).head(1)
    path = RESULTS / 'confirmation.csv'
    confirmation = pd.read_csv(path) if path.exists() else pd.DataFrame()
    choice = decision['choice']
    # Keep diagnostics for losing development settings as well as any finalists.
    diagnostic_run = Run(BASE)
    development_segments, development_folds = [], []
    for setting in screening.sort_values('mean_rmse').groupby('name', sort=False).head(1).to_dict(orient='records'):
        for scheme in [42, 71]:
            b = diagnostic_run.read_cache('v5', scheme)
            component = diagnostic_run.read_cache(setting['name'], scheme)
            p = (1 - setting['weight']) * b + setting['weight'] * component
            for label, mask in [('<=40M', diagnostic_run.y <= 40e6),
                                ('40-80M', (diagnostic_run.y > 40e6) & (diagnostic_run.y <= 80e6)),
                                ('>80M', diagnostic_run.y > 80e6)]:
                development_segments.append({'name': setting['name'], 'weight': setting['weight'],
                    'scheme': scheme, 'segment': label, 'rows': int(mask.sum()),
                    'v5_rmse': float(np.sqrt(np.mean((diagnostic_run.y[mask] - b[mask]) ** 2))),
                    'candidate_rmse': float(np.sqrt(np.mean((diagnostic_run.y[mask] - p[mask]) ** 2))),
                    'candidate_sse_share': float(np.sum((diagnostic_run.y[mask] - p[mask]) ** 2) / np.sum((diagnostic_run.y - p) ** 2))})
            for fold, (_, indices) in enumerate(splits(diagnostic_run.train, scheme)):
                development_folds.append({'name': setting['name'], 'weight': setting['weight'],
                    'scheme': scheme, 'fold': fold, 'rows': len(indices),
                    'tail_rows': int((diagnostic_run.y[indices] > 80e6).sum()),
                    'v5_rmse': float(np.sqrt(np.mean((diagnostic_run.y[indices] - b[indices]) ** 2))),
                    'candidate_rmse': float(np.sqrt(np.mean((diagnostic_run.y[indices] - p[indices]) ** 2)))})
    pd.DataFrame(development_segments).to_csv(RESULTS / 'development_segments.csv', index=False)
    pd.DataFrame(development_folds).to_csv(RESULTS / 'development_folds.csv', index=False)
    if not decision['candidates']:
        pd.DataFrame(columns=['name', 'scheme', 'segment', 'rows', 'v5_rmse', 'candidate_rmse',
                              'candidate_sse_share']).to_csv(RESULTS / 'segments.csv', index=False)
        pd.DataFrame(columns=['name', 'scheme', 'fold', 'rows', 'tail_rows', 'v5_rmse',
                              'candidate_rmse']).to_csv(RESULTS / 'folds.csv', index=False)
    # Preserve the scored files and original sources; never promote to the workspace root.
    for name, expected in audit['preserved_v5'].items():
        assert sha(BASE / 'improvement_v5' / name) == expected
        assert sha(ROOT / 'scored_v5_backup' / name) == expected
    for name, expected in audit['source_provenance']['sources'].items():
        assert sha(BASE / name) == expected
    conclusion = ('The candidate passes the predeclared checks and is qualified for one portal trial. '
                  'Its portal RMSE is unknown.' if choice['weight'] else
                  'No new candidate passed the predeclared gates. Keep the portal-scored v5 model. '
                  'The delivered CSV is a byte-identical copy of v5; uploading it again would not improve the score.')
    lines = [
        '# Improvement 6 — measured result', '',
        '**Student:** submission · submission · **Team:** submission', '',
        f'**Decision: {decision["decision"]}.** {conclusion}', '',
        '## Score context', '',
        '- User-confirmed v5 portal RMSE: **12,046,722.3691139**.',
        '- Second place in the supplied screenshot: **11,588,949.5634295**; first: **4,796,740.5600053**.',
        '- Required reductions from v5: **3.80%** and **60.18%**, respectively. No new portal score or rank is claimed.', '',
        '## Completed experiments', '',
        'All six predeclared settings were evaluated on both five-fold development assignments (42/71), '
        'with two model seeds. Each setting was tested alone and at 25%, 50%, 75% weight with v5. '
        'All validation targets remained unchanged and were scored in INR. The table shows the best '
        'development blend for each of the six settings.', '',
        table(best[['name', 'weight', 'gain_42', 'gain_71', 'mean_rmse']]), '',
        'Positive gain means lower RMSE than v5. Full results, including all losing settings, are in `results/screening.csv`.', '',
        'The frozen-v5 implementation was freshly refitted on both development assignments and reproduced '
        'the historical predictions exactly (maximum difference INR 0.00). Any reused confirmation baseline '
        'is checked against original input, source, contract and file hashes; assignment 18421 is additionally '
        'retrained in the delivered notebook.', '',
        '## Frozen confirmation', '',
        table(confirmation) if len(confirmation) else 'No setting passed both development assignments; no finalist was advanced.', '',
        'Acceptance requires positive gain on both random confirmation assignments and no more than 1% '
        'city-group RMSE deterioration. The development shortlist was locked before any new confirmation. '
        'For any city comparison, groups use the existing v5 clean alias map and v5 is refitted on exactly those same groups. '
        'The map remains approximate for unrecognized spellings.', '',
        'Decision details:' if decision['candidates'] else 'The empty shortlist triggered the planned v5 fallback; no new confirmation or portal trial was warranted.', '',
        *[f'- `{c["name"]}` at {c["weight"]:.0%}: **{"PASS" if c["accepted"] else "FAIL"}**; '
          f'mean random-confirmation gain INR {c["mean_confirmation_gain"]:,.2f}; '
          f'city change {100*c["city_relative_increase"]:+.4f}%.' for c in decision['candidates']], '',
        'Turnover-bin RMSE and squared-error shares for all six settings are in `results/development_segments.csv`; '
        'fold and tail counts are in `results/development_folds.csv`. For any finalists, corresponding confirmation '
        'diagnostics are in `results/segments.csv` and `results/folds.csv`. No subset diagnostic replaces full RMSE.', '',
        '## Correctness and reproducibility', '',
        '- All four original CSV hashes match the earlier measured inputs.',
        '- Data directory is explicit; defaults to the workspace. No Downloads path is required.',
        '- Native LightGBM category vocabularies and early stopping are fitted inside training folds; '
        'early stopping uses original-scale-equivalent RMSE on targets scaled by one million.',
        '- Square-root correction uses three inner OOF folds; outer validation labels are excluded.',
        '- Distribution weights use cross-fitted train/test membership probabilities, without target or ID; '
        'density ratios are clipped to 0.5–2.0, then normalized. Supplied unlabeled test covariates are used only by this experiment.',
        f'- Fresh v5 full refit maximum difference from scored CSV: **INR {checks["v5_full_refit_max_difference"]:.8f}**.',
        f'- Standalone notebook: **{execution["code_cells"]} code cells executed in a new process**, '
        'one complete five-fold assignment freshly retrained without caches, and all selected full fits freshly retrained.',
        f'- Notebook full-fit maximum prediction difference: **INR {execution["maximum_full_fit_difference"]:.8f}**.',
        '- Passed row permutation, missing values, unseen city, target/ID invariance, CSV round-trip, '
        f'finite nonnegative predictions and {checks["malformed_cases_rejected"]} malformed-file rejections.',
        '- Final CSV: **500 rows × 2 columns**, original restaurant IDs/order, 501 lines including header.', '',
        '## Limits and submission handling', '',
        'Every split reuses labels examined in previous rounds. Results are robustness evidence, not an untouched '
        'external test or an unbiased assessment of the complete historical search. Local RMSE must not be '
        'subtracted from the portal score to forecast a new score.', '',
        'The portal scoring subset, remaining allowance and deadline timezone are unverified. **No portal '
        'uploads were performed.** The maximum planned allowance is two qualified trials. Preserve exact CSV hashes '
        'and returned portal scores in `portal_trials.csv`. No unconditional prediction-distance score bound is applied.', '',
        '## Files', '',
        f'- `{STEM}.csv` — recommended submission or preserved fallback, according to the decision above.',
        f'- `{STEM}.ipynb` — standalone executed Colab-compatible notebook.',
        '- `scored_v5_backup/` — exact scored v5 CSV and matching original notebook.',
        '- `results/` — complete predictions, fold diagnostics, frozen lock and execution evidence.',
        '- `delivery_manifest.json` — hashes, versions, decision and provenance.', '',
        f'CSV SHA256: `{sha(ROOT / (STEM + ".csv"))}`', '',
        'Primary implementation references: [LightGBM native categories](https://lightgbm.readthedocs.io/en/stable/Advanced-Topics.html), '
        '[scikit-learn validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).', '',
    ]
    (ROOT / 'REPORT.md').write_text('\n'.join(lines))
    manifest = {'decision': decision, 'inputs': audit['inputs'], 'runtime': audit['runtime'],
                'preserved_v5': audit['preserved_v5'], 'checks': checks, 'notebook_execution': execution,
                'portal_uploads': 0, 'maximum_portal_trials': 2, 'portal_v5_rmse': audit['portal_v5_rmse'],
                'outputs': {name: sha(ROOT / name) for name in [STEM + '.csv', STEM + '.ipynb', 'REPORT.md']},
                'implementation': {p.name: sha(p) for p in ROOT.glob('*.py')},
                'candidate_lock_sha256': sha(RESULTS / 'candidate_lock.json'),
                'experiment_plan_sha256': sha(ROOT / 'experiment_plan.json')}
    save_json(ROOT / 'delivery_manifest.json', manifest)
    trial_log = ROOT / 'portal_trials.csv'
    if not trial_log.exists():
        pd.DataFrame([{'version': 'v5', 'csv_sha256': audit['preserved_v5'][STEM + '.csv'],
                       'portal_rmse': audit['portal_v5_rmse'], 'rank': '', 'submitted_at': '',
                       'source': 'user confirmed 26 October 2026'}]).to_csv(trial_log, index=False)
    (ROOT / 'README.md').write_text('''# Improvement 6

Read REPORT.md first: it states whether a new candidate passed or v5 is retained.
The executed notebook is standalone and accepts the four official competition CSVs.
In Colab choose CPU, run all, then download the CSV and executed notebook.

Local reproduction requires the versions recorded in results/audit.json. The launcher
selects the bundled desktop Python 3.12 when present and configures the existing
workspace dependencies and OpenMP library without changing the system.

```sh
python3 improvement_v6/run_local.py --data-dir /path/to/competition/data
```

Alternatively, run these stages with your compatible configured Python environment:

```sh
python prepare_reference.py
python workflow.py all --data-dir /path/to/competition/data
python deliver.py build
python execute_notebook.py --data-dir /path/to/competition/data
python deliver.py finalize
```

The default data directory for workflow.py and execute_notebook.py is the workspace.
Existing matching prediction caches are hash-checked; stale caches raise an error.
The frozen lock prevents changing the shortlist when resuming. To run an independent
new research round, use a new output directory/version rather than replacing results.
The saved notebook uses no prediction caches and requires no helper Python files.
No portal upload is performed. Check portal scoring details, remaining submissions
and deadline timezone before any qualified trial. Keep the original v5 files.
''')
    print(json.dumps({'decision': decision['decision'], 'outputs': manifest['outputs']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['build', 'finalize'])
    args = parser.parse_args()
    build_notebook() if args.stage == 'build' else finalize()
