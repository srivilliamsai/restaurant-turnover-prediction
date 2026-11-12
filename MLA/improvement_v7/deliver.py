"""Create portable Colab research and independently executable candidate notebooks."""
import argparse
import ast
import base64
import json
import textwrap
import zlib
from pathlib import Path
import numpy as np
import pandas as pd
import nbformat as nb
from workflow import ROOT, BASE, OUT, STEM, sha, save


def encoded(values):
    payload = base64.b64encode(zlib.compress(np.asarray(values, dtype='<f8').tobytes())).decode()
    return "np.frombuffer(zlib.decompress(base64.b64decode(" + repr(payload) + ")), dtype='<f8').copy()"


def source_function(name):
    source = (ROOT / 'workflow.py').read_text()
    tree = ast.parse(source)
    return next(ast.get_source_segment(source, n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def versions():
    v = json.loads((OUT / 'audit.json').read_text())['runtime']
    return {name: v[key] for name, key in [('numpy', 'numpy'), ('pandas', 'pandas'),
            ('scikit-learn', 'sklearn'), ('catboost', 'catboost'), ('scipy', 'scipy')]}


def package_cell():
    return '''import os, sys, json, hashlib, base64, zlib, types, subprocess, importlib.metadata
from pathlib import Path
''' + '\nPACKAGES = ' + repr(versions()) + '''
install = []
for name, version in PACKAGES.items():
    try:
        installed = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed != version:
        install.append(f'{name}=={version}')
if install:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *install])
import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'matplotlib'])
    import matplotlib.pyplot as plt
print('Python:', sys.version.split()[0])
print({p: importlib.metadata.version(p) for p in PACKAGES})
'''


def book_builder():
    cells = []

    def md(s):
        cells.append(nb.v4.new_markdown_cell(textwrap.dedent(s).strip()))

    def code(s):
        cells.append(nb.v4.new_code_cell(textwrap.dedent(s).strip()))

    return cells, md, code


def write_book(cells, path):
    book = nb.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': json.loads((OUT / 'audit.json').read_text())['runtime']['python']},
        'colab': {'name': path.name, 'provenance': []}})
    for i, cell in enumerate(cells):
        if cell.cell_type == 'code':
            compile(cell.source, f'generated_cell_{i}', 'exec')
    nb.validate(book)
    nb.write(book, path)


def data_cell():
    inputs = json.loads((OUT / 'audit.json').read_text())['inputs']
    return '''DATA_DIR = Path(os.environ.get('MLA_DATA_DIR', '.')).resolve()
OUTPUT_DIR = Path(os.environ.get('MLA_OUTPUT_DIR', '.')).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REQUIRED = ['Train_dataset.csv', 'Test_dataset.csv', 'Data_Dictionary.csv', 'Sample_Solution.csv']
if not all((DATA_DIR / n).is_file() for n in REQUIRED):
    try:
        from google.colab import files
    except ImportError:
        raise FileNotFoundError('Place the four official CSVs here, or set MLA_DATA_DIR.')
    files.upload()
    DATA_DIR = Path.cwd()
INPUT_HASHES = {n: hashlib.sha256((DATA_DIR / n).read_bytes()).hexdigest() for n in REQUIRED}
''' + '\nEXPECTED_INPUT_HASHES = ' + repr(inputs) + '''
assert INPUT_HASHES == EXPECTED_INPUT_HASHES, 'Input files differ from the measured experiment.'
print('PASS: four official input files and their SHA256 hashes.')
'''


def build_candidate(index, choice):
    cells, md, code = book_builder()
    trial = f'trial_{index}'
    audit = json.loads((OUT / 'audit.json').read_text())
    screening = pd.read_csv(OUT / 'screening.csv')
    confirmation = pd.read_csv(OUT / 'confirmation.csv')
    expected = np.load(OUT / f'full_fit_{trial}.npz')
    baseline_oof = np.load(OUT / 'cache/v5_18421.npz')['prediction']
    component_oof = np.load(OUT / f"{choice['name']}_18421.npz")['oof']
    reference_source = (ROOT / 'reference.py').read_text()
    models_source = (ROOT / 'models.py').read_text()
    contract = json.loads((ROOT / 'v5_contract.json').read_text())
    md(f'''# Restaurant Turnover — V7 {trial}

**Student:** submission · **Register number:** submission · **Team:** submission

Predictor: **{choice['name']}**, new-component weight **{choice['weight']:.0%}**.
Local validation status: **{'local champion' if choice['local_champion'] else 'exploratory candidate with mixed validation'}**.

This is a model-generated experimental CSV, not a confirmed portal improvement.
The user-confirmed v5 score is **12,046,722.3691139**. Only a returned portal RMSE
can determine whether this candidate improves it. Select a CPU runtime in Colab,
run all cells, upload the four official CSVs when asked, and save the executed notebook.
Submit the final coding assignment only once, after choosing the best portal-scored CSV.
''')
    md('''## Runtime and data

The notebook embeds the exact predictor implementation and frozen v5 reference.
It requires no earlier improvement folders. Checkpoints are saved in the output
folder; a browser disconnect can resume them while that runtime survives. For
persistence across runtime deletion, choose `USE_GOOGLE_DRIVE=True` below and
authorize Drive yourself. Free Colab runtime duration and resources are variable:
https://research.google.com/colaboratory/faq.html
''')
    code(package_cell())
    code('''USE_GOOGLE_DRIVE = False
if USE_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
''' + f"    os.environ['MLA_OUTPUT_DIR'] = '/content/drive/MyDrive/MLA_v7/{trial}'\n" + data_cell())
    code('REFERENCE_SOURCE = ' + repr(reference_source) + '''
ref = types.ModuleType('reference')
exec(compile(REFERENCE_SOURCE, '<frozen v5>', 'exec'), ref.__dict__)
sys.modules['reference'] = ref
''')
    code(models_source)
    code('CONTRACT = json.loads(' + repr(json.dumps(contract)) + ')\n'
         + 'CHOICE = json.loads(' + repr(json.dumps(choice)) + ')\n' + '''
train, test, y = load_data(DATA_DIR)
changed = train.copy()
changed['Annual Turnover'] = 1e30
changed['Registration Number'] = -7
pd.testing.assert_frame_equal(features(changed), features(train))
assert features(train).columns.equals(features(test).columns)
assert np.isclose(ref.rmse([0, 3], [0, 0]), np.sqrt(4.5))
print('Train:', train.shape, 'Test:', test.shape)
print('PASS: source alignment, metric, target/ID exclusion and feature schema.')
''')
    md('''## Actual recorded comparisons

These tables contain the completed development and frozen confirmation results.
They are not new scores from this execution. The next cells retrain one complete
five-fold comparison and the full predictor, then verify their predictions.
All metrics retain the original INR targets. Turnover-band classification is an
internal modeling technique; its output is converted into continuous expected turnover.
''')
    code('development = pd.DataFrame(json.loads(' + repr(screening.to_json(orient='records', double_precision=15)) + '))\n'
         + 'confirmation = pd.DataFrame(json.loads(' + repr(confirmation.to_json(orient='records', double_precision=15)) + '))\n' + '''
best = development.sort_values('mean_rmse').groupby('family', sort=False).head(1)
print('Best development setting per family:')
print(best[['name', 'weight', 'gain_42', 'gain_71']].to_string(index=False))
print('Confirmation for this candidate:')
print(confirmation[confirmation['name'] == CHOICE['name']].to_string(index=False))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
positions = np.arange(len(best))
axes[0].barh(positions - .16, best['gain_42'] / 1000, height=.3, label='42', color='#2563eb')
axes[0].barh(positions + .16, best['gain_71'] / 1000, height=.3, label='71', color='#0f766e')
axes[0].set_yticks(positions, best['family'])
axes[0].axvline(0, color='#334155')
axes[0].set(xlabel='RMSE improvement over v5 (INR thousands)', title='Recorded development results')
axes[0].legend(title='Assignment')
axes[1].hist(y / 1e6, bins=45, color='#2563eb', edgecolor='white')
axes[1].set(xlabel='Annual turnover (INR millions)', ylabel='Restaurants', yscale='log', title='All original labels remain in scoring')
fig.tight_layout()
plt.show()
''')
    md('''## Fresh five-fold reproduction, with resumable checkpoints

Compressed reference arrays are used only to check freshly trained results. They
are never substituted for model predictions. Existing checkpoints are reused only
when input, source, configuration, fold and file hashes match. Delete the checkpoint
folder or set `REUSE_CHECKPOINTS=False` for a completely fresh repetition.

Residual models generate three-fold v5 predictions inside each outer training fold.
No outer validation labels enter the residual model or its preprocessing.
''')
    code('EXPECTED_BASELINE_OOF = ' + encoded(baseline_oof) + '\n'
         + 'EXPECTED_COMPONENT_OOF = ' + encoded(component_oof) + '\n'
         + 'SOURCE_HASHES = ' + repr({'reference': sha(ROOT / 'reference.py'), 'models': sha(ROOT / 'models.py')}) + '\n' + '''
REUSE_CHECKPOINTS = os.environ.get('V7_REPRODUCTION_FORCE_FRESH') != '1'
checkpoint_root = OUTPUT_DIR / 'validation_checkpoints'
checkpoint_root.mkdir(parents=True, exist_ok=True)
cache_reads = 0
fresh_baseline = np.zeros(len(y))
fresh_component = np.zeros(len(y))
for fold, (a, b) in enumerate(split_indices(train, 18421)):
    checkpoint = checkpoint_root / f'fold_{fold}.npz'
    meta = checkpoint.with_suffix('.json')
    context = {'inputs': INPUT_HASHES, 'sources': SOURCE_HASHES, 'choice': CHOICE,
               'contract': CONTRACT, 'packages': PACKAGES, 'fold': fold,
               'train_ids': train.iloc[a]['Registration Number'].tolist(),
               'valid_ids': train.iloc[b]['Registration Number'].tolist()}
    if REUSE_CHECKPOINTS and checkpoint.exists() and meta.exists():
        saved = json.loads(meta.read_text())
        assert saved['context'] == context
        assert saved['sha256'] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        cached = np.load(checkpoint, allow_pickle=False)
        baseline_part, component_part = cached['baseline'], cached['component']
        cache_reads += 1
        print('Verified checkpoint:', fold)
    else:
        print('Fresh outer fold:', fold + 1, flush=True)
        baseline_part = v5_fold_prediction(train.iloc[a], y[a], train.iloc[b], CONTRACT, fold)
        inner = inner_v5_oof(train.iloc[a], y[a], CONTRACT, 731 + fold, print) if CHOICE['family'] == 'residual' else None
        one = []
        for seed in SEEDS:
            model = fit_component(train.iloc[a], y[a], SPECS[CHOICE['name']], seed + fold, inner)
            one.append(predict_component(model, train.iloc[b], baseline_part))
        component_part = np.mean(one, axis=0)
        temporary = checkpoint.with_suffix('.partial.npz')
        np.savez_compressed(temporary, baseline=baseline_part, component=component_part)
        temporary.replace(checkpoint)
        meta.write_text(json.dumps({'context': context, 'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}))
    fresh_baseline[b], fresh_component[b] = baseline_part, component_part
assert np.max(np.abs(fresh_baseline - EXPECTED_BASELINE_OOF)) < .01
assert np.max(np.abs(fresh_component - EXPECTED_COMPONENT_OOF)) < .01
fresh_selected = (1 - CHOICE['weight']) * fresh_baseline + CHOICE['weight'] * fresh_component
print('PASS: complete five-fold prediction reproduction within INR 0.01; reused checkpoints:', cache_reads)
fresh_segments = []
for name, mask in [('all', np.ones(len(y), bool)), ('<=40M', y <= 40e6),
                   ('40-80M', (y > 40e6) & (y <= 80e6)), ('>80M', y > 80e6)]:
    fresh_segments.append({'segment': name, 'rows': int(mask.sum()),
        'v5_rmse': ref.rmse(y[mask], fresh_baseline[mask]),
        'candidate_rmse': ref.rmse(y[mask], fresh_selected[mask])})
print(pd.DataFrame(fresh_segments).to_string(index=False))
''')
    md('''## Fresh final fit and CSV checks

Every selected full-data model is trained again. Expected test predictions are
reproduction references, not hidden test targets. The score bound below assumes
the same 500 rows and ordinary RMSE stated in the competition instructions; it is
a necessary condition for beating the shown leader, not evidence of accuracy.
''')
    code(source_function('check_behavior'))
    code('EXPECTED_TEST = ' + encoded(expected['prediction']) + '\n'
         + 'EXPECTED_V5 = ' + encoded(expected['baseline']) + '\n' + '''
full_inner = inner_v5_oof(train, y, CONTRACT, 731, print) if CHOICE['family'] == 'residual' else None
test_prediction, baseline_test, fitted = fit_submission(train, test, y, CONTRACT, CHOICE, full_inner)
maximum_full_fit_difference = float(np.max(np.abs(test_prediction - EXPECTED_TEST)))
assert maximum_full_fit_difference < .01
assert np.max(np.abs(baseline_test - EXPECTED_V5)) < .01
checks = check_behavior(fitted, test, test_prediction)
output_path = OUTPUT_DIR / 'submission.csv'
pd.DataFrame({'Registration Number': test['Registration Number'], 'Annual Turnover': test_prediction}).to_csv(output_path, index=False)
roundtrip = pd.read_csv(output_path)
ref.check_submission(roundtrip, test)
assert np.max(np.abs(roundtrip['Annual Turnover'].to_numpy() - test_prediction)) < .01
assert len(output_path.read_text().splitlines()) == 501
rms_change = float(np.sqrt(np.mean((test_prediction - baseline_test) ** 2)))
lower_bound = max(0, 12046722.3691139 - rms_change)
print('PASS: fresh final fit, CSV round-trip and behavioral checks.')
print('Maximum full-fit difference (INR):', maximum_full_fit_difference)
print('Checks:', json.dumps(checks, indent=2))
print('CSV SHA256:', hashlib.sha256(output_path.read_bytes()).hexdigest())
print('Prediction RMS change from v5:', rms_change)
print('Conditional portal RMSE lower bound:', lower_bound)
print('Can mathematically beat the shown 4.7967M leader:', lower_bound < 4796740.5600053)
print('Actual candidate portal RMSE: UNKNOWN until submitted.')
print(roundtrip.head().to_string(index=False))
''')
    md('''## Choosing the final submission

Upload this trial's CSV to the hackathon, then record its returned RMSE and file
hash. Retain the lowest observed portal score, including the preserved v5 result.
Submit only the matching final executed notebook to the one-time coding assignment
after that choice. Local cross-validation scores do not determine a portal rank.
All validation assignments reuse labels examined in previous work.
''')
    code('''try:
    from google.colab import files
except ImportError:
    print('Local file:', output_path)
else:
    files.download(str(output_path))
''')
    path = ROOT / 'candidates' / trial / f'{STEM}.ipynb'
    write_book(cells, path)
    return path


def build_research():
    cells, md, code = book_builder()
    md('''# V7 restaurant-turnover research — resumable CPU workflow

This notebook reruns the 12 bounded settings, locks one configuration per family,
confirms them and creates up to two experimental CSVs. It uses the four official
inputs and embeds all its own source and the preserved scored v5 reference.
It does not require previous improvement folders or upload anything to Olympus.

CPU is sufficient for this small dataset. Your screenshot shows approximately
12.7 GB of RAM; a GPU subscription is not required for this implementation. Free
Colab runtime availability varies. For checkpoints to survive a deleted runtime,
enable Google Drive below and grant access yourself. Upload the four official input
CSVs again after a runtime reset; completed, hash-verified folds will be reused.
''')
    code(package_cell())
    code('''USE_GOOGLE_DRIVE = False
if USE_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    os.environ['MLA_OUTPUT_DIR'] = '/content/drive/MyDrive/MLA_v7/research'
''' + data_cell())
    source_files = {name: (ROOT / name).read_text() for name in [
        'reference.py', 'models.py', 'workflow.py', 'prepare_resume.py', 'v5_contract.json',
        'reference_provenance.json', 'scored_reference.json']}
    preserved = {p.name: base64.b64encode(p.read_bytes()).decode() for p in (ROOT / 'scored_v5_backup').iterdir() if p.suffix in ['.csv', '.ipynb']}
    code('SOURCE_FILES = json.loads(' + repr(json.dumps(source_files)) + ')\n'
         + 'PRESERVED_V5 = json.loads(' + repr(json.dumps(preserved)) + ')\n' + '''
WORK_ROOT = OUTPUT_DIR / 'improvement_v7'
WORK_ROOT.mkdir(parents=True, exist_ok=True)
for name, source in SOURCE_FILES.items():
    (WORK_ROOT / name).write_text(source)
backup = WORK_ROOT / 'scored_v5_backup'
backup.mkdir(exist_ok=True)
for name, payload in PRESERVED_V5.items():
    (backup / name).write_bytes(base64.b64decode(payload))
print('Research root:', WORK_ROOT)
print('Resume checkpoints:', WORK_ROOT / 'results/cache')
''')
    md('''## Execute the bounded research

Each stage is a subprocess using the embedded source. Completed folds are reused
only with matching inputs, configurations, sources and file hashes. Interrupted
partial writes are recomputed. No additional model settings are added after the
shortlist is frozen. Depending on CPU availability, nested residual work can take
substantially longer than the simple models.
''')
    for stage in ['audit', 'screen', 'confirm', 'fit']:
        code("subprocess.run([sys.executable, str(WORK_ROOT / 'prepare_resume.py')], check=True)\n"
             + f"subprocess.run([sys.executable, str(WORK_ROOT / 'workflow.py'), {stage!r}, '--data-dir', str(DATA_DIR)], check=True)\n")
    code('''decision = json.loads((WORK_ROOT / 'results/decision.json').read_text())
print(json.dumps(decision, indent=2))
print('Actual portal scores remain unknown until the experimental CSVs are uploaded.')
import zipfile
archive = OUTPUT_DIR / 'V7_research_results.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for path in WORK_ROOT.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            z.write(path, path.relative_to(OUTPUT_DIR))
try:
    from google.colab import files
except ImportError:
    print('Saved research bundle:', archive)
else:
    files.download(str(archive))
''')
    write_book(cells, ROOT / 'V7_Research_Colab.ipynb')


def markdown_table(frame):
    frame = frame.copy()
    for c in frame.select_dtypes('number'):
        frame[c] = frame[c].map(lambda v: f'{v:,.2f}')
    return '| ' + ' | '.join(frame.columns) + ' |\n| ' + ' | '.join(['---'] * len(frame.columns)) + ' |\n' + '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in frame.itertuples(index=False, name=None))


def finalize():
    decision = json.loads((OUT / 'decision.json').read_text())
    audit = json.loads((OUT / 'audit.json').read_text())
    checks = json.loads((OUT / 'submission_checks.json').read_text()) if decision['trials'] else []
    portable = json.loads((OUT / 'portable_and_cache_checks.json').read_text())
    provenance = json.loads((ROOT / 'reference_provenance.json').read_text())
    assert sha(ROOT / 'reference.py') == provenance['reference_sha256']
    screen = pd.read_csv(OUT / 'screening.csv')
    confirmation = pd.read_csv(OUT / 'confirmation.csv')
    best = screen.sort_values('mean_rmse').groupby('family', sort=False).head(1)
    # Conditional paired-row uncertainty; do not use it to retune the frozen choices.
    y = pd.read_csv(BASE / 'Train_dataset.csv')['Annual Turnover'].to_numpy(float)
    intervals = []
    for candidate in decision['candidates']:
        baseline_errors, candidate_errors = [], []
        for scheme in [18421, 18422]:
            b = np.load(OUT / 'cache' / f'v5_{scheme}.npz')['prediction']
            component = np.load(OUT / f'{candidate["name"]}_{scheme}.npz')['oof']
            p = (1 - candidate['weight']) * b + candidate['weight'] * component
            baseline_errors.append((y - b) ** 2)
            candidate_errors.append((y - p) ** 2)
        baseline_errors = np.mean(baseline_errors, axis=0)
        candidate_errors = np.mean(candidate_errors, axis=0)
        rng = np.random.default_rng(8127)
        gains = []
        for _ in range(2000):
            indices = rng.integers(0, len(y), len(y))
            gains.append(np.sqrt(baseline_errors[indices].mean()) - np.sqrt(candidate_errors[indices].mean()))
        low, high = np.quantile(gains, [.025, .975])
        intervals.append({'name': candidate['name'], 'conditional_gain_ci95_low': float(low),
                          'conditional_gain_ci95_high': float(high), 'rows_bootstrapped': len(y),
                          'repetitions': 2000})
    pd.DataFrame(intervals).to_csv(OUT / 'conditional_uncertainty.csv', index=False)
    notebook_results = []
    outputs = {}
    for i, check in enumerate(checks, 1):
        directory = ROOT / 'candidates' / f'trial_{i}'
        execution = json.loads((OUT / f'notebook_execution_trial_{i}.json').read_text())
        assert execution['maximum_full_fit_difference'] < .01 and execution['maximum_oof_difference'] < .01
        notebook_results.append(execution)
        for ext in ['csv', 'ipynb']:
            path = directory / f'{STEM}.{ext}'
            outputs[str(path.relative_to(ROOT))] = sha(path)
    for name, h in audit['preserved_v5'].items():
        assert sha(ROOT / 'scored_v5_backup' / name) == h
        original = BASE / 'improvement_v5' / name
        if original.exists():
            assert sha(original) == h
    lines = ['# Improvement 7 — measured research and experimental portal trials', '',
        '**Student:** submission · submission · **Team:** submission', '',
        f'**Decision:** {decision["decision"]}. Actual new portal scores are unknown. The preserved '
        'v5 score remains **12,046,722.3691139** until a better returned score is observed.', '',
        ('**First-place feasibility:** None of these exports can mathematically reach the shown leader under the stated score assumptions. '
         'They are incremental research trials.' if checks and not any(c['can_mathematically_beat_shown_leader'] for c in checks) else
         '**First-place feasibility:** The prediction-distance check is necessary only; no first-place result is established.'), '',
        '## Data audit and scope', '',
        f'- {audit["target_rows"]:,} training rows; 500 test rows. All original labels retained; **zero verified unit errors**.',
        f'- {audit["tail_above_80M"]} targets exceed INR 80M. Only {audit["nearest_neighbor_tail_above80M"]} of their nearest feature neighbors also exceeds INR 80M; '
        'this is a diagnostic, not proof that any label is wrong.',
        '- No identical full-feature train/test rows. Target and restaurant IDs are excluded from features.',
        '- Used only supplied data and original implementations. No external target matching or portal-fitted row changes.',
        '- User-supplied rules specify RMSE across test predictions, unlimited CSV submissions and one final coding submission.', '',
        '## Completed bounded research', '',
        'Twelve settings: four turnover-band models (depth 3/5, with/without inner-OOF temperature calibration), '
        'four tier-specialist settings (depth 4/6, pooling strengths 100/300), and four residual corrections '
        '(Ridge alpha 100/1000; shallow CatBoost L2/Huber). Two model seeds and both five-fold development '
        'assignments 42/71 were used. Candidate weights were 25%, 50%, 75% and 100%. Ridge fits are deterministic; '
        'their repeated seed fits do not constitute independent evidence.', '',
        'Best development configuration in each family (positive gain means improvement over v5):', '',
        markdown_table(best[['name', 'family', 'weight', 'gain_42', 'gain_71', 'mean_rmse']]), '',
        'All 48 model/blend configurations, including losing settings, are in `results/screening.csv`.', '',
        'Residual training used three-fold v5 predictions generated entirely inside each outer training fold. '
        'These nested predictions were shared only when training rows, inputs, code, configuration and hashes matched. '
        'Tier specialists shrink toward a global model and fall back to it for small or unknown groups. '
        'Band models output probability-weighted turnover means; the final evaluation is regression RMSE.', '',
        '## Frozen confirmation', '', markdown_table(confirmation), '',
        'The best eligible configuration per family was frozen before confirmation on 18421, 18422, '
        'target-stratified five-fold seed 27421 and cleaned-city groups. v5 uses exactly the same folds. '
        'Previously hash-verified actual v5 predictions were reused where the fold assignments matched; '
        'stratified and canonical-city references were freshly trained.', '',
        'A local champion requires positive gain on both random confirmation assignments and city deterioration '
        'no greater than 1%. An exploratory portal trial may have mixed results: random mean deterioration '
        'must be at most 1%, city at most 3%, and stratified at most 2%. These limits were set before confirmation. '
        'Trial order prefers local champions, then the best random confirmation mean, with different families.', '',
        'Turnover-bin errors and squared-error shares are in `results/segments.csv`; fold/tail counts are in `results/folds.csv`.', '',
        '**Material tradeoff:** The residual finalist improves the full metric by reducing higher-turnover errors, '
        'while worsening the <=40M segment on both random confirmation assignments (about INR 214k–238k RMSE). '
        'The tier alternative improves that lower-turnover segment but worsens the full metric. This is why '
        'neither the overall local win nor the segment win guarantees a portal gain on different restaurants.', '',
        'Conditional paired-row bootstrap intervals for the two random confirmation assignments are in '
        '`results/conditional_uncertainty.csv`. They reuse fitted predictions and do not account for historical '
        'model selection, retraining or test-distribution uncertainty; they were not used to retune the shortlist.', '',
        'For the residual finalist, the conditional 95% interval is approximately **INR -30,844 to +84,926** '
        'and includes zero. The repeated positive comparisons are encouraging, but a generalization improvement '
        'is not established conclusively. Treat the export as a portal trial, not a promised better score.', '',
        '## Experimental files and rank-1 feasibility', '']
    for i, check in enumerate(checks, 1):
        choice = check['choice']
        execution = notebook_results[i - 1]
        lines += [f'### Trial {i}: {choice["name"]}, weight {choice["weight"]:.0%}', '',
            f'- Status: **{"local champion" if choice["local_champion"] else "exploratory; mixed local validation"}**.',
            f'- Mean random-confirmation RMSE gain: **INR {choice["mean_confirmation_gain"]:,.2f}**.',
            f'- Prediction RMS difference from scored v5: **INR {check["prediction_rms_difference_from_v5"]:,.2f}**.',
            f'- Conditional portal RMSE lower bound: **INR {check["conditional_portal_rmse_lower_bound"]:,.2f}**.',
            f'- Can mathematically beat the shown 4,796,740.56 leader under the stated assumptions: **{check["can_mathematically_beat_shown_leader"]}**.',
            f'- CSV: `candidates/trial_{i}/{STEM}.csv`; matching executed notebook in the same folder.',
            f'- Fresh notebook: {execution["code_cells"]} code cells, a complete five-fold comparison and full fitting; '
            f'maximum prediction difference **INR {execution["maximum_full_fit_difference"]:.8f}**.',
            f'- Passed permutation, missing/unseen values, target/ID invariance, alignment and {check["malformed_cases_rejected"]} malformed-file rejections.', '']
    lines += [
        'The bound uses `RMSE(new) >= RMSE(v5) - RMS(new_prediction - v5_prediction)`. It assumes '
        'the same 500 restaurants, the supplied v5 score and ordinary unweighted RMSE as stated in the rules. '
        'A candidate must differ by more than INR 7,249,981.81 in RMS terms to be capable of beating the shown leader. '
        'Meeting that condition would not prove accuracy. Small changes may still improve the current score.', '',
        '## Colab, reproducibility and submission', '',
        '- These experiments and candidate-notebook checks ran locally on CPU; no Colab account was accessed.',
        '- `V7_Research_Colab.ipynb` is a portable driver for the same bounded research and is not represented as cloud-executed.',
        '- Candidate notebooks embed all predictor code, require only the four official CSVs, and were executed in fresh local processes.',
        '- Fold checkpoints have input/code/configuration/fold/file hashes and can resume. Use optional Google Drive storage '
        'in Colab to survive runtime deletion; local runtime storage alone is temporary.',
        '- Verified the embedded research source in an isolated folder without earlier projects, checked valid checkpoint reuse, '
        'rejected changed configurations and corrupted contents, and recovered a write missing its completion metadata.',
        '- No paid resources are needed for this implementation. Free Colab resources and lifetime vary; the screenshot estimate is not a guarantee.',
        '- Upload trial CSVs separately and record returned scores in `portal_trials.csv`. Keep the lowest observed score, including v5.',
        '- Submit the one-time coding assignment only after choosing the best scored CSV, using its matching executed notebook.',
        '- No portal uploads were performed. No new score, first place, or superiority over v5 is claimed from an unobserved portal result.', '',
        'Every validation assignment reuses labels examined in earlier rounds. The results are robustness evidence, '
        'not an untouched external test or an unbiased evaluation of the complete historical search.', '',
        'Primary references: [CatBoost probability prediction](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_predict_proba), '
        '[Google Colab runtime/storage limits](https://research.google.com/colaboratory/faq.html).', '']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines))
    outputs.update({'REPORT.md': sha(ROOT / 'REPORT.md'), 'V7_Research_Colab.ipynb': sha(ROOT / 'V7_Research_Colab.ipynb')})
    manifest = {'inputs': audit['inputs'], 'runtime': audit['runtime'], 'decision': decision,
                'outputs': outputs, 'preserved_v5': audit['preserved_v5'], 'checks': checks,
                'notebook_executions': notebook_results, 'portal_uploads': 0,
                'portable_and_checkpoint_checks': portable,
                'implementation': {p.name: sha(p) for p in ROOT.glob('*.py')},
                'experiment_plan_sha256': sha(ROOT / 'experiment_plan.json'),
                'candidate_lock_sha256': sha(OUT / 'candidate_lock.json')}
    save(ROOT / 'delivery_manifest.json', manifest)
    portal = ROOT / 'portal_trials.csv'
    if not portal.exists():
        rows = [{'version': 'v5', 'csv_sha256': audit['preserved_v5'][STEM + '.csv'], 'portal_rmse': 12046722.3691139,
                 'rank': '', 'submitted_at': '', 'status': 'user-confirmed score'}]
        rows += [{'version': c['trial'], 'csv_sha256': c['csv_sha256'], 'portal_rmse': '', 'rank': '',
                  'submitted_at': '', 'status': 'not submitted'} for c in checks]
        pd.DataFrame(rows).to_csv(portal, index=False)
    (ROOT / 'README.md').write_text('''# V7 restaurant turnover

Read REPORT.md before uploading a CSV. Files in candidates/trial_1 and trial_2
are distinct experimental candidates, each with its matching executed notebook.
The report states their local evidence and mathematical first-place feasibility.
Portal scores remain unknown; preserve v5 until a better score is observed.

For Colab: upload a candidate notebook, select CPU, run all, then upload the four
official CSVs. To rerun all research use V7_Research_Colab.ipynb instead. Checkpoints
resume on the same runtime; enable the notebook's optional Google Drive setting
to keep checkpoints through runtime deletion. No credentials are embedded.

Local reproduction uses the compatible Python/dependency versions in results/audit.json:

```sh
python3 improvement_v7/run_local.py --data-dir /path/to/the/four/official/csvs
```

Or, with the compatible environment already configured, from improvement_v7:

```sh
python workflow.py all --data-dir /path/to/the/four/official/csvs
python deliver.py build
python execute_notebooks.py --data-dir /path/to/the/four/official/csvs
python deliver.py finalize
```

workflow.py defaults to the workspace data directory and can train references
from scratch without older improvement folders. Preserved v5 files and its minimal
hash manifest are included. Stale caches are rejected. Initial portability-check
artifacts are archived separately and are not used in the final results.
Submit only one final executed coding notebook after choosing the best portal CSV.
''')
    (ROOT / 'SUBMISSION_INSTRUCTIONS.md').write_text('''# How to test the v7 candidates

1. Read REPORT.md for the measured evidence and first-place feasibility of each trial.
2. Upload only the CSV inside candidates/trial_1 to the restaurant-turnover Hackathon.
3. Record the returned RMSE with trial_1 and its exact CSV hash in portal_trials.csv.
4. Repeat with the separate CSV inside candidates/trial_2 if a second trial exists.
5. Keep the file with the lowest observed portal RMSE, including the preserved v5 file.
6. Submit only its matching executed notebook to LLJ_Oct_2026 after the final choice.

The trial CSVs have the same required filename in different folders. Keep the folders
separate so scores are associated with the right version. Upload individual CSVs to
the Hackathon, not the delivery ZIP or research notebook. The research notebook is
for rerunning experiments; the notebook beside each trial CSV reproduces that trial.

The original v5 remains backed up and was not replaced. No portal upload was made
by these scripts. An exploratory candidate has no known portal score until tested.
''')
    import zipfile
    bundle = ROOT / 'V7_Delivery.zip'
    include = [ROOT / name for name in ['REPORT.md', 'README.md', 'SUBMISSION_INSTRUCTIONS.md',
        'delivery_manifest.json', 'experiment_plan.json', 'portal_trials.csv',
        'V7_Research_Colab.ipynb', 'v5_contract.json', 'reference_provenance.json', 'scored_reference.json']]
    include += list(ROOT.glob('*.py'))
    for directory in ['candidates', 'results', 'scored_v5_backup']:
        include += [p for p in (ROOT / directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in include:
            archive.write(path, Path('improvement_v7') / path.relative_to(ROOT))
    print(json.dumps({'outputs': outputs, 'trials': [c['trial'] for c in checks]}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['research', 'build', 'finalize'])
    args = parser.parse_args()
    if args.stage == 'research':
        build_research()
    elif args.stage == 'build':
        decision = json.loads((OUT / 'decision.json').read_text())
        for i, choice in enumerate(decision['trials'], 1):
            build_candidate(i, choice)
        build_research()
    else:
        finalize()
