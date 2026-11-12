"""Build the standalone, executed V9 research delivery and measured report."""
import argparse
import base64
import json
import textwrap
import zlib
import zipfile
import nbformat as nb
import pandas as pd
from engine import ROOT, STEM, Engine, sha, save

OUT = ROOT / 'results'


def build(output_name='V9_Research_Colab.ipynb'):
    book = nb.v4.new_notebook(metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12'}, 'colab': {'name': output_name, 'provenance': []}})
    def md(s): book.cells.append(nb.v4.new_markdown_cell(s))
    def code(s): book.cells.append(nb.v4.new_code_cell(s))
    md('''# Restaurant Turnover — V9 measured research

submission · submission · Team submission

Current best: V7, user-reported portal RMSE **12,024,316.5355221**. Local CPU screening completed four EBM settings and sixteen blends; none qualified. **The four TabPFN settings have not trained yet.** Select a free T4 GPU and Run all to test all eight planned settings, apply the strict validation gates and freshly reproduce V7. This notebook does not promise a better portal score or first place.

All source is embedded. Only the four official data CSVs are required; previous project directories are not needed. The implementation uses our own feature/model code and, for TabPFN, the official general-purpose v2 pretrained checkpoint. No competitor predictions or trained competition models enter the predictor. All validation targets remain in INR scoring.''')
    code('''import os, sys, json, base64, hashlib, subprocess, importlib.metadata, tempfile
from pathlib import Path
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'LOKY_MAX_CPU_COUNT']:
    os.environ[key] = '2'
PACKAGES = {'numpy':'2.5.4', 'pandas':'3.0.6', 'scikit-learn':'1.9.1', 'scipy':'1.18.1', 'catboost':'1.2.10', 'interpret-core':'0.7.8'}
RUN_TABPFN = os.environ.get('V9_CPU_ONLY') != '1'
if RUN_TABPFN: PACKAGES['tabpfn'] = '9.2.0'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
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
if RUN_TABPFN:
    import torch
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available(): print('GPU:', torch.cuda.get_device_name(0))
else: print('CPU notebook verification: the TabPFN/GPU branch is not executed.')
''')
    md('''## Inputs and storage

Set local MLA_DATA_DIR or upload the four original CSVs when prompted. Default execution is fresh. Optional Google Drive storage requires your authorization; free Colab resources vary. Set REUSE_CHECKPOINTS=True to resume matching checkpoints.''')
    code('''USE_GOOGLE_DRIVE = False
REUSE_CHECKPOINTS = False
if USE_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    os.environ['MLA_OUTPUT_DIR'] = '/content/drive/MyDrive/MLA_v9'
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
WORK_ROOT = OUTPUT_DIR / 'v9_resumable' if REUSE_CHECKPOINTS else Path(tempfile.mkdtemp(prefix='v9_fresh_', dir=OUTPUT_DIR))
WORK_ROOT.mkdir(parents=True, exist_ok=True)
print('Research root:', WORK_ROOT)
''')
    names = ['legacy_reference.py', 'legacy_models.py', 'baseline.py', 'modeling.py', 'engine.py',
             'workflow.py', 'pilot.py', 'gpu_runner.py', 'verify_core.py', 'candidate_notebook.py', 'reference_manifest.json', 'v5_contract.json']
    paths = [ROOT / n for n in names] + sorted((ROOT / 'baseline_snapshots').glob('*.npz')) + sorted((ROOT / 'scored_v7_backup').glob('*'))
    bundle = {str(p.relative_to(ROOT)): base64.b64encode(p.read_bytes()).decode() for p in paths if p.is_file()}
    hashes = {n: sha(ROOT / n) for n in bundle}
    packed = base64.b64encode(zlib.compress(json.dumps(bundle).encode(), level=9)).decode()
    wrapped = '\n'.join(textwrap.wrap(packed, width=120))
    code('import zlib\nBUNDLE = json.loads(zlib.decompress(base64.b64decode("""\n' + wrapped + '\n""")))\nFILE_HASHES = ' + json.dumps(hashes, indent=2) + '''
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
import verify_core
verify_core.run(DATA_DIR, e.out)
print('PASS: input, implementation, snapshot and scored-file hashes; feature/cache/fold checks.')
''')
    for stage, text in [('pilot', 'Check feasibility and reproducibility at actual outer-fold size. GPU failures are recorded, never reported as training results.'),
        ('screen', 'Fit the available declared settings on five folds of assignments 42 and 71; at most eight settings and thirty-two blends.'),
        ('refine', 'Refine only settings improving on both assignments, then freeze.'),
        ('confirm', 'Confirm only frozen finalists. No finalist means no fabricated confirmation score.'),
        ('export', 'Freshly reproduce scored V7, then export qualified candidates or its exact fallback.')]:
        md('## ' + stage.capitalize() + '\n\n' + text)
        code(f'e.cache.stage = {stage!r}\nworkflow.{stage}(e)\n')
    md('''## Measured results

Only candidates improving on both development assignments advance. Confirmation requires improvement on all three random assignments and at most 1% deterioration on city and stratified checks. Positive gains mean lower RMSE than V7. These figures are recomputed from this execution. Cross-validation scores and portal scores describe different datasets.''')
    code('''table = pd.read_csv(e.out / 'screening.csv')
best = table.groupby('family', sort=False).head(1)
print(best[['name', 'weight', 'gain_42', 'gain_71', 'mean_rmse']].to_string(index=False))
decision = json.loads((e.out / 'decision.json').read_text())
print('Decision:', decision['decision'])
print('Feasibility:', (e.out / 'pilot.json').read_text())
print('GPU checkpoint:', (WORK_ROOT / 'tabpfn_checkpoint.json').read_text() if (WORK_ROOT / 'tabpfn_checkpoint.json').exists() else 'No GPU checkpoint used')
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

A qualified candidate folder contains its CSV and matching freshly executed coding notebook. The fallback is byte-identical to the already scored V7. Preserve its matching notebook. The research archive includes original sources, provenance and measured results. It is not a portal CSV. No Olympus or one-time coding upload is performed.''')
    code('''import zipfile
import candidate_notebook
candidate_notebook.build_and_verify(e, decision)
archive = OUTPUT_DIR / 'V9_research_results.zip'
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
        if cell.cell_type == 'code': compile(cell.source, '<V9 notebook>', 'exec')
    nb.validate(book)
    nb.write(book, ROOT / output_name)
    print('Built standalone research notebook.', flush=True)



if __name__ == '__main__': build()
