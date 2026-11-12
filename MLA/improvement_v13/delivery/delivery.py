"""Measured V13 reports, independently executed standalone notebook and archive."""
import base64
import hashlib
import json
import lzma
import os
import shutil
import textwrap
import zipfile
import zlib
from pathlib import Path
import pandas as pd
from core import ROOT, STEM, PINNED, MODEL_FILES, sha, save


def build_notebook(e, decision, target):
    import nbformat
    files = {}
    for name in MODEL_FILES + ['authorization.json', 'resolved_models.json']:
        files[name] = (ROOT / name).read_bytes()
    for name in ['scored_v7.csv', 'scored_v9.csv']:
        files['reference/' + name] = (ROOT / 'reference' / name).read_bytes()
    files['reference/expected.csv'] = (target / (STEM + '.csv')).read_bytes()
    manifest = {'choice': decision['choices'][0], 'inputs': e.inputs,
                'files': {name: hashlib.sha256(value).hexdigest() for name, value in files.items()},
                'v7_checkpoint_sha256': sha(ROOT / 'reference/verified_v7.pkl')}
    manifest['files']['reference/verified_v7.pkl'] = manifest['v7_checkpoint_sha256']
    files['standalone_manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    payload = base64.b64encode(zlib.compress(json.dumps({n: base64.b64encode(v).decode() for n, v in files.items()}).encode(), 9)).decode()
    model = base64.b64encode(lzma.compress((ROOT / 'reference/verified_v7.pkl').read_bytes(), preset=6)).decode()
    book = nbformat.v4.new_notebook(metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                              'colab': {'name': STEM + '.ipynb'}})
    def md(source):
        book.cells.append(nbformat.v4.new_markdown_cell(source))
    def code(source, hidden=False):
        cell = nbformat.v4.new_code_cell(source)
        if hidden:
            cell.metadata.update(cellView='form', source_hidden=True)
        book.cells.append(cell)
    md('# V13 qualified turnover model\n\nUse a free Tesla T4 and Run all. Upload the four official CSVs when requested. '
       'The selected target treatment and blend weight were frozen before confirmation. All validation errors use original turnover labels.\n\n'
       'The V7 share uses your own independently verified original-Mac fitted model, embedded below with its training source. '
       'Fresh Linux V7 training previously differed numerically; identical Linux retraining is not claimed. '
       'The V9 TabPFN component and selected V13 component fit afresh. Expected predictions are used only after inference for reproduction checks.\n\n'
       'TabPFN v2: Prior Labs; Hollmann et al., Nature 2025, doi:10.1038/s41586-024-08328-6. '
       'Pinned checkpoint and dependencies are recorded in the embedded resolved_models.json.')
    pins = {key: value.split('+')[0] for key, value in PINNED.items()}
    code("""import os,sys,json,base64,hashlib,lzma,zlib,tempfile,subprocess,importlib.metadata
from pathlib import Path
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LOKY_MAX_CPU_COUNT']:os.environ[key]='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
DATA_DIR=Path(os.environ.get('MLA_DATA_DIR','/content' if Path('/content').is_dir() else '.')).resolve()
OUTPUT_DIR=Path(os.environ.get('MLA_V13_OUTPUT_DIR','.')).resolve();OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
required=['Train_dataset.csv','Test_dataset.csv','Data_Dictionary.csv','Sample_Solution.csv']
if not all((DATA_DIR/n).is_file() for n in required):
    from google.colab import files
    files.upload();DATA_DIR=Path.cwd()
assert all((DATA_DIR/n).is_file() for n in required),'Upload all four official CSVs.'
PINS=""" + repr(pins) + """
needed=[]
for name,version in PINS.items():
    try:current=importlib.metadata.version(name).split('+')[0]
    except importlib.metadata.PackageNotFoundError:current=None
    if current!=version:needed.append(name+'=='+version)
if needed:subprocess.check_call([sys.executable,'-m','pip','install','-q',*needed])
WORK=Path(tempfile.mkdtemp(prefix='v13_fresh_',dir=OUTPUT_DIR))
MODEL_PARTS=[]
print('Fresh execution directory:',WORK)
""")
    md('## Embedded verified V7 state\n\nRun every collapsed cell; a separate model upload is unnecessary.')
    for start in range(0, len(model), 190000):
        code('MODEL_PARTS.append("""\n' + '\n'.join(textwrap.wrap(model[start:start + 190000], 120)) + '\n""")', True)
    code('BUNDLE=json.loads(zlib.decompress(base64.b64decode("""\n' + '\n'.join(textwrap.wrap(payload, 120)) + '\n""")))\n' + """
for name,encoded in BUNDLE.items():
    relative=Path(name);assert not relative.is_absolute() and '..' not in relative.parts
    path=WORK/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(base64.b64decode(encoded))
manifest=json.loads((WORK/'standalone_manifest.json').read_text())
model_bytes=lzma.decompress(base64.b64decode(''.join(MODEL_PARTS)))
assert hashlib.sha256(model_bytes).hexdigest()==manifest['v7_checkpoint_sha256']
(WORK/'reference/verified_v7.pkl').write_bytes(model_bytes)
del MODEL_PARTS,model_bytes,BUNDLE
""", True)
    md('## Fresh training and checks\n\nA separate Python process fits both TabPFN components, reproduces V9, '
       'checks the complete blend and verifies the exported predictions within INR 0.01.')
    code("""environment=os.environ.copy()
environment.update(MLA_DATA_DIR=str(DATA_DIR),MLA_V13_OUTPUT_DIR=str(OUTPUT_DIR))
process=subprocess.Popen([sys.executable,'-u',str(WORK/'fresh_worker.py')],env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,cwd=WORK)
for line in process.stdout:print(line,end='',flush=True)
assert process.wait()==0,'Fresh verification failed; preserve scored V9.'
checks=json.loads((OUTPUT_DIR/'fresh_execution_checks.json').read_text())
assert checks['status']=='PASS' and checks['maximum_final_difference_INR']<.01
print('Verified CSV hash:',checks['csv_sha256'])
""")
    md('## Manual portal trial\n\nUse the verified CSV for one manual trial after checking the deadline and timezone. '
       'Keep the exact submission with the lowest observed portal RMSE, and its matching notebook.')
    code('csv_path=OUTPUT_DIR/' + repr(STEM + '.csv') + """
assert csv_path.is_file();print('CSV:',csv_path)
if os.environ.get('MLA_AUTOMATED_RUN')!='1':
    try:from google.colab import files
    except ImportError:pass
    else:files.download(str(csv_path))
""")
    for cell in book.cells:
        if cell.cell_type == 'code':
            compile(cell.source, '<V13 notebook>', 'exec')
    nbformat.validate(book)
    path = target / (STEM + '.ipynb')
    nbformat.write(book, path)
    return book, path


def build_and_execute(e, decision, target):
    import nbformat
    from nbclient import NotebookClient
    book, path = build_notebook(e, decision, target)
    fresh = target / 'fresh_execution'
    fresh.mkdir(exist_ok=True)
    keys = ['MLA_DATA_DIR', 'MLA_V13_OUTPUT_DIR', 'MLA_AUTOMATED_RUN']
    previous = {key: os.environ.get(key) for key in keys}
    os.environ.update(MLA_DATA_DIR=str(e.data), MLA_V13_OUTPUT_DIR=str(fresh), MLA_AUTOMATED_RUN='1')
    try:
        NotebookClient(book, timeout=max(1, int(e.budget.remaining('export')) - 120), kernel_name='python3',
                       resources={'metadata': {'path': str(ROOT)}}).execute()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    nbformat.write(book, path)
    checks = json.loads((fresh / 'fresh_execution_checks.json').read_text())
    assert checks['status'] == 'PASS' and checks['maximum_final_difference_INR'] < .01
    assert all(cell.execution_count is not None for cell in book.cells if cell.cell_type == 'code')
    assert not [output for cell in book.cells if cell.cell_type == 'code'
                for output in cell.get('outputs', []) if output.get('output_type') == 'error']


def report(e, decision, target):
    pilot = json.loads((e.out / 'pilot.json').read_text())
    lines = ['# V13 measured report', '',
             '**' + ('One candidate qualified for a manual portal trial.' if decision['qualified'] else 'Retain V9. No new V13 submission is recommended.') + '**', '',
             'Best observed portal RMSE: V9, 11,904,515.9864996. Validation results are not portal scores.',
             'Three fixed treatments: Q99 and Q995 cap training labels at fold-local percentiles; SQ2 uses the square-root predictive second moment. '
             'Every training row is retained. Every validation metric uses the original uncapped labels.', '',
             '## Feasibility', '', '```json', json.dumps(pilot['settings'], indent=2), '```', '',
             'Screened settings: ' + (', '.join(pilot['screen_settings']) or 'none') + '.', '',
             '## Development', '', 'Each new model was blended with V9 at 25%, 50%, 75%, 100%.']
    if (e.out / 'screening.csv').exists():
        lines += ['', '| Setting | Weight | Gain 42, INR | Gain 71, INR | Mean relative gain |',
                  '|---|---:|---:|---:|---:|']
        for row in pd.read_csv(e.out / 'screening.csv').to_dict('records'):
            lines.append(f"| {row['name']} | {row['weight']:.0%} | {row['gain_42']:,.2f} | {row['gain_71']:,.2f} | {row['mean_relative_gain']:.5%} |")
    lines += ['', '## Confirmation and decision', '', decision['decision']]
    if not decision.get('finalists'):
        lines += ['Confirmation and bootstrap qualification were not performed because no development candidate qualified. '
                  'No new V13 CSV or executed V13 submission notebook is claimed. Original scored V9 files are preserved byte-for-byte.']
    for finalist in decision.get('finalists', []):
        lines += ['', '```json', json.dumps(finalist, indent=2), '```']
    lines += ['', 'Development requires improvement on both assignments and mean relative gain >=0.20%. '
              'One finalist is frozen before confirmation. Confirmation requires positive gains on all three random assignments, '
              'mean relative gain >=0.20%, a positive lower 95% paired-row gain bound from 5,000 resamples, '
              'and at most 1% deterioration on either city or stratified checks. '
              'Previously examined labels provide conditional robustness evidence, not an untouched test.', '',
              '## Compute and reproduction', '',
              f"Conservative task elapsed charge: {e.budget.state()['seconds'] / 3600:.4f} hours; approved maximum eight hours. Free T4 only. No portal upload was performed.",
              'V7 uses the user-owned original-Mac fitted model independently verified in V9. Fresh Linux V7 retraining is not claimed. '
              'Both required TabPFN components train afresh in a qualified standalone notebook; expected CSVs are numerical verification witnesses only.',
              'TabPFN v2 attribution: Prior Labs; Hollmann et al., Nature 2025, doi:10.1038/s41586-024-08328-6. '
              'Checkpoint revision, SHA256, packages, target thresholds, sources and fold identities accompany this report.', '',
              'Reconfirm the portal deadline and timezone before the single manual trial. Record its returned RMSE against the exact CSV hash, '
              'retain the lower observed score, and keep the one-time coding upload pending the final choice. No rank is promised.']
    (target / 'V13_Report.md').write_text('\n'.join(lines) + '\n')


def package(e, decision, target):
    target = Path(target)
    target.mkdir(exist_ok=True)
    for path in e.out.iterdir():
        if path.is_file() and path.suffix in ['.json', '.csv']:
            shutil.copy2(path, target / path.name)
    for name in ['PLAN.md', 'README.md', 'authorization.json', 'resolved_models.json',
                 'reference_manifest.json', 'baseline_registry.json', 'local_tests.json']:
        if (ROOT / name).exists():
            shutil.copy2(ROOT / name, target / name)
    source = target / 'source'
    source.mkdir(exist_ok=True)
    for path in ROOT.glob('*.py'):
        shutil.copy2(path, source / path.name)
    for name in ['v5_contract.json', 'city_map.json']:
        shutil.copy2(ROOT / name, source / name)
    report(e, decision, target)
    manifest = {'inputs': e.inputs, 'sources': e.sources, 'decision': decision,
                'observed_portal_best': {'version': 'V9', 'rmse': 11904515.9864996},
                'paid_resources': False, 'portal_uploads': 0, 'compute_budget': e.budget.state()}
    manifest['files'] = {str(path.relative_to(target)): sha(path) for path in target.rglob('*')
                         if path.is_file() and path != target / 'manifest.json'}
    save(target / 'manifest.json', manifest)
    archive = ROOT / 'V13_Delivery.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for folder in [target, ROOT / 'provenance', ROOT / 'reference', e.out / 'cache']:
            for path in folder.rglob('*'):
                if path.is_file() and '.partial' not in path.name:
                    prefix = '' if folder == target else folder.name + '/'
                    bundle.write(path, prefix + str(path.relative_to(folder)))
        for name in __import__('legacy_models').REQUIRED:
            bundle.write(e.data / name, 'official_inputs/' + name)
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        for name, expected in manifest['files'].items():
            assert hashlib.sha256(bundle.read(name)).hexdigest() == expected
    log_path = target / 'V13_Report.md'
    print('V13 delivery complete:', log_path, flush=True)
