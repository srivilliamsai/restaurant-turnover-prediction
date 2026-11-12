"""Measured V20 reports, independently executed standalone notebook and archive."""
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
    files = {name:(ROOT/name).read_bytes() for name in MODEL_FILES+['authorization.json','resolved_models.json']}
    for name in ['scored_v7.csv','scored_v9.csv','scored_v14.csv']:
        files['reference/'+name]=(ROOT/'reference'/name).read_bytes()
    files['reference/expected.csv']=(target/(STEM+'.csv')).read_bytes()
    manifest={'choice':decision['choices'][0],'inputs':e.inputs,
              'files':{name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()},
              'v7_checkpoint_sha256':sha(ROOT/'reference/verified_v7.pkl')}
    manifest['files']['reference/verified_v7.pkl']=manifest['v7_checkpoint_sha256']
    files['standalone_manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    payload=base64.b64encode(zlib.compress(json.dumps({n:base64.b64encode(v).decode() for n,v in files.items()}).encode(),9)).decode()
    model=base64.b64encode(lzma.compress((ROOT/'reference/verified_v7.pkl').read_bytes(),preset=6)).decode()
    book=nbformat.v4.new_notebook(metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'colab':{'name':STEM+'.ipynb'}})
    def md(text):book.cells.append(nbformat.v4.new_markdown_cell(text))
    def code(text,hidden=False):
        cell=nbformat.v4.new_code_cell(text)
        if hidden:cell.metadata.update(cellView='form',source_hidden=True)
        compile(text,'<V20 standalone notebook>','exec');book.cells.append(cell)
    md('# V20 qualified turnover model\n\nUse a free Tesla T4 and run all cells. Upload the four official CSV files. The route and blend weight were locked before confirmation. The V7 component retains its independently verified original-Mac fitted state; the required TabPFN controls and every selected V20 component fit afresh. Expected predictions are read only after inference for verification. Sources and checkpoint hashes are embedded. TabICLv2 and NVIDIA Kumo Tabular retain their authorship and original licenses as applicable to the selected route.')
    code("""import os,sys,json,base64,hashlib,lzma,zlib,tempfile,subprocess
from pathlib import Path
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LOKY_MAX_CPU_COUNT']:os.environ[key]='2'
os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
DATA_DIR=Path(os.environ.get('MLA_DATA_DIR','/content' if Path('/content').exists() else '.')).resolve()
OUTPUT_DIR=Path(os.environ.get('MLA_V20_OUTPUT_DIR','.')).resolve();OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
required=['Train_dataset.csv','Test_dataset.csv','Data_Dictionary.csv','Sample_Solution.csv']
if not all((DATA_DIR/n).is_file() for n in required):
    from google.colab import files
    files.upload();DATA_DIR=Path.cwd()
assert all((DATA_DIR/n).is_file() for n in required)
WORK=Path(tempfile.mkdtemp(prefix='v20_fresh_',dir=OUTPUT_DIR))
MODEL_PARTS=[]
print('Fresh execution directory:',WORK)
""")
    md('## Embedded verified original-Mac state\n\nRun every collapsed cell. A separate fitted-model upload is unnecessary.')
    for start in range(0,len(model),190000):
        code('MODEL_PARTS.append("""\n'+'\n'.join(textwrap.wrap(model[start:start+190000],120))+'\n""")',True)
    code('BUNDLE=json.loads(zlib.decompress(base64.b64decode("""\n'+'\n'.join(textwrap.wrap(payload,120))+'\n""")))\n'+"""
for name,encoded in BUNDLE.items():
    relative=Path(name);assert not relative.is_absolute() and '..' not in relative.parts
    path=WORK/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(base64.b64decode(encoded))
manifest=json.loads((WORK/'standalone_manifest.json').read_text())
raw=lzma.decompress(base64.b64decode(''.join(MODEL_PARTS)))
assert hashlib.sha256(raw).hexdigest()==manifest['v7_checkpoint_sha256']
(WORK/'reference/verified_v7.pkl').write_bytes(raw)
del MODEL_PARTS,raw,BUNDLE
""",True)
    md('## Pinned environments and fresh fitting\n\nThe baseline and candidate libraries use separate environments. New candidates use the declared precision-aware numerical agreement rule; V14 controls retain their stricter reproduction check.')
    code("""environment=os.environ.copy()
environment.update(MLA_V20_SETUP_ROOT=str(WORK),MLA_DATA_DIR=str(DATA_DIR),MLA_V20_OUTPUT_DIR=str(OUTPUT_DIR),MLA_V20_CANDIDATE_PYTHON=str(WORK/'tabicl_env/bin/python'))
setup=subprocess.run([sys.executable,'-u',str(WORK/'setup_environment.py')],env=environment,cwd=WORK)
assert setup.returncode==0,'Pinned environment setup failed.'
process=subprocess.Popen([sys.executable,'-u',str(WORK/'fresh_worker.py')],env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,cwd=WORK)
for line in process.stdout:print(line,end='',flush=True)
assert process.wait()==0,'Fresh fitting or verification failed; retain V14.'
checks=json.loads((OUTPUT_DIR/'fresh_execution_checks.json').read_text())
assert checks['status']=='PASS' and checks['prediction_agreement_passed']
print('Verified CSV hash:',checks['csv_sha256'])
""")
    md('## Manual portal trial\n\nUse the verified CSV for at most one manual trial. Retain the file and matching notebook with the lowest observed portal RMSE.')
    code('csv_path=OUTPUT_DIR/'+repr(STEM+'.csv')+"""
assert csv_path.is_file();print('CSV:',csv_path)
if os.environ.get('MLA_AUTOMATED_RUN')!='1':
    try:from google.colab import files
    except ImportError:pass
    else:files.download(str(csv_path))
""")
    nbformat.validate(book);path=target/(STEM+'.ipynb');nbformat.write(book,path)
    return book,path


def build_and_execute(e, decision, target):
    import nbformat
    from nbclient import NotebookClient
    book, path = build_notebook(e, decision, target)
    fresh = target / 'fresh_execution'
    fresh.mkdir(exist_ok=True)
    keys = ['MLA_DATA_DIR', 'MLA_V20_OUTPUT_DIR', 'MLA_AUTOMATED_RUN']
    previous = {key: os.environ.get(key) for key in keys}
    os.environ.update(MLA_DATA_DIR=str(e.data), MLA_V20_OUTPUT_DIR=str(fresh), MLA_AUTOMATED_RUN='1')
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
    assert checks['status'] == 'PASS' and checks['prediction_agreement_passed']
    assert all(cell.execution_count is not None for cell in book.cells if cell.cell_type == 'code')
    assert not [output for cell in book.cells if cell.cell_type == 'code'
                for output in cell.get('outputs', []) if output.get('output_type') == 'error']


def report(e, decision, target):
    pilot = json.loads((e.out / 'pilot.json').read_text())
    lines = ['# V20 measured report', '',
        '**' + ('One V20 candidate qualified for a manual portal trial.' if decision['qualified'] else 'Retain V14: no new V20 submission is recommended.') + '**', '',
        'Best observed portal RMSE: V14, 11,736,110.7877709. V20 has no portal score. Local validation is not a portal result.', '',
        '## Fixed experiment', '',
        'I8/I16: eight/sixteen-member TabICLv2. KS8-CLEAN: eight-member Kumo Small in an isolated environment without cuDF/Numba, with a fixed quantile-grid mean. The equal mean uses I8 and KS8-CLEAN when both pass engineering. Engineering admission is frozen before development scores.',
        'Formula: (1-w)*V14+w*M, w in [0.10,0.25,0.50,0.75,1.00]. All 3,493 original labels remain in evaluation.', '',
        '## Engineering', '', '```json', json.dumps(pilot, indent=2), '```', '',
        '## Development', '',
        '| Setting | Weight | RMSE 42 | RMSE 71 | Gain 42 | Gain 71 | Mean relative gain |',
        '|---|---:|---:|---:|---:|---:|---:|']
    path = e.out / 'screening.csv'
    if path.exists():
        for row in pd.read_csv(path).to_dict('records'):
            lines.append(f"| {row['name']} | {row['weight']:.0%} | {row['rmse_42']:,.2f} | {row['rmse_71']:,.2f} | {row['gain_42']:,.2f} | {row['gain_71']:,.2f} | {row['mean_relative_gain']:.5%} |")
    lines += ['', '## Decision and validation', '', decision['decision']]
    if not decision.get('finalists'):
        lines += ['Confirmation, bootstrap qualification, a new CSV and a fresh V20 submission notebook were skipped because no development candidate qualified. Byte-identical scored V14 files are preserved under KEEP_V14 names. Engineering exclusions are not accuracy failures.']
    for finalist in decision.get('finalists', []):
        lines += ['', '```json', json.dumps(finalist, indent=2), '```']
    lines += ['', 'Require positive gains on both development assignments and mean gain >=0.20% versus V14. Freeze one finalist before confirmation. Require all three random confirmation gains positive, mean relative gain >=0.20%, positive lower paired-row 95% gain bound (5,000 draws), and city/stratified deterioration <=1%. These are historically examined labels and conditional robustness checks.', '',
        '## Reproduction and compute', '',
        f"Elapsed charge at packaging: {e.budget.state()['seconds']/3600:.4f} hours under the six-hour cap, including setup and interruptions. Free T4 only; no portal upload.",
        'V14 control reproduction must remain below INR 0.01. New candidate fresh/reload/query-layout/CSV agreement uses atol=10 INR and rtol=1e-6, plus at most INR 10 labeled-pilot RMSE drift. No output rounding is used.',
        'The V7 portion retains the verified original-Mac fitted-state boundary. Qualified notebooks freshly fit the required TabPFN controls and new component. Expected predictions are verification witnesses after inference.',
        'At most one manual portal trial is recommended after every gate passes. Keep the lowest observed submission and leave the one-time coding submission pending final model choice.', '',
        'Sources: official soda-inria TabICL, NVIDIA structured-data-models and Prior Labs TabPFN repositories. Exact sources, versions, hashes, failures and controls accompany the delivery.']
    (target / 'V20_Report.md').write_text('\n'.join(lines) + '\n')


def package(e, decision, target):
    target = Path(target)
    target.mkdir(exist_ok=True)
    for path in e.out.iterdir():
        if path.is_file() and path.suffix in ['.json', '.csv']:
            shutil.copy2(path, target / path.name)
    for name in ['PLAN.md', 'README.md', 'authorization.json', 'resolved_models.json',
                 'reference_manifest.json', 'baseline_registry.json', 'local_tests.json', 'deadline_check.json', 'RESEARCH_SOURCES.md', 'evidence_audit.json', 'planning_manifest.json', 'candidate_dependency_locks.json', 'engineering_setup_history.json','source_registry.json','checkpoint_registry.json','setup_status.json','runtime_lock.json','protocol_proposed.json']:
        if (ROOT / name).exists():
            shutil.copy2(ROOT / name, target / name)
    if (ROOT/'setup_conflict_evidence').exists():
        shutil.copytree(ROOT/'setup_conflict_evidence',target/'setup_conflict_evidence',dirs_exist_ok=True)
    source = target / 'source'
    source.mkdir(exist_ok=True)
    for path in ROOT.glob('*.py'):
        shutil.copy2(path, source / path.name)
    for name in ['v5_contract.json', 'city_map.json']:
        shutil.copy2(ROOT / name, source / name)
    report(e, decision, target)
    manifest = {'inputs': e.inputs, 'sources': e.sources, 'decision': decision,
                'observed_portal_best': {'version': 'V14', 'rmse': 11736110.7877709},
                'paid_resources': False, 'portal_uploads': 0, 'compute_budget': e.budget.state()}
    manifest['files'] = {str(path.relative_to(target)): sha(path) for path in target.rglob('*')
                         if path.is_file() and path != target / 'manifest.json' and not any(part in path.parts for part in ['model_weights', '__pycache__']) and path.suffix != '.safetensors'}
    save(target / 'manifest.json', manifest)
    archive = ROOT / 'V20_Delivery.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for folder in [target, ROOT / 'provenance', ROOT / 'reference', ROOT / 'engineering_sources', e.out / 'cache']:
            for path in folder.rglob('*'):
                if path.is_file() and '.partial' not in path.name and not any(part in path.parts for part in ['model_weights', '__pycache__']) and path.suffix != '.safetensors':
                    prefix = '' if folder == target else folder.name + '/'
                    bundle.write(path, prefix + str(path.relative_to(folder)))
        for name in __import__('legacy_models').REQUIRED:
            bundle.write(e.data / name, 'official_inputs/' + name)
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        for name, expected in manifest['files'].items():
            assert hashlib.sha256(bundle.read(name)).hexdigest() == expected
    log_path = target / 'V20_Report.md'
    print('V20 delivery complete:', log_path, flush=True)

