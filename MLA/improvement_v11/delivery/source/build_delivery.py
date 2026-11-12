"""Build and execute a portable selected-model notebook and measured report."""
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
import nbformat
from core import ROOT,STEM,PINNED,sha,save


def build_notebook(e,decision,target):
    files={}
    for p in ROOT.glob('*.py'):
        if p.name not in ['initialize.py','colab_control.py']:
            files[p.name]=p.read_bytes()
    for name in ['city_map.json','reference_manifest.json','v5_contract.json','source_inventory.json','resolved_models.json','baseline_registry.json']:
        files[name]=(ROOT/name).read_bytes()
    for name in ['scored_v7.csv','scored_v9.csv','v7_original_platform_reproduction.json']:
        files['reference/'+name]=(ROOT/'reference'/name).read_bytes()
    expected=target/(STEM+'.csv') if decision['qualified'] else ROOT/'reference/scored_v9.csv'
    files['reference/expected.csv']=expected.read_bytes()
    manifest={'inputs':e.inputs,'choice':decision['choice'] if decision['qualified'] else None,
        'files':{name:hashlib.sha256(value).hexdigest() for name,value in files.items()},
        'v7_checkpoint_sha256':e.manifest['v7_checkpoint_sha256'],
        'reproduction_boundary':'V7 uses independently verified original-Mac fitted state; each TabPFN component trains afresh.',
        'execution_contract':json.loads((ROOT/'resolved_models.json').read_text()),'expected_predictions_used_only_for_verification':True}
    files['standalone_manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    packed=base64.b64encode(zlib.compress(json.dumps({n:base64.b64encode(v).decode() for n,v in files.items()}).encode(),9)).decode()
    model=base64.b64encode(lzma.compress((ROOT/'reference/verified_v7.pkl').read_bytes(),preset=6)).decode()
    book=nbformat.v4.new_notebook(metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'colab':{'name':'V11_Selected_Colab.ipynb' if decision['qualified'] else 'V9_Verification_Colab.ipynb'}})
    md=lambda s:book.cells.append(nbformat.v4.new_markdown_cell(s))
    def code(s,hidden=False):
        cell=nbformat.v4.new_code_cell(s)
        if hidden:cell.metadata.update(cellView='form',source_hidden=True)
        book.cells.append(cell)
    md('# '+('V11 qualified model' if decision['qualified'] else 'V9 retained: fresh reproduction')+'\n\n'
       'Choose a free T4 GPU, upload the four official CSVs when asked, then use Run all. '
       'The model uses every original training target. Restaurant IDs are used only to label the output.\n\n'
       'V7 is loaded from your own independently verified original-Mac fitted model. All V7 training source is included. '
       'Fresh V7 training on Linux previously failed the reproduction tolerance; this notebook does not claim otherwise. '
       'The required TabPFN and selected TabICL/TabPFN components train afresh here. Expected CSV values are read only after predictions to check reproduction.\n\n'
       'TabPFN attribution: Prior Labs, Hollmann et al., Nature 2025, doi:10.1038/s41586-024-08328-6. '
       'Official synthetic-pretrained TabPFN/TabICL checkpoints and dependencies are pinned in resolved_models.json. TabICLv2 attribution: soda-inria/tabicl, ICML 2026.')
    pins={k:v.split('+')[0] for k,v in PINNED.items()}
    code("""import os,sys,json,base64,hashlib,lzma,zlib,tempfile,subprocess,importlib.metadata
from pathlib import Path
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LOKY_MAX_CPU_COUNT']:os.environ[key]='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
DATA_DIR=Path(os.environ.get('MLA_DATA_DIR','/content' if Path('/content').is_dir() else '.')).resolve()
OUTPUT_DIR=Path(os.environ.get('MLA_V11_OUTPUT_DIR','.')).resolve();OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
required=['Train_dataset.csv','Test_dataset.csv','Data_Dictionary.csv','Sample_Solution.csv']
if not all((DATA_DIR/n).is_file() for n in required):
    from google.colab import files
    files.upload();DATA_DIR=Path.cwd()
assert all((DATA_DIR/n).is_file() for n in required),'Upload all four official CSV files.'
PINS="""+repr(pins)+"""
needed=[]
for name,version in PINS.items():
    try:current=importlib.metadata.version(name).split('+')[0]
    except importlib.metadata.PackageNotFoundError:current=None
    if current!=version:needed.append(name+'=='+version)
if needed:subprocess.check_call([sys.executable,'-m','pip','install','-q',*needed])
WORK=Path(tempfile.mkdtemp(prefix='v11_fresh_',dir=OUTPUT_DIR))
MODEL_PARTS=[]
print('Fresh execution directory:',WORK)
""")
    md('## Embedded verified V7 model\n\nRun every collapsed cell. No separate model upload is needed.')
    for start in range(0,len(model),190000):
        code('MODEL_PARTS.append("""\n'+'\n'.join(textwrap.wrap(model[start:start+190000],120))+'\n""")',True)
    code('BUNDLE=json.loads(zlib.decompress(base64.b64decode("""\n'+'\n'.join(textwrap.wrap(packed,120))+'\n""")))\n'+"""
for name,encoded in BUNDLE.items():
    relative=Path(name);assert not relative.is_absolute() and '..' not in relative.parts
    path=WORK/relative;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(base64.b64decode(encoded))
manifest=json.loads((WORK/'standalone_manifest.json').read_text())
model_bytes=lzma.decompress(base64.b64decode(''.join(MODEL_PARTS)))
assert hashlib.sha256(model_bytes).hexdigest()==manifest['v7_checkpoint_sha256']
(WORK/'reference/verified_v7.pkl').write_bytes(model_bytes)
del MODEL_PARTS,model_bytes,BUNDLE
""",True)
    md('## Fresh training and verification\n\nThis fits the frozen TabPFN configuration, checks the preserved V9 baseline, '
       'and verifies the CSV and selected model behavior. It requires the recorded free T4 execution environment.')
    code("""environment=os.environ.copy()
environment['MLA_DATA_DIR']=str(DATA_DIR)
environment['MLA_V11_OUTPUT_DIR']=str(OUTPUT_DIR)
process=subprocess.Popen([sys.executable,'-u',str(WORK/'fresh_worker.py')],env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,cwd=WORK)
for line in process.stdout:print(line,end='',flush=True)
assert process.wait()==0,'Execution failed; preserve the existing scored V9.'
checks=json.loads((OUTPUT_DIR/'fresh_execution_checks.json').read_text())
assert checks['status']=='PASS' and checks['maximum_final_difference_INR']<.01
print('Verified CSV hash:',checks['csv_sha256'])
""")
    md('## CSV and final choice\n\n'+('Use the qualified V11 CSV for one portal trial and retain whichever exact file has the lowest observed RMSE.'
       if decision['qualified'] else 'No V11 blend qualified. Retain the original scored V9 CSV; this notebook separately verifies its predictions.')+
       ' Keep the one-time coding upload pending the final model choice. This notebook performs no portal upload.')
    output_name=STEM+'.csv' if decision['qualified'] else 'V9_Reproduced.csv'
    code('csv_path=OUTPUT_DIR/'+repr(output_name)+"""
assert csv_path.is_file()
print('CSV:',csv_path)
if os.environ.get('MLA_AUTOMATED_RUN')!='1':
    try:
        from google.colab import files
    except ImportError:pass
    else:files.download(str(csv_path))
""")
    for cell in book.cells:
        if cell.cell_type=='code':compile(cell.source,'<standalone V11 notebook>','exec')
    nbformat.validate(book)
    path=target/(STEM+'.ipynb' if decision['qualified'] else 'V9_Verification_Colab.ipynb')
    nbformat.write(book,path)
    return book,path,manifest

def build_and_execute(e,decision,target):
    from nbclient import NotebookClient
    book,path,_=build_notebook(e,decision,target);fresh=target/'fresh_execution';fresh.mkdir(exist_ok=True)
    keys=['MLA_DATA_DIR','MLA_V11_OUTPUT_DIR','MLA_AUTOMATED_RUN'];previous={k:os.environ.get(k) for k in keys}
    os.environ.update(MLA_DATA_DIR=str(e.data),MLA_V11_OUTPUT_DIR=str(fresh),MLA_AUTOMATED_RUN='1')
    try:NotebookClient(book,timeout=6900,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    finally:
        for k,v in previous.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v
    nbformat.write(book,path);checks=json.loads((fresh/'fresh_execution_checks.json').read_text())
    assert checks['status']=='PASS' and checks['maximum_final_difference_INR']<.01
    assert all(c.execution_count is not None for c in book.cells if c.cell_type=='code')
    assert not [o for c in book.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.get('output_type')=='error']
    print('Fresh V11 notebook execution and reproduction PASS.',flush=True)

def report(e,decision,target):
    pilot=json.loads((e.out/'pilot.json').read_text());ledger=e.cache.state()
    lines=['# V11 measured report','','**'+('One candidate passed the stricter local gates; one portal trial is allowed after reproduction.' if decision['qualified'] else 'Retain V9. No new V11 CSV is recommended.')+'**','',
           'Best observed portal score remains V9: 11,904,515.9864996. No V11 portal score is inferred from validation. All 3,493 original targets and original test IDs/order are retained. IDs and validation labels are excluded from predictors.',
           '', 'Features are the unchanged 40-column V9 builder. Four predeclared model settings use raw targets in INR millions; feasibility may exclude a family before development metrics are read. Blends use (1-w)*V9+w*M with w 0.25/0.50/0.75/1.00.',
           '', '## Feasibility','',json.dumps(pilot.get('families',{}),indent=2),'',
           'Engineering-only selected settings: '+', '.join(pilot['screen_settings']),
           '', '## Screening','']
    path=e.out/'screening.csv'
    if path.exists():
        import pandas as pd
        rows=pd.read_csv(path).to_dict('records');lines+=['| Setting | Weight | Gain 42 | Gain 71 | Mean relative gain |','|---|---:|---:|---:|---:|']
        for r in rows:lines.append(f"| {r['name']} | {r['weight']:.0%} | {r['gain_42']:,.2f} | {r['gain_71']:,.2f} | {r['mean_relative_gain']:.4%} |")
        if rows and not decision.get('choice'):
            best=min(rows,key=lambda r:(r['mean_rmse'],r['weight'],r['estimators'],r['name']))
            mean_gain=(best['gain_42']+best['gain_71'])/2
            lines+=['',f"The best development blend was {best['name']} at {best['weight']:.0%} weight: mean RMSE reduction INR {mean_gain:,.2f} ({best['mean_relative_gain']:.5%}). It failed the predeclared 0.20% minimum. No finalist was frozen, so confirmation, bootstrap qualification, new CSV export and fresh V11 notebook execution were correctly skipped. No V11 portal trial is recommended. The preserved V9 CSV and notebook are byte-identical to the scored originals; this round makes no new numerical reproduction claim."]
    notes=e.out/'completion_notes.json'
    if notes.exists():lines+=['','## Execution record','',json.dumps(json.loads(notes.read_text()),indent=2)]
    lines+=['','## Frozen confirmation','','Choice: '+json.dumps(decision.get('choice')),'']
    for scheme,v in decision.get('metrics',{}).items():lines.append(f"- {scheme}: V9 {v['v9_rmse']:,.2f}; candidate {v['candidate_rmse']:,.2f}; gain {v['gain']:,.2f} INR.")
    if decision.get('uncertainty'):lines+=['','Conditional paired-row bootstrap: '+json.dumps(decision['uncertainty'])]
    lines+=['','Development requires positive gains on both assignments and mean relative gain >=0.20%. Confirmation requires all three random gains positive, mean relative gain >=0.20%, a 5,000-resample paired-row lower 95% gain bound above zero, and city/stratified deterioration <=1%. Reused labels and conditional intervals do not constitute an untouched external test. All failures remain reported.',
            '', '## Reproduction boundary','','V7 uses the user-owned original-Mac fitted state verified against scored V7. Linux V7 retraining is not claimed to reproduce it. The unchanged V9 TabPFN v2 and any qualified new backbone train afresh in the standalone notebook. Expected predictions are read only after inference to test reproduction; they are not predictors.',
            '', 'Official pretrained checkpoints come from Prior Labs and the TabICLv2 maintainers (soda-inria/tabicl). Revisions, hashes, parameters and exact dependencies are recorded in resolved_models.json. No competition-trained public model or competitor predictions are used.',
            '', '## Compute and submission','',f"Recorded task charge: {ledger['seconds']/3600:.4f} hours under the separate eight-hour V11 maximum. Free compute only. Earlier V10 accounting was not reset or extended. The final compute ledger is authoritative.",
            '', 'No automatic portal upload. After every gate and fresh reproduction passes, allow one user-uploaded trial and record its exact CSV hash and actual score. Retain the lowest observed score. Keep the one-time coding upload pending. The displayed October 29, 11:59 PM deadline timezone still needs verification. No first-place result is promised.']
    (target/'V11_Report.md').write_text('\n'.join(lines)+'\n')

def package(e,decision,target):
    target=Path(target);target.mkdir(exist_ok=True,parents=True)
    for p in e.out.iterdir():
        if p.is_file() and p.suffix in ['.json','.csv']:shutil.copy2(p,target/p.name)
    for name in ['resolved_models.json','baseline_registry.json','source_inventory.json','PLAN.md','authorization.json','README.md','IMPLEMENTATION_PROGRESS.md','V9_V10_REVIEW.json','REVIEW_SOURCES.json']:
        if (ROOT/name).exists():shutil.copy2(ROOT/name,target/name)
    for folder in ['provenance']:
        if (ROOT/folder).exists():shutil.copytree(ROOT/folder,target/folder,dirs_exist_ok=True)
    source=target/'source';source.mkdir(exist_ok=True)
    for p in ROOT.glob('*.py'):
        if p.name!='colab_control.py':shutil.copy2(p,source/p.name)
    transport=source/'transport';transport.mkdir(exist_ok=True)
    for p in (ROOT/'transport').glob('*.py'):shutil.copy2(p,transport/p.name)
    for name in ['city_map.json','reference_manifest.json','v5_contract.json']:shutil.copy2(ROOT/name,source/name)
    report(e,decision,target)
    manifest={'inputs':e.inputs,'sources':e.sources,'decision':decision,'portal_uploads':0,'paid_resources_requested':False,'observed_portal_best':{'version':'V9','rmse':11904515.9864996},'compute_budget':e.cache.state()}
    manifest['files']={str(p.relative_to(target)):sha(p) for p in target.rglob('*') if p.is_file() and p!=target/'manifest.json'}
    save(target/'manifest.json',manifest);archive=ROOT/'V11_Delivery.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in target.rglob('*'):
            if p.is_file():z.write(p,str(p.relative_to(target)))
        for n in ['Train_dataset.csv','Test_dataset.csv','Data_Dictionary.csv','Sample_Solution.csv']:z.write(e.data/n,'official_inputs/'+n)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for rel,want in manifest['files'].items():assert hashlib.sha256(z.read(rel)).hexdigest()==want
    print('V11 measured delivery packaged:',archive,flush=True)


