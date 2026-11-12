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
    for name in ['city_map.json','reference_manifest.json','v5_contract.json','source_inventory.json']:
        files[name]=(ROOT/name).read_bytes()
    for name in ['scored_v7.csv','scored_v9.csv','v7_original_platform_reproduction.json']:
        files['reference/'+name]=(ROOT/'reference'/name).read_bytes()
    expected=target/(STEM+'.csv') if decision['qualified'] else ROOT/'reference/scored_v9.csv'
    files['reference/expected.csv']=expected.read_bytes()
    manifest={'inputs':e.inputs,'choice':decision['choice'] if decision['qualified'] else None,
        'files':{name:hashlib.sha256(value).hexdigest() for name,value in files.items()},
        'v7_checkpoint_sha256':e.manifest['v7_checkpoint_sha256'],
        'reproduction_boundary':'V7 uses independently verified original-Mac fitted state; each TabPFN component trains afresh.',
        'execution_contract':e.manifest['execution_contract'],'expected_predictions_used_only_for_verification':True}
    files['standalone_manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    packed=base64.b64encode(zlib.compress(json.dumps({n:base64.b64encode(v).decode() for n,v in files.items()}).encode(),9)).decode()
    model=base64.b64encode(lzma.compress((ROOT/'reference/verified_v7.pkl').read_bytes(),preset=6)).decode()
    book=nbformat.v4.new_notebook(metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'colab':{'name':'V10_Selected_Colab.ipynb' if decision['qualified'] else 'V9_Verification_Colab.ipynb'}})
    md=lambda s:book.cells.append(nbformat.v4.new_markdown_cell(s))
    def code(s,hidden=False):
        cell=nbformat.v4.new_code_cell(s)
        if hidden:cell.metadata.update(cellView='form',source_hidden=True)
        book.cells.append(cell)
    md('# '+('V10 qualified model' if decision['qualified'] else 'V9 retained: fresh reproduction')+'\n\n'
       'Choose a free T4 GPU, upload the four official CSVs when asked, then use Run all. '
       'The model uses every original training target. Restaurant IDs are used only to label the output.\n\n'
       'V7 is loaded from your own independently verified original-Mac fitted model. All V7 training source is included. '
       'Fresh V7 training on Linux previously failed the reproduction tolerance; this notebook does not claim otherwise. '
       'TabPFN trains afresh here. Expected CSV values are read only after predictions to check reproduction.\n\n'
       'TabPFN attribution: Prior Labs, Hollmann et al., Nature 2025, doi:10.1038/s41586-024-08328-6. '
       'The official general-purpose v2 regression checkpoint and execution dependencies are pinned.')
    pins={k:v.split('+')[0] for k,v in PINNED.items()}
    code("""import os,sys,json,base64,hashlib,lzma,zlib,tempfile,subprocess,importlib.metadata
from pathlib import Path
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LOKY_MAX_CPU_COUNT']:os.environ[key]='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
DATA_DIR=Path(os.environ.get('MLA_DATA_DIR','/content' if Path('/content').is_dir() else '.')).resolve()
OUTPUT_DIR=Path(os.environ.get('MLA_V10_OUTPUT_DIR','.')).resolve();OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
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
WORK=Path(tempfile.mkdtemp(prefix='v10_fresh_',dir=OUTPUT_DIR))
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
environment['MLA_V10_OUTPUT_DIR']=str(OUTPUT_DIR)
process=subprocess.Popen([sys.executable,'-u',str(WORK/'fresh_worker.py')],env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,cwd=WORK)
for line in process.stdout:print(line,end='',flush=True)
assert process.wait()==0,'Execution failed; preserve the existing scored V9.'
checks=json.loads((OUTPUT_DIR/'fresh_execution_checks.json').read_text())
assert checks['status']=='PASS' and checks['maximum_final_difference_INR']<.01
print('Verified CSV hash:',checks['csv_sha256'])
""")
    md('## CSV and final choice\n\n'+('Use the qualified V10 CSV for one portal trial and retain whichever exact file has the lowest observed RMSE.'
       if decision['qualified'] else 'No V10 blend qualified. Retain the original scored V9 CSV; this notebook separately verifies its predictions.')+
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
        if cell.cell_type=='code':compile(cell.source,'<standalone V10 notebook>','exec')
    nbformat.validate(book)
    path=target/(STEM+'.ipynb' if decision['qualified'] else 'V9_Verification_Colab.ipynb')
    nbformat.write(book,path)
    return book,path,manifest


def write_report(e,decision,target):
    import pandas as pd
    table=pd.read_csv(e.out/'screening.csv')
    lines=['# V10 measured report','',
        '**'+('One V10 candidate passed all declared gates.' if decision['qualified'] else 'Retain scored V9. No qualified V10 submission is recommended.')+'**','',
        'Current observed portal best: V9, RMSE 11,904,515.9864996; rank 9 in the supplied screenshot. No V10 portal score has been observed.','',
        'All 3,493 original targets were retained. Six fixed TabPFN feature settings and 24 fixed replacement blends were evaluated. '
        'V7 retains a 50% share; replacement weights alter only the TabPFN half. No target filters, external labels, fitted calibrator or prediction rounding were used.','',
        'For P settings, input-removal invariance is verified for the new TabPFN component. The frozen V7 and original T9 shares retain their original features; the full blend is not claimed to ignore Food Rating/Order Wait Time globally.','',
        'A numerical preflight found INR 1.295133 batch-size dependence in native float32 inference. '
        'The required batch-invariance check therefore uses fixed 500-row internal batches with a frozen training-feature padding row; padded outputs are discarded. '
        'All V9 TabPFN validation controls are refitted on the same folds with that adapter, and original OOF witnesses remain preserved. '
        'The full 500-row submission uses its original shape and must still reproduce the scored V9 within INR 0.01.','',
        '## Development: best blend for each setting','',
        '| Setting | Replacement weight | Gain seed 42 | Gain seed 71 |','|---|---:|---:|---:|']
    for name,g in table.groupby('name',sort=False):
        r=g.sort_values(['mean_rmse','replacement_weight','feature_count','name']).iloc[0]
        lines.append(f"| {name} | {r.replacement_weight:.0%} | {r.gain_42:,.2f} | {r.gain_71:,.2f} |")
    lines += ['','Every blend, including failures, is retained in screening.csv. Component-alone, fold, segment and row-concentration diagnostics accompany the report.']
    if decision['choice']:
        choice=decision['choice'];lines+=['','## Frozen finalist','',f"{choice['name']}; replacement weight {choice['replacement_weight']:.0%}.",'',
            '| Check | V9 RMSE | V10 RMSE | Gain |','|---|---:|---:|---:|']
        for scheme,m in decision['metrics'].items():lines.append(f"| {scheme} | {m['v9_rmse']:,.2f} | {m['candidate_rmse']:,.2f} | {m['gain']:,.2f} |")
        ci=decision['uncertainty']['ci95_INR'];lines+=['',f'Conditional 5,000-resample paired-row mean-gain interval: [{ci[0]:,.2f}, {ci[1]:,.2f}] INR. '
            'It excludes selection, retraining and hidden-test distribution uncertainty. Confirmation labels have been examined historically.']
    else:lines+=['','No blend improved V9 on both development assignments, so confirmation fitting was not required.']
    lines+=['','## Reproduction and provenance','',
        'The frozen V7 state was independently freshly fitted and verified on its original Mac platform. '
        'Fresh Linux V7 fitting is not claimed to reproduce it. The delivery freshly fits all required TabPFN components in a separate standalone notebook execution. '
        'Expected predictions are verification witnesses only; they are not features or inference lookups.','',
        'Fresh execution must reproduce the exported predictions within INR 0.01. Exact dependency, source, input, checkpoint and output hashes are recorded. '
        'Only free T4 compute was requested. Preserve the original V9 files and keep the one-time coding upload pending the final choice.','',
        'The Gemini source inventory records every inspected export without executing those scripts. Their large ensemble was not adopted: '
        'its globally precomputed target encodings compromise validation, its weights optimize a restricted target range, and some drafts are incomplete.','',
        'TabPFN attribution: Prior Labs; Hollmann et al., Nature 2025, doi:10.1038/s41586-024-08328-6. '
        'No first-place score is promised. Deadline timezone remains to be verified before a portal upload.']
    ledger_path=e.out/'compute_budget.json'
    if ledger_path.exists():
        ledger=json.loads(ledger_path.read_text())
        interruption=ledger.get('by_stage',{}).get('unobserved_runtime_interruption',0)
        lines+=['','## Compute accounting','',
            f"Recorded global charge at report creation: {(ledger.get('prior_charged_seconds',0)+ledger['seconds'])/3600:.3f} hours under the authorized {ledger.get('global_cap_hours',18)}-hour total cap. "
            f"This includes {interruption/3600:.3f} hours charged conservatively for an unobserved disconnected interval; actual computation during that interval is unknown. "
            'Two detached free-runtime attempts ended without a completed screen fold. Interrupted pilot time is retained. '
            'Subsequent actual training uses foreground notebook execution with supervised limits. The final compute_budget.json is authoritative.']
    (target/'V10_Report.md').write_text('\n'.join(lines)+'\n')


def build_and_execute(e,decision,target):
    from nbclient import NotebookClient
    book,path,manifest=build_notebook(e,decision,target)
    fresh=target/'fresh_execution';fresh.mkdir(exist_ok=True)
    prior={k:os.environ.get(k) for k in ['MLA_DATA_DIR','MLA_V10_OUTPUT_DIR','MLA_AUTOMATED_RUN']}
    os.environ.update(MLA_DATA_DIR=str(e.data),MLA_V10_OUTPUT_DIR=str(fresh),MLA_AUTOMATED_RUN='1')
    try:
        client=NotebookClient(book,timeout=10800,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}})
        client.execute()
    finally:
        for k,v in prior.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v
    nbformat.write(book,path)
    checks=json.loads((fresh/'fresh_execution_checks.json').read_text())
    assert checks['status']=='PASS' and checks['maximum_final_difference_INR']<.01
    assert all(c.execution_count is not None for c in book.cells if c.cell_type=='code')
    for name in ['screening.csv','component_diagnostics.csv','development_segments.csv','development_folds.csv',
                 'development_concentration.json','confirmation.csv','confirmation_segments.csv','confirmation_folds.csv',
                 'confirmation_concentration.json','candidate_lock.json','decision.json','pilot.json','audit.json',
                 'experiment_plan.json','feature_verification.json','compute_budget.json','baseline_inference_differences.json',
                 'runtime_interruption.json','second_runtime_interruption.json','third_runtime_interruption.json',
                 'budget_amendment.json','resumed_runtime_checks.json','pruned_model_verification.json']:
        p=e.out/name
        if p.exists():shutil.copy2(p,target/name)
    shutil.copy2(ROOT/'source_inventory.json',target/'source_inventory.json')
    for name in ['adapter_verification.json','local_feature_verification.json','notebook_structure_verification.json','independent_review.json']:
        if (ROOT/name).exists():shutil.copy2(ROOT/name,target/name)
    source=target/'source';source.mkdir(exist_ok=True)
    for p in ROOT.glob('*.py'):
        if p.name not in ['initialize.py','colab_control.py']:shutil.copy2(p,source/p.name)
    for name in ['city_map.json','v5_contract.json','reference_manifest.json']:shutil.copy2(ROOT/name,source/name)
    for name in ['PLAN.md','README.md']:
        if (ROOT/name).exists():shutil.copy2(ROOT/name,target/name)
    write_report(e,decision,target)
    manifest.update(decision=decision,standalone_fresh_checks=checks,portal_uploads=0,paid_resources_requested=False,
                    observed_portal_best={'version':'V9','rmse':11904515.9864996},
                    files={str(p.relative_to(target)):sha(p) for p in target.rglob('*') if p.is_file() and p.name!='manifest.json'})
    save(target/'manifest.json',manifest)
    archive=ROOT/'V10_Delivery.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in target.rglob('*'):
            if p.is_file():z.write(p,str(p.relative_to(target)))
        for name in ['Train_dataset.csv','Test_dataset.csv','Data_Dictionary.csv','Sample_Solution.csv']:
            z.write(e.data/name,'official_inputs/'+name)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print('Fresh notebook verification PASS; delivery archive:',archive,flush=True)


def finalize_receipt(e,target):
    """Refresh hashes and the budget after the outer export charge completes."""
    target=Path(target)
    manifest=json.loads((target/'manifest.json').read_text())
    for name in ['compute_budget.json','pruned_model_verification.json']:
        if (e.out/name).exists():shutil.copy2(e.out/name,target/name)
    if (e.out/'budget_history').exists():
        shutil.copytree(e.out/'budget_history',target/'budget_history',dirs_exist_ok=True)
    for name in ['adapter_verification.json','local_feature_verification.json','notebook_structure_verification.json','independent_review.json']:
        if (ROOT/name).exists():shutil.copy2(ROOT/name,target/name)
    for folder in ['interrupted_pilot_diagnostics','colab_results_before_batch_fix']:
        original=ROOT/folder
        if original.exists():
            for p in original.rglob('*'):
                if p.is_file() and p.suffix in ['.json','.log']:
                    dest=target/'preflight_evidence'/folder/p.relative_to(original)
                    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    manifest['files']={str(p.relative_to(target)):sha(p) for p in target.rglob('*') if p.is_file() and p!=target/'manifest.json'}
    manifest['final_compute_accounting']=json.loads((e.out/'compute_budget.json').read_text())
    accounting=manifest['final_compute_accounting']
    total=accounting.get('prior_charged_seconds',0)+accounting['seconds']
    cap=accounting.get('global_cap_hours',18)
    assert total<=cap*3600,'V10 total authorized budget exceeded.'
    manifest['global_budget']={'authorized_hours':cap,'charged_hours':total/3600,'prior_charges_preserved':True}
    save(target/'manifest.json',manifest)
    archive=ROOT/'V10_Delivery.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in target.rglob('*'):
            if p.is_file():z.write(p,str(p.relative_to(target)))
        for name in ['Train_dataset.csv','Test_dataset.csv','Data_Dictionary.csv','Sample_Solution.csv']:
            z.write(e.data/name,'official_inputs/'+name)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print('Final delivery receipt and complete compute ledger verified.',flush=True)
