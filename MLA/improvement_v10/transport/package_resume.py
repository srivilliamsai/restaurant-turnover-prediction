"""Package verified progress without credentials or a new submission claim."""
import ast
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
import nbformat

root=Path(__file__).resolve().parents[1]
target=root/'recovery';target.mkdir(exist_ok=True)
files={}
for p in root.glob('*.py'):
    if p.name not in ['initialize.py','colab_control.py']:files[p.name]=p
for name in ['PLAN.md','README.md','source_inventory.json','city_map.json','reference_manifest.json','v5_contract.json',
             'adapter_verification.json','local_feature_verification.json','notebook_structure_verification.json']:
    if (root/name).exists():files[name]=root/name
for p in (root/'reference').rglob('*'):
    if p.is_file():files[str(p.relative_to(root))]=p
for p in (root/'colab_results/results').rglob('*'):
    if p.is_file() and '.partial' not in p.name:files['results/'+str(p.relative_to(root/'colab_results/results'))]=p
for name in ['Train_dataset.csv','Test_dataset.csv','Data_Dictionary.csv','Sample_Solution.csv']:
    files['data/'+name]=root.parent/name
fold_pairs=len(list((root/'colab_results/results/cache').glob('*.npz')))
progress_path=root/'colab_results/results/screening_progress.json'
progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'settings_complete':0}
ordered=['C','S','C+S','C+S+G','P','C+S+P'];complete=ordered[:progress['settings_complete']]
manifest={'status':'verified_partial_research_not_a_qualified_submission','fold_pairs':fold_pairs,
          'observed_portal_best':{'version':'V9','rmse':11904515.9864996},
          'complete_settings':complete,'incomplete_settings':[s for s in ordered if s not in complete],
          'files':{name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in files.items()},
          'credentials_included':False,'portal_uploads':0}
(target/'resume_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=target/'V10_Resume_Bundle.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for name,p in files.items():z.write(p,name)
    z.write(target/'resume_manifest.json','resume_manifest.json')
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
archive_sha=hashlib.sha256(archive.read_bytes()).hexdigest()
pins=next(ast.literal_eval(node.value) for node in ast.parse((root/'core.py').read_text()).body
          if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PINNED' for t in node.targets))
book=nbformat.v4.new_notebook(metadata={'kernelspec':{'name':'python3','language':'python','display_name':'Python 3'},
                                     'colab':{'name':'V10_Resume_Colab.ipynb'}})
book.cells.append(nbformat.v4.new_markdown_cell('''# Resume the fixed V10 research workflow

This research notebook is not a qualified portal submission. V9 remains the observed best.
Choose a free T4, run the cells in order, approve Google Drive mounting, and upload
`V10_Resume_Bundle.zip` when asked. Drive stores every completed fold directly so it
survives local app/browser interruptions. No paid resources are assumed.

The bundle preserves 26 verified fold pairs: ten V9 controls, ten C folds, and six S folds.
The six representations and four blend weights stay fixed. No V10 candidate may be
exported before the full screen and all five confirmation assignments pass.

The recorded budget amendment permits a 24-hour total cap. Prior charges are preserved;
the remaining epoch allowance is read from its ledger and may not be reset.
The code stops at its budget limits. The revised-city V7 comparison must still be fitted
on the original Mac after finalist selection; it cannot be replaced by a Linux refit.'''))
book.cells.append(nbformat.v4.new_code_cell("""import os,sys,json,hashlib,zipfile,subprocess,importlib.metadata
from pathlib import Path
from google.colab import drive,files
drive.mount('/content/drive')
BASE=Path('/content/drive/MyDrive/MLA_V10_Verified_Research')
BASE.mkdir(parents=True,exist_ok=True)
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LOKY_MAX_CPU_COUNT']:os.environ[key]='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
bundle=Path('/content/V10_Resume_Bundle.zip')
if not bundle.exists():files.upload()
assert bundle.exists(),'Upload V10_Resume_Bundle.zip.'
assert hashlib.sha256(bundle.read_bytes()).hexdigest()=="""+repr(archive_sha)+""",'Wrong or corrupt resume bundle.'
with zipfile.ZipFile(bundle) as z:
    manifest=json.loads(z.read('resume_manifest.json'))
    for name,expected in manifest['files'].items():
        assert hashlib.sha256(z.read(name)).hexdigest()==expected,name
    existing_results=(BASE/'results').exists()
    for name in z.namelist():
        relative=Path(name);assert not relative.is_absolute() and '..' not in relative.parts
        if existing_results and relative.parts[0]=='results':continue
        path=BASE/relative;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():assert path.read_bytes()==z.read(name),'Use a fresh Drive folder for changed source: '+name
        else:path.write_bytes(z.read(name))
print('Verified frozen sources and preserved existing Drive checkpoints:',BASE)
"""))
book.cells.append(nbformat.v4.new_code_cell('PINS='+repr(pins)+"""
required=[name+'=='+version.split('+')[0] for name,version in PINS.items()]
subprocess.check_call([sys.executable,'-m','pip','install','-q',*required,'nbformat','nbclient'])
print('Pinned packages installed; stages run in fresh child Python processes.')
"""))
book.cells.append(nbformat.v4.new_code_cell("""environment=os.environ.copy()
environment['PYTHONDONTWRITEBYTECODE']='1'
def stage(name):
    allowance=json.loads((BASE/'results/compute_budget.json').read_text())['maximum_hours']
    argv=[sys.executable,'-u',str(BASE/'workflow.py'),name,'--data-dir',str(BASE/'data'),'--output-dir',str(BASE/'results'),'--budget-hours',str(allowance)]
    process=subprocess.Popen(argv,env=environment,cwd=BASE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    for line in process.stdout:print(line,end='',flush=True)
    assert process.wait()==0,'Stage failed or budget exhausted. Preserve V9 and inspect results/stage_status.json.'
# The passing, source-verified pilot and completed fold pairs are already in the bundle.
assert json.loads((BASE/'results/pilot.json').read_text())['status']=='PASS'
stage('screen')
lock=json.loads((BASE/'results/candidate_lock.json').read_text())
print('Frozen finalist:',lock['choice'])
stage('confirm')
decision=BASE/'results/decision.json'
if decision.exists():
    stage('export')
else:
    print('City baseline pending. Do not upload a V10 CSV. Preserve V9.')
    print('Provide candidate_lock.json to your original Mac workflow, fit refit_city.py, and place its verified NPZ/JSON in this results folder before rerunning confirm.')
"""))
book.cells.append(nbformat.v4.new_markdown_cell('''If the runtime stops, reconnect a free T4, mount the same Drive folder and rerun.
Existing result files are preserved and the workflow validates their source, fold,
configuration and prediction hashes. Never reset the compute ledger to bypass its limit.
Keep the one-time coding upload pending the final model choice.'''))
for cell in book.cells:
    if cell.cell_type=='code':compile(cell.source,'<V10 recovery notebook>','exec')
nbformat.validate(book);nbformat.write(book,target/'V10_Resume_Colab.ipynb')
report='''# V10 recovery report

Observed best remains V9: portal RMSE **11,904,515.9864996**, rank 9 in the supplied screenshot.
No V10 portal upload has occurred.

The free Colab runtime has ended. **26 complete fold pairs were independently verified**:
ten V9 controls, ten C folds, six S folds. No results are inferred for unbacked-up work.

The completed C representation improves local full-target RMSE on both development assignments.
Its best declared replacement weight is 100%: gains **INR 4,925.0581** (42) and
**INR 3,600.4814** (71). Mean local RMSE is **19,727,868.1346**. These are development
results, not a confirmed finalist or a portal forecast. Five representations remain incomplete.

The corrected GPU pilot passed with zero measured repeat, separate-batch and individual-row
differences. Source, input, feature-map, vocabulary, cache and preserved V9 hashes were checked.
The inference adapter changes the two V9 control RMSEs by only INR 0.069916 and INR 0.063143.

All original targets remain. The source inventory, failed/interrupted diagnostics and accounting
are retained. The runtime interruptions are charged conservatively; exact unobserved computation
is unknown. The human explicitly approved a 24-hour total cap; prior charges are carried forward.

The attached research notebook and bundle resume the exact six-setting experiment and store
folds directly on Google Drive. The notebook has been structurally validated; it has not yet
completed a fresh full research execution or final CSV reproduction. No qualified V10 CSV is delivered.

Keep the scored V9 CSV and matching notebook. Do not upload the research ZIP to the portal.
'''
(target/'RECOVERY_REPORT.md').write_text(report)
print(json.dumps({'archive':str(archive),'archive_sha256':archive_sha,'notebook':str(target/'V10_Resume_Colab.ipynb'),
                  'qualified_submission':False,'verified_fold_pairs':fold_pairs,'bytes':archive.stat().st_size}))
