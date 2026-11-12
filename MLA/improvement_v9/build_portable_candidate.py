"""Build a reviewable draft; release still requires corrected confirmation."""
import base64
import hashlib
import json
import lzma
import textwrap
import zlib
from pathlib import Path
import nbformat as nb
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parent
run=root/'colab_cli_runs/20261026T160148Z'
out=run/'recovery_draft';out.mkdir(exist_ok=True)
diagnostic=run/'export_diagnostics'
comparison=json.loads((diagnostic/'local_comparison.json').read_text())
portable=json.loads((diagnostic/'portable_inference.json').read_text())
full=json.loads((run/'final_component/checks.json').read_text())
assert comparison['maximum_difference_from_scored_V7_INR']<.01
assert portable['maximum_difference_from_scored_V7_INR']<.01
assert full['maximum_repeat_difference_INR']<.01
original=json.loads((run/'partial_backup/results/decision.json').read_text())
choice={k:original['trials'][0][k] for k in ['name','mode','weight','family']}
base=pd.read_csv(root/'scored_v7_backup/submission.csv')
component=np.load(run/'final_component/tabpfn_full_fit.npz',allow_pickle=False)
assert np.array_equal(component['ids'],base['Registration Number'].to_numpy())
prediction=(1-choice['weight'])*base['Annual Turnover'].to_numpy()+choice['weight']*component['prediction']
expected=base.copy();expected['Annual Turnover']=prediction
expected.to_csv(out/'expected.csv',index=False)
names=['legacy_reference.py','legacy_models.py','baseline.py','modeling.py','engine.py',
       'workflow.py','pilot.py','gpu_runner.py','reference_manifest.json','v5_contract.json','portable_candidate_worker.py']
sources={n:(root/n).read_bytes() for n in names}
sha=lambda value:hashlib.sha256(value).hexdigest()
model=(diagnostic/'local_v7_fit.pkl').read_bytes()
assert sha(model)==portable['checkpoint_sha256']
manifest={'status':'draft_until_corrected_confirmation_passes','choice':choice,
          'inputs':full['inputs'],'source_hashes':{n:sha(v) for n,v in sources.items()},
          'v7_checkpoint_sha256':sha(model),'tabpfn_execution_contract':full['execution_contract'],
          'baseline_fresh_training_platform':{k:comparison[k] for k in ['python','system','machine']},
          'baseline_fresh_reproduction_difference_INR':comparison['maximum_difference_from_scored_V7_INR'],
          'frozen_candidate_lock_sha256':original['lock_sha256']}
sources['portable_manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
sources['scored_v7.csv']=base.to_csv(index=False).encode()
sources['expected.csv']=expected.to_csv(index=False).encode()
bundle={n:base64.b64encode(v).decode() for n,v in sources.items()}
packed=base64.b64encode(zlib.compress(json.dumps(bundle).encode(),9)).decode()
model_payload=base64.b64encode(lzma.compress(model,preset=6)).decode()
book=nb.v4.new_notebook(metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'colab':{'name':'V9_Portable_Colab.ipynb'}})
md=lambda s:book.cells.append(nb.v4.new_markdown_cell(s))
def code(s,hidden=False):
 c=nb.v4.new_code_cell(s)
 if hidden:c.metadata.update({'collapsed':True,'jupyter':{'source_hidden':True}})
 book.cells.append(c)
md('''# Restaurant turnover — frozen V9 candidate

submission · submission · submission

**Draft until the accompanying corrected confirmation report passes.** The selected ensemble is 50% your own V7 model and 50% raw-target TabPFN v2 with V7 features and 16 estimators. No first-place score is promised.

Use a free T4 GPU, upload the four official CSVs when requested, then run all cells. No paid resources or portal uploads are performed.

V7 was freshly trained and reproduced on the original Mac platform. This notebook embeds that verified fitted model, checks its hash and reproduces its predictions in Colab. It trains TabPFN afresh on all 3,493 original targets. It does **not** claim to retrain V7 identically on Linux; that earlier attempt failed and is preserved in the report. The V7 training implementation is included in the embedded sources.

TabPFN attribution: Prior Labs; Hollmann et al., Nature 2025, doi:10.1038/s41586-024-08328-6. The general-purpose official v2 regression checkpoint is hash-pinned. No competitor predictions or competition-trained weights are used.''')
code('''import os,sys,json,base64,hashlib,subprocess,tempfile,importlib.metadata,lzma,zlib
from pathlib import Path
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LOKY_MAX_CPU_COUNT']:os.environ[key]='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
DATA_DIR=Path(os.environ.get('MLA_DATA_DIR','/content' if Path('/content').is_dir() else '.')).resolve()
OUTPUT_DIR=Path(os.environ.get('V9_PORTABLE_OUTPUT_DIR','.')).resolve();OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
required=['Train_dataset.csv','Test_dataset.csv','Data_Dictionary.csv','Sample_Solution.csv']
if not all((DATA_DIR/n).is_file() for n in required):
    from google.colab import files
    files.upload();DATA_DIR=Path.cwd()
assert all((DATA_DIR/n).is_file() for n in required),'All four official CSVs are required.'
PACKAGES={'numpy':'2.5.4','pandas':'3.0.6','scikit-learn':'1.9.1','scipy':'1.18.1','catboost':'1.2.10','interpret-core':'0.7.8','tabpfn':'9.2.0','torch':'2.11.0'}
needed=[]
for package,version in PACKAGES.items():
    try:current=importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:current=None
    if current is None or current.split('+')[0]!=version:needed.append(f'{package}=={version}')
if needed:subprocess.check_call([sys.executable,'-m','pip','install','-q',*needed])
# Model execution uses a fresh child interpreter so updated libraries are not
# mixed with packages already imported by an interactive notebook kernel.
WORK=Path(tempfile.mkdtemp(prefix='v9_portable_',dir=OUTPUT_DIR))
MODEL_PARTS=[]
print('Fresh work directory:',WORK)
''')
md('## Embedded verified model\n\nThese collapsed cells contain your own V7 fitted model. Run all of them; no extra model file upload is needed.')
for offset in range(0,len(model_payload),190000):
 chunk='\n'.join(textwrap.wrap(model_payload[offset:offset+190000],120))
 code('MODEL_PARTS.append("""\n'+chunk+'\n""")',True)
wrapped='\n'.join(textwrap.wrap(packed,120))
code('BUNDLE=json.loads(zlib.decompress(base64.b64decode("""\n'+wrapped+'\n""")))\n'+'''
for name,value in BUNDLE.items():
    relative=Path(name);assert not relative.is_absolute() and '..' not in relative.parts
    (WORK/relative).write_bytes(base64.b64decode(value))
model_bytes=lzma.decompress(base64.b64decode(''.join(MODEL_PARTS)))
manifest=json.loads((WORK/'portable_manifest.json').read_text())
assert hashlib.sha256(model_bytes).hexdigest()==manifest['v7_checkpoint_sha256']
(WORK/'verified_v7.pkl').write_bytes(model_bytes)
del MODEL_PARTS,model_bytes,BUNDLE
''',True)
md('''## Fresh selected-component training and CSV verification

This cell trains TabPFN and checks both components for missing/unseen values, row permutation, and ID/target exclusion. It requires exactly 500 finite nonnegative predictions in original test order. Expected predictions are used only after fitting to check reproduction within INR 0.01.''')
code('''environment=os.environ.copy()
environment['V9_PORTABLE_DATA_DIR']=str(DATA_DIR)
environment['V9_PORTABLE_OUTPUT_DIR']=str(OUTPUT_DIR)
process=subprocess.Popen([sys.executable,'-u',str(WORK/'portable_candidate_worker.py')],env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
for line in process.stdout:print(line,end='',flush=True)
assert process.wait()==0,'Execution failed. Keep scored V7 and inspect the error above.'
checks=json.loads((OUTPUT_DIR/'fresh_execution_checks.json').read_text())
assert checks['status']=='PASS' and checks['maximum_final_difference_INR']<.01
print('Verified CSV SHA256:',checks['csv_sha256'])
''')
md('''## Download and final choice

Use the CSV only if the corrected confirmation report qualifies this candidate. Compare its actual returned portal RMSE with V7's observed 12,024,316.5355221 and retain the lower observed score. Preserve the one-time coding upload until that choice.''')
code('''csv_path=OUTPUT_DIR/'submission.csv'
assert csv_path.is_file()
try:
    from google.colab import files
except ImportError:print('CSV:',csv_path)
else:files.download(str(csv_path))
''')
for c in book.cells:
 if c.cell_type=='code':compile(c.source,'<portable notebook>','exec')
nb.validate(book);nb.write(book,out/'V9_Portable_Colab.ipynb')
(out/'portable_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Built portable draft:',out/'V9_Portable_Colab.ipynb')
print('Notebook bytes:',(out/'V9_Portable_Colab.ipynb').stat().st_size)
