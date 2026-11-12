"""Self-contained coding notebooks for qualified CSVs, freshly executed before use."""
import base64
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import textwrap
import zlib
import nbformat as nb


def build_and_verify(e, decision):
    from engine import ROOT, STEM, sha, save
    records=[]
    for number,choice in enumerate(decision['trials'],1):
        directory=e.out.parent/'candidates'/f'trial_{number}'
        csv=directory/f'{STEM}.csv';path=directory/f'{STEM}.ipynb'
        names=['legacy_reference.py','legacy_models.py','baseline.py','modeling.py','engine.py',
               'workflow.py','pilot.py','gpu_runner.py','reference_manifest.json','v5_contract.json']
        bundle={n:base64.b64encode((ROOT/n).read_bytes()).decode() for n in names}
        bundle.update({f'scored_v7_backup/{p.name}':base64.b64encode(p.read_bytes()).decode()
                       for p in (ROOT/'scored_v7_backup').glob('*')})
        bundle['expected.csv']=base64.b64encode(csv.read_bytes()).decode()
        packages={'numpy':'2.5.4','pandas':'3.0.6','scikit-learn':'1.9.1','scipy':'1.18.1','catboost':'1.2.10','interpret-core':'0.7.8'}
        if choice['family']=='tabpfn':packages['tabpfn']='9.2.0'
        book=nb.v4.new_notebook(metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'colab':{'name':path.name}})
        book.cells=[nb.v4.new_markdown_cell('# Restaurant Turnover — qualified V9 coding submission\n\nsubmission · submission · submission\n\nFrozen configuration: '+json.dumps({k:choice[k] for k in ['name','mode','weight']})+'\n\nThis notebook trains the selected implementation afresh. Its expected CSV is used only for a post-fit reproduction check. Portal accuracy is unknown until this exact CSV is evaluated.')]
        book.cells.append(nb.v4.new_code_cell('''import os,sys,json,base64,hashlib,subprocess,tempfile,importlib.metadata
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LOKY_MAX_CPU_COUNT']:os.environ[k]='2'
PACKAGES='''+repr(packages)+'''
needed=[]
for package,version in PACKAGES.items():
    try: current=importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError: current=None
    if current!=version:needed.append(f'{package}=={version}')
if needed:subprocess.check_call([sys.executable,'-m','pip','install','-q',*needed])
import numpy as np,pandas as pd
DATA_DIR=Path(os.environ.get('V9_CANDIDATE_INPUT_DIR','.')).resolve()
required=['Train_dataset.csv','Test_dataset.csv','Data_Dictionary.csv','Sample_Solution.csv']
if not all((DATA_DIR/n).exists() for n in required):
    from google.colab import files
    files.upload();DATA_DIR=Path.cwd()
OUTPUT=Path(os.environ.get('V9_CANDIDATE_OUTPUT_DIR','.')).resolve();OUTPUT.mkdir(parents=True,exist_ok=True)
WORK=Path(tempfile.mkdtemp(prefix='v9_candidate_',dir=OUTPUT))
'''))
        packed=base64.b64encode(zlib.compress(json.dumps(bundle).encode(),9)).decode()
        wrapped='\n'.join(textwrap.wrap(packed,width=120))
        book.cells.append(nb.v4.new_code_cell('import zlib\nBUNDLE=json.loads(zlib.decompress(base64.b64decode("""\n'+wrapped+'\n""")))'+'''
for relative,value in BUNDLE.items():
    p=WORK/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(value))
sys.path.insert(0,str(WORK))
import baseline,modeling,legacy_models
from engine import Engine,check_behavior,validate_csv
e=Engine(DATA_DIR,WORK/'results')
CHOICE='''+repr({k:choice[k] for k in ['name','mode','weight','family']})+'''
print('Frozen selection:',CHOICE)
'''))
        book.cells.append(nb.v4.new_code_cell('''b,fitted_base=baseline.fit_full(e.train,e.test,e.y,e.contract,print)
expected_base=pd.read_csv(WORK/'scored_v7_backup'/'submission.csv')['Annual Turnover'].to_numpy(float)
assert np.max(np.abs(b-expected_base))<.01
fitted=modeling.fit_model(e.train,e.y,modeling.SPECS[CHOICE['name']],CHOICE['mode'],print)
p=(1-CHOICE['weight'])*b+CHOICE['weight']*modeling.predict_model(fitted,e.test)
predict=lambda raw:(1-CHOICE['weight'])*baseline.predict(fitted_base,raw)+CHOICE['weight']*modeling.predict_model(fitted,raw)
checks=check_behavior(predict,e.test,p)
expected=pd.read_csv(WORK/'expected.csv')['Annual Turnover'].to_numpy(float)
difference=float(np.max(np.abs(p-expected)));assert difference<.01
frame=pd.DataFrame({'Registration Number':e.test['Registration Number'],'Annual Turnover':p})
validate_csv(frame,e.test)
output_path=OUTPUT/'submission.csv';frame.to_csv(output_path,index=False)
validate_csv(pd.read_csv(output_path),e.test)
print('PASS: fresh reproduction within INR 0.01; max difference:',difference)
print('CSV SHA256:',hashlib.sha256(output_path.read_bytes()).hexdigest())
print('Checks:',checks)
'''))
        book.cells.append(nb.v4.new_code_cell('''try:
    from google.colab import files
except ImportError:pass
else:files.download(str(output_path))
'''))
        for cell in book.cells:
            if cell.cell_type=='code':compile(cell.source,'<candidate>','exec')
        nb.validate(book);nb.write(book,path)
        target=directory/'fresh_execution';target.mkdir(exist_ok=True)
        environment=os.environ.copy();environment['V9_CANDIDATE_INPUT_DIR']=str(e.data_dir);environment['V9_CANDIDATE_OUTPUT_DIR']=str(target)
        started=time.monotonic()
        with (directory/'notebook_execution.log').open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'candidate_notebook.py'),'execute',str(path)],env=environment,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=10800)
        executed=nb.read(path,as_version=4)
        assert all(c.execution_count is not None for c in executed.cells if c.cell_type=='code')
        records.append({'trial':number,'notebook_sha256':sha(path),'csv_sha256':sha(csv),'fresh_process':True,'seconds':time.monotonic()-started})
    save(e.out/'candidate_notebook_checks.json',records)
    return records


def execute(path):
    book=nb.read(path,as_version=4);namespace={'__name__':'__main__'};count=0
    # A standalone process has no attached Colab frontend: suppress download widgets.
    sys.modules['google.colab']=None
    for c in book.cells:
        if c.cell_type!='code':continue
        count+=1;capture=io.StringIO();c.execution_count=count
        with contextlib.redirect_stdout(capture):exec(compile(c.source,'<candidate>','exec'),namespace)
        c.outputs=[nb.v4.new_output('stream',name='stdout',text=capture.getvalue())]
        print(capture.getvalue(),flush=True);nb.write(book,path)


if __name__=='__main__':
    if sys.argv[1]=='execute':execute(Path(sys.argv[2]))
    else:
        from engine import Engine
        e=Engine();build_and_verify(e,json.loads((e.out/'decision.json').read_text()))
