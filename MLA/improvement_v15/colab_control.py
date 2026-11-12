"""V15 Colab transport; authenticated URLs and credentials are never printed."""
import argparse,base64,hashlib,json,logging,os,subprocess,sys,zipfile
from pathlib import Path
import requests
from colab_cli.common import state
from colab_cli.contents import ContentsClient
from colab_cli.runtime import ColabRuntime
os.umask(0o077)
logging.disable(logging.CRITICAL)
ROOT=Path(__file__).resolve().parent
REMOTE='/content/mla_v15'
SESSION='v15-free-t4'
CLI=ROOT.parent/'improvement_v9/colab_cli_env/bin/colab'
_request=requests.request
def bounded_request(*a,**kw):
    kw.setdefault('timeout',(10,60))
    return _request(*a,**kw)
requests.request=bounded_request

def execute(code,timeout=180):
    session=state.get_session(SESSION)
    assert session.accelerator.lower()=='t4'
    def kernel(k):
        session.kernel_id=k
        state.store.add(session)
    def started(k):
        session.session_id=k
        state.store.add(session)
    runtime=ColabRuntime(session.url,session.token,kernel_id=session.kernel_id,session_id=session.session_id,on_kernel_started=kernel,on_session_started=started)
    def output(item):
        if item.get('output_type')=='stream':print(item.get('text',''),end='',flush=True)
        elif item.get('output_type')=='error':print(item.get('ename','Error')+': remote V15 execution failed; inspect saved logs.',flush=True)
    try:
        result=runtime.execute_code(code,output_hook=output,timeout=timeout)
        assert not any(x.get('output_type')=='error' for x in result),'Remote V15 execution failed'
    finally:runtime.stop()

def read(contents,path):
    x=contents._request('GET',path,params={'content':'1'})
    return base64.b64decode(x['content']) if x.get('format')=='base64' else str(x['content']).encode()
def put(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.backup_partial')
    temp.write_bytes(data)
    temp.replace(path)

def snapshot():
    contents=ContentsClient(state.get_session(SESSION))
    local=ROOT/'colab_results'
    names=['resolved_models.json','run.log','setup.log']
    names+=['results/'+n for n in ['stage_status.json','compute_budget.json','runtime_heartbeat.json','pilot.json','pilot_progress.json','audit.json','experiment_plan.json','reference_verification.json','feature_verification.json','moment_verification.json','screening.csv','component_diagnostics.csv','development_segments.csv','development_folds.csv','development_concentration.json','candidate_lock.json','confirmation_progress.json','confirmation.csv','confirmation_segments.csv','confirmation_folds.csv','decision.json']]
    for name in names:
        try:put(local/name,read(contents,REMOTE+'/'+name))
        except FileNotFoundError:pass
    new=0
    for folder in ['cache','pilot_checkpoints']:
        try:entries=contents.list_dir(REMOTE+'/results/'+folder)['content']
        except FileNotFoundError:continue
        available={x['name'] for x in entries}
        for name in sorted(available):
            if not name.endswith('.json') or '.partial' in name:continue
            path=local/'results'/folder/name
            if path.exists():continue
            raw=read(contents,REMOTE+'/results/'+folder+'/'+name)
            metadata=json.loads(raw)
            if folder=='cache':
                data_name=name[:-5]+'.npz'
                if data_name not in available:continue
                data=read(contents,REMOTE+'/results/'+folder+'/'+data_name)
                assert hashlib.sha256(data).hexdigest()==metadata['sha256']
                put(path.with_suffix('.npz'),data)
                new+=1
            else:assert hashlib.sha256(json.dumps(metadata['result'],sort_keys=True,allow_nan=False).encode()).hexdigest()==metadata['result_sha256']
            put(path,raw)
    status_path=local/'results/stage_status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {}
    print(json.dumps({'snapshot':'saved','new_complete_folds':new,'stage_status':status}),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['allocate','probe','setup','upload','stage','snapshot','collect','release','usage'])
    parser.add_argument('value',nargs='?')
    args=parser.parse_args()
    if args.action=='usage':
        p=subprocess.run([str(CLI),'--auth','oauth2','usage'],capture_output=True,text=True,timeout=120)
        assert p.returncode==0,'Free compute status check failed; credentials suppressed.'
        for line in p.stdout.splitlines():
            if any(line.startswith(prefix) for prefix in ['Current balance:', 'Usage rate:', 'Active assignments:']):print(line)
    elif args.action=='allocate':
        p=subprocess.run([str(CLI),'--auth','oauth2','new','-s',SESSION,'--gpu','T4'],capture_output=True,text=True,timeout=180)
        assert p.returncode==0,'Free T4 allocation failed; authenticated details suppressed.'
        session=state.get_session(SESSION)
        assert session.accelerator.lower()=='t4'
        print(json.dumps({'allocated':True,'owned_session':SESSION,'accelerator':'T4','high_memory_requested':False}),flush=True)
    elif args.action=='probe':
        execute("""import sys,torch,json,importlib.metadata
print(json.dumps({'python':sys.version,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'torch':torch.__version__,'cuda':torch.version.cuda}),flush=True)
""",180)
    elif args.action=='release':
        p=subprocess.run([str(CLI),'--auth','oauth2','stop','-s',SESSION],capture_output=True,text=True,timeout=120)
        assert p.returncode==0,'Owned free runtime release failed'
        print('Owned V15 free T4 released.',flush=True)
    elif args.action=='setup':
        execute("""import os,sys,subprocess
from pathlib import Path
root=Path('/content/mla_v15');root.mkdir(exist_ok=True)
requirements=['numpy==2.5.4','pandas==3.0.6','scikit-learn==1.9.1','scipy==1.18.1','catboost==1.2.10','interpret-core==0.7.8','tabpfn==9.2.0','tabicl==2.2.0','torch==2.11.0','skrub','pyarrow','safetensors','nbformat','nbclient']
with (root/'setup.log').open('a') as log:
    p=subprocess.run([sys.executable,'-m','pip','install','-q',*requirements],stdout=log,stderr=subprocess.STDOUT,timeout=1500)
    assert p.returncode==0,'Pinned installation failed; inspect setup.log.'
print('V15 pinned dependencies installed; restart kernel before training.',flush=True)
""",1650)
    elif args.action=='upload':
        frozen=json.loads((ROOT/'source_frozen.json').read_text())
        for name,want in frozen['sources'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==want,'V15 source changed after preflight: '+name
        archive=ROOT/'runtime_bundle.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
            for folder in ['reference','provenance','results']:
                for p in (ROOT/folder).rglob('*'):
                    if p.is_file() and '.partial' not in p.name:bundle.write(p,str(p.relative_to(ROOT)))
            for p in ROOT.iterdir():
                if p.is_file() and p.suffix in ['.py','.json','.md']:bundle.write(p,p.name)
            for name in ['Train_dataset.csv','Test_dataset.csv','Data_Dictionary.csv','Sample_Solution.csv']:bundle.write(ROOT.parent/name,'data/'+name)
        ContentsClient(state.get_session(SESSION)).upload(str(archive),REMOTE+'/runtime_bundle.zip')
        execute("""import zipfile
from pathlib import Path
root=Path('/content/mla_v15')
with zipfile.ZipFile(root/'runtime_bundle.zip') as bundle:
    assert bundle.testzip() is None
    for name in bundle.namelist():assert not Path(name).is_absolute() and '..' not in Path(name).parts
    bundle.extractall(root)
print('V15 sources, official inputs and verified controls uploaded.',flush=True)
""")
    elif args.action=='stage':
        assert args.value in ['audit','pilot','screen','confirm','export']
        code="""import os,sys,subprocess,json,time
from pathlib import Path
root=Path('/content/mla_v15');sys.path.insert(0,str(root))
from core import Engine
stage=STAGE
e=Engine(root/'data',root/'results')
timeout=e.budget.remaining(stage)
assert timeout>0,'V15 stage allowance exhausted'
env=os.environ.copy()
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LOKY_MAX_CPU_COUNT']:env[key]='2'
env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
with (root/'run.log').open('a') as log:
    p=subprocess.Popen([sys.executable,'-u',str(root/'workflow.py'),stage,'--data-dir',str(root/'data'),'--output-dir',str(root/'results')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=root)
    print('V15 foreground stage:',stage,'timeout seconds',round(timeout),flush=True)
    try:code=p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        import signal
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        raise RuntimeError('V15 stage timeout; completed checkpoints preserved.')
    assert code==0,'V15 stage failed; inspect run.log and saved stage status.'
print('V15 stage complete:',stage,flush=True)
""".replace('STAGE',repr(args.value))
        execute(code,10000)
    elif args.action=='snapshot':snapshot()
    elif args.action=='collect':
        execute("""import zipfile
from pathlib import Path
root=Path('/content/mla_v15')
with zipfile.ZipFile(root/'collected_results.zip','w',zipfile.ZIP_DEFLATED) as bundle:
    for folder in ['results','delivery']:
        for p in (root/folder).rglob('*'):
            if p.is_file() and '.partial' not in p.name and not any(x in p.parts for x in ['model_weights','__pycache__']) and p.suffix != '.safetensors':bundle.write(p,str(p.relative_to(root)))
    for name in ['V15_Delivery.zip','resolved_models.json','run.log','setup.log']:
        p=root/name
        if p.exists():bundle.write(p,name)
print('V15 measured result archive prepared.',flush=True)
""",300)
        local=ROOT/'colab_results'
        local.mkdir(exist_ok=True)
        archive=local/'collected_results.zip'
        ContentsClient(state.get_session(SESSION)).download(REMOTE+'/collected_results.zip',str(archive))
        with zipfile.ZipFile(archive) as bundle:
            assert bundle.testzip() is None
            for name in bundle.namelist():assert not Path(name).is_absolute() and '..' not in Path(name).parts
            bundle.extractall(local)
        print('V15 measured outputs collected locally.',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as error:
        print('V15 transport failure:',type(error).__name__,'; authenticated details suppressed.',flush=True)
        raise SystemExit(1)

