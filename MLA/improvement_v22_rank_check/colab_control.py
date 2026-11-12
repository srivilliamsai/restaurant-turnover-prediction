"""V22 Colab transport; authenticated URLs and credentials are never printed."""
import argparse,base64,hashlib,json,logging,os,subprocess,sys,zipfile
from pathlib import Path
import requests
from colab_cli import auth as colab_auth
from colab_cli.common import state
from colab_cli.contents import ContentsClient
from colab_cli.runtime import ColabRuntime
os.umask(0o077)
logging.disable(logging.CRITICAL)
def no_interactive_signin(*args, **kwargs):
    raise RuntimeError('Existing Colab credentials could not refresh; user sign-in is required.')
colab_auth._run_remote_flow = no_interactive_signin
ROOT=Path(__file__).resolve().parent
REMOTE='/content/mla_v22_rank_check'
SESSION='v22-rank-check-t4'
CLI=ROOT.parent/'improvement_v9/colab_cli_env/bin/colab'
_request=requests.request
def bounded_request(*a,**kw):
    kw.setdefault('timeout',(10,60))
    return _request(*a,**kw)
requests.request=bounded_request

def execute(code,timeout=180):
    # A verified server marker also handles a lost websocket completion message.
    import threading,time,uuid
    session=state.get_session(SESSION)
    assert session.accelerator.lower()=='t4'
    if not session.kernel_id:
        # Select an existing kernel, or create exactly one before connecting.
        for attempt in range(45):
            response=requests.request('GET',session.url.rstrip('/')+'/api/kernels',params={'authuser':'0','colab-runtime-proxy-token':session.token})
            if response.ok:
                ready=response.json()
                if not ready:
                    response=requests.request('POST',session.url.rstrip('/')+'/api/kernels',params={'authuser':'0','colab-runtime-proxy-token':session.token},json={'name':'python3'},timeout=(10,90))
                    response.raise_for_status()
                    ready=[response.json()]
                if ready:
                    session.kernel_id=ready[0]['id']
                    state.store.add(session)
                    print(json.dumps({'existing_kernel_selected':True}),flush=True)
                    break
            time.sleep(2)
        assert session.kernel_id,'Owned Colab default kernel did not become ready.'
    def kernel(k):
        session.kernel_id=k
        state.store.add(session)
    def started(k):
        session.session_id=k
        state.store.add(session)
    runtime=ColabRuntime(session.url,session.token,kernel_id=session.kernel_id,session_id=session.session_id,on_kernel_started=kernel,on_session_started=started)
    ident=uuid.uuid4().hex
    marker=REMOTE+'/transport_markers/'+ident+'.json'
    body="import json as _transport_json\nfrom pathlib import Path as _TransportPath\n_transport_path=_TransportPath("+repr(marker)+")\n_transport_path.parent.mkdir(parents=True,exist_ok=True)\nif _transport_path.exists():\n    return\n_transport_path.write_text(_transport_json.dumps({'id':"+repr(ident)+",'status':'running'}))\ntry:\n    exec(compile("+repr(code)+",'<V22 owned foreground command>','exec'),{'__name__':'__main__'})\nexcept BaseException as _transport_error:\n    _transport_path.write_text(_transport_json.dumps({'id':"+repr(ident)+",'status':'error','error_type':type(_transport_error).__name__}))\n    raise\nelse:\n    _transport_path.write_text(_transport_json.dumps({'id':"+repr(ident)+",'status':'ok'}))\n"
    wrapped='def _v22_owned_command():\n'+''.join('    '+line+'\n' for line in body.splitlines())+'\n_v22_owned_command()\n'
    failures=[]
    def output(item):
        if item.get('output_type')=='stream':print(item.get('text',''),end='',flush=True)
        elif item.get('output_type')=='error':
            import re
            reason=re.sub(r'https?://\S+','<url>',str(item.get('evalue',''))).replace(session.token,'<credential>')
            print(item.get('ename','Error')+': '+reason[:1000],flush=True)
    def run():
        try:runtime.execute_code(wrapped,output_hook=output,timeout=timeout)
        except BaseException as error:
            failures.append(type(error).__name__)
            # Safe diagnostic only; exception text can contain authenticated URLs.
            import traceback
            print(json.dumps({'connection_error_type':type(error).__name__,
                              'frames':[{'file':Path(x.filename).name,'line':x.lineno,'function':x.name} for x in traceback.extract_tb(error.__traceback__)[-5:]]}),flush=True)
    worker=threading.Thread(target=run,daemon=True)
    worker.start()
    contents=ContentsClient(session)
    deadline=time.monotonic()+timeout
    reconnects=0
    failed_since=None
    try:
        while time.monotonic()<deadline:
            try:record=json.loads(read(contents,marker))
            except FileNotFoundError:record=None
            if record is not None:
                assert record['id']==ident
                assert record['status']!='error','Remote command failed: '+record.get('error_type','unknown')
                if record['status']=='ok':
                    print(json.dumps({'server_completion_verified':True}),flush=True)
                    return
            if record is None and failures and not worker.is_alive():
                failed_since=failed_since or time.monotonic()
                if time.monotonic()-failed_since>=15:
                    assert reconnects<2,'Owned kernel connection failed before execution after bounded retries'
                    runtime.stop()
                    session=state.get_session(SESSION)
                    session.session_id=None
                    state.store.add(session)
                    runtime=ColabRuntime(session.url,session.token,kernel_id=session.kernel_id,
                        on_kernel_started=kernel,on_session_started=started)
                    contents=ContentsClient(session)
                    failures.clear();failed_since=None;reconnects+=1
                    print(json.dumps({'owned_connection_retry':reconnects}),flush=True)
                    worker=threading.Thread(target=run,daemon=True)
                    worker.start()
            time.sleep(3)
        raise TimeoutError('Owned V22 command exceeded its bounded allowance; saved folds retained.')
    finally:
        runtime.stop()
        worker.join(timeout=5)

def read(contents,path):
    x=contents._request('GET',path,params={'content':'1'})
    return base64.b64decode(x['content']) if x.get('format')=='base64' else str(x['content']).encode()
def put(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.backup_partial')
    temp.write_bytes(data)
    temp.replace(path)

def snapshot():
    from concurrent.futures import ThreadPoolExecutor
    contents=ContentsClient(state.get_session(SESSION))
    root_entries=contents.list_dir(REMOTE)['content']  # A vanished runtime must not look like a successful empty backup.
    root_names={entry['name'] for entry in root_entries}
    result_names={entry['name'] for entry in contents.list_dir(REMOTE+'/results')['content']} if 'results' in root_names else set()
    local=ROOT/'colab_results'
    names=['resolved_models.json','run.log','setup.log']
    names+=['results/'+n for n in ['stage_status.json','compute_budget.json','runtime_heartbeat.json','pilot.json','pilot_progress.json','pilot_C8.log','pilot_KS8.log','pilot_D8.log','pilot_KM8.log','pilot_CONTROLS.log','control_pilot.json','current_fit_progress.json','audit.json','experiment_plan.json','reference_verification.json','feature_verification.json','moment_verification.json','screening.csv','component_diagnostics.csv','development_segments.csv','development_folds.csv','development_concentration.json','candidate_lock.json','confirmation_progress.json','confirmation.csv','confirmation_segments.csv','confirmation_folds.csv','decision.json']]
    names=[name for name in names if (name.split('/',1)[1] in result_names if name.startswith('results/') else name in root_names)]
    def copy_record(name):
        try:put(local/name,read(contents,REMOTE+'/'+name))
        except FileNotFoundError:pass
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(copy_record,names))
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
    parser.add_argument('action',choices=['allocate','probe','setup','upload','stage','pipeline','snapshot','collect','release','usage','mac-upload','account-check','progress'])
    parser.add_argument('value',nargs='?')
    args=parser.parse_args()
    if args.action=='progress':
        client=ContentsClient(state.get_session(SESSION))
        for name in ['setup_status.json','results/pilot_progress.json','results/stage_status.json']:
            try:
                record=json.loads(read(client,REMOTE+'/'+name))
                if name=='setup_status.json':record={k:v for k,v in record.items() if k not in ['tabicl','kumo','baseline','candidates']}
                print(json.dumps({'file':name,'value':record}),flush=True)
            except FileNotFoundError:pass
    elif args.action=='usage':
        p=subprocess.run([str(CLI),'--auth','oauth2','usage'],capture_output=True,text=True,timeout=120)
        assert p.returncode==0,'Free compute status check failed; credentials suppressed.'
        for line in p.stdout.splitlines():
            if any(line.startswith(prefix) for prefix in ['Current balance:', 'Usage rate:', 'Active assignments:']):print(line)
    elif args.action=='allocate':
        p=subprocess.run([str(CLI),'--auth','oauth2','new','-s',SESSION,'--gpu','T4'],capture_output=True,text=True,timeout=180)
        if p.returncode:
            raw=(p.stdout+p.stderr).lower()
            diagnosis={'status':'allocation_failed','exit_code':p.returncode,
                'indicators':[word for word in ['quota','too many','not available','unavailable','resource exhausted','resource_exhausted','403','401','429','502','503','504','timeout','readtimeout','connectionerror','auth','permission'] if word in raw]}
            (ROOT/'transport_run').mkdir(exist_ok=True)
            (ROOT/'transport_run/allocation_diagnosis.json').write_text(json.dumps(diagnosis,indent=2)+'\n')
            print(json.dumps(diagnosis),flush=True)
        assert p.returncode==0,'Free T4 allocation failed; authenticated details suppressed.'
        session=state.get_session(SESSION)
        assert session.accelerator.lower()=='t4'
        print(json.dumps({'allocated':True,'owned_session':SESSION,'accelerator':'T4','high_memory_requested':False}),flush=True)
    elif args.action=='probe':
        execute("""import sys,torch,json,importlib.metadata
print(json.dumps({'python':sys.version,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'torch':torch.__version__,'cuda':torch.version.cuda,'dependencies':{k:importlib.metadata.version(k) for k in ['numpy','pandas','scikit-learn','scipy','catboost','interpret-core','tabpfn','tabicl','numba','synthefy']}}),flush=True)
""",180)
    elif args.action=='release':
        p=subprocess.run([str(CLI),'--auth','oauth2','stop','-s',SESSION],capture_output=True,text=True,timeout=120)
        assert p.returncode==0,'Owned free runtime release failed'
        print('Owned V22 free T4 released.',flush=True)
    elif args.action=='account-check':
        import time
        response=state.client.session.get('https://www.googleapis.com/oauth2/v2/userinfo',timeout=(10,30))
        response.raise_for_status();identity=hashlib.sha256(response.json()['id'].encode()).hexdigest()
        authorization=json.loads((ROOT/'authorization.json').read_text());assert identity==authorization['account_identity_sha256']
        active=state.client.list_assignments();usage=state.client.get_consumption_user_info()
        record={'user_selected_account_verified':True,'active_assignments':len(active),'epoch':time.time(),'account_identity_sha256':identity}
        (ROOT/'transport_run').mkdir(exist_ok=True);put(ROOT/'transport_run/account_check.json',json.dumps(record,indent=2).encode())
        assert not active,'An existing runtime is active; do not allocate another or stop unrelated work'
        print(json.dumps(record))
    elif args.action=='mac-upload':
        contents=ContentsClient(state.get_session(SESSION))
        archive=ROOT/'mac_controls.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
            for p in (ROOT/'mac_controls').glob('*'):
                if p.suffix in ['.json','.npz'] and '.partial' not in p.name:bundle.write(p,'mac_controls/'+p.name)
        contents.upload(str(archive),REMOTE+'/mac_controls.zip')
        execute("from pathlib import Path;import zipfile\nwith zipfile.ZipFile('/content/mla_v22_rank_check/mac_controls.zip') as z:\n for n in z.namelist():assert not Path(n).is_absolute() and '..' not in Path(n).parts\n z.extractall('/content/mla_v22_rank_check')\nprint('Verified original-Mac controls transferred.')",180)
    elif args.action=='setup':
        execute("""import os,sys,subprocess
from pathlib import Path
root=Path('/content/mla_v22_rank_check');root.mkdir(exist_ok=True)
requirements=['numpy==2.5.4','pandas==3.0.6','scikit-learn==1.9.1','scipy==1.18.1','catboost==1.2.10','interpret-core==0.7.8','tabpfn==9.2.0','tabicl==2.2.0','torch==2.11.0','skrub==0.10.1','pyarrow==23.0.1','safetensors','nbformat','nbclient','psutil']
with (root/'setup.log').open('a') as log:
    p=subprocess.run([sys.executable,'-m','pip','install','-q',*requirements],stdout=log,stderr=subprocess.STDOUT,timeout=1500)
    assert p.returncode==0,'Pinned installation failed; inspect setup.log.'
print('V22 pinned dependencies installed; restart kernel before training.',flush=True)
""",1650)
    elif args.action=='upload':
        frozen=json.loads((ROOT/'source_frozen.json').read_text())
        for name,want in frozen['sources'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==want,'V22 source changed after preflight: '+name
        archive=ROOT/'runtime_bundle.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
            for folder in ['reference','provenance','results','original_results','engineering_sources','mac_controls']:
                for p in (ROOT/folder).rglob('*'):
                    if p.is_file() and '.partial' not in p.name and (folder!='engineering_sources' or p.suffix in ['.py','.json','.toml','.md']):bundle.write(p,str(p.relative_to(ROOT)))
            for p in ROOT.iterdir():
                if p.is_file() and p.suffix in ['.py','.json','.md']:bundle.write(p,p.name)
            for name in ['Train_dataset.csv','Test_dataset.csv','Data_Dictionary.csv','Sample_Solution.csv']:bundle.write(ROOT.parent/name,'data/'+name)
        # The Mac engineering witness is uploaded without model credentials.
        ContentsClient(state.get_session(SESSION)).upload(str(archive),REMOTE+'/runtime_bundle.zip')
        execute("""import zipfile
from pathlib import Path
root=Path('/content/mla_v22_rank_check')
with zipfile.ZipFile(root/'runtime_bundle.zip') as bundle:
    assert bundle.testzip() is None
    import json,shutil,hashlib
    previous=root/'source_frozen.json'
    incoming=json.loads(bundle.read('source_frozen.json'))
    if previous.exists() and json.loads(previous.read_text())!=incoming:
        results=root/'results'
        assert not list((results/'cache').glob('*.npz')) and not (results/'screening.csv').exists(),'Source repair would invalidate existing accuracy results'
        archive=root/'pre_source_repair'/hashlib.sha256(previous.read_bytes()).hexdigest()[:16]
        archive.mkdir(parents=True,exist_ok=True)
        if results.exists():
            shutil.copytree(results,archive/'results',dirs_exist_ok=True)
            for path in results.iterdir():
                if path.name=='compute_budget.json':continue
                if path.is_dir():shutil.rmtree(path)
                else:path.unlink()
        shutil.copy2(previous,archive/'source_frozen.json')
        (root/'resolved_models.json').unlink(missing_ok=True)
    for name in bundle.namelist():assert not Path(name).is_absolute() and '..' not in Path(name).parts
    bundle.extractall(root)
print('V22 sources, official inputs and verified controls uploaded.',flush=True)
""")
    elif args.action in ['stage','pipeline']:
        stages=['audit','pilot','screen','confirm','export'] if args.action=='pipeline' else [args.value]
        assert all(x in ['audit','pilot','screen','confirm','export'] for x in stages)
        code="""import os,sys,subprocess,json,time
from pathlib import Path
root=Path('/content/mla_v22_rank_check');sys.path.insert(0,str(root))
from core import Engine
stage=STAGE
e=Engine(root/'data',root/'results')
timeout=e.budget.remaining(stage)
assert timeout>0,'V22 stage allowance exhausted'
env=os.environ.copy()
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LOKY_MAX_CPU_COUNT']:env[key]='2'
env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
with (root/'run.log').open('a') as log:
    p=subprocess.Popen([sys.executable,'-u',str(root/'rank_check.py'),stage,'--data-dir',str(root/'data'),'--output-dir',str(root/'results')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=root)
    print('V22 foreground stage:',stage,'timeout seconds',round(timeout),flush=True)
    try:code=p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        import signal
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        raise RuntimeError('V22 stage timeout; completed checkpoints preserved.')
    assert code==0,'V22 stage failed; inspect run.log and saved stage status.'
print('V22 stage complete:',stage,flush=True)
""".replace('STAGE',repr(args.value))
        if args.action=='pipeline':
            # Keep one foreground kernel connection throughout the fixed workflow.
            # Each unchanged stage still enforces its own persisted wall-clock budget.
            code='\n'.join("exec(compile("+repr(code.replace('stage='+repr(args.value),'stage='+repr(stage)))+",'<V22 fixed pipeline stage>','exec'))" for stage in stages)
        execute(code,28800)
    elif args.action=='snapshot':snapshot()
    elif args.action=='collect':
        execute("""import zipfile,json,shutil,sys
from pathlib import Path
root=Path('/content/mla_v22_rank_check')
fresh=root/'delivery/fresh_execution'
temporary=list(fresh.glob('v22_fresh_*'))
if temporary:
    checks=json.loads((fresh/'fresh_execution_checks.json').read_text())
    assert checks['status']=='PASS' and checks['maximum_final_difference_INR']<.01
    for path in temporary:
        assert path.is_dir() and path.parent==fresh
        shutil.rmtree(path)
    sys.path.insert(0,str(root))
    from core import Engine
    import delivery
    delivery.package(Engine(root/'data',root/'results'),json.loads((root/'results/decision.json').read_text()),root/'delivery')
    print('Verified temporary fitting environments removed from delivery; executed notebook and result checks retained.',flush=True)
with zipfile.ZipFile(root/'collected_results.zip','w',zipfile.ZIP_DEFLATED) as bundle:
    for folder in ['results','delivery']:
        for p in (root/folder).rglob('*'):
            if p.is_file() and '.partial' not in p.name and not any(x in p.parts for x in ['model_weights','__pycache__']) and p.suffix != '.safetensors':bundle.write(p,str(p.relative_to(root)))
    for name in ['V22_Delivery.zip','resolved_models.json','run.log','setup.log']:
        p=root/name
        if p.exists():bundle.write(p,name)
print('V22 measured result archive prepared.',flush=True)
""",300)
        local=ROOT/'colab_results'
        local.mkdir(exist_ok=True)
        archive=local/'collected_results.zip'
        ContentsClient(state.get_session(SESSION)).download(REMOTE+'/collected_results.zip',str(archive))
        with zipfile.ZipFile(archive) as bundle:
            assert bundle.testzip() is None
            for name in bundle.namelist():assert not Path(name).is_absolute() and '..' not in Path(name).parts
            bundle.extractall(local)
        print('V22 measured outputs collected locally.',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as error:
        print('V22 transport failure:',type(error).__name__,'; authenticated details suppressed.',flush=True)
        raise SystemExit(1)



