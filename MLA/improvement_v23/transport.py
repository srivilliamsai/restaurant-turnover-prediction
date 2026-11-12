"""Own one free Colab runtime. Never log authenticated URLs or credentials."""
import argparse,base64,hashlib,importlib.util,json,logging,os,subprocess,sys,time,zipfile
from pathlib import Path
logging.disable(logging.CRITICAL)
ROOT=Path(__file__).resolve().parent
SESSION='v23-limix2-t4';REMOTE='/content/mla_v23'
spec=importlib.util.spec_from_file_location('v23_transport_impl',ROOT.parent/'improvement_v22_rank_check/colab_control.py')
impl=importlib.util.module_from_spec(spec);spec.loader.exec_module(impl)
impl.ROOT=ROOT;impl.SESSION=SESSION;impl.REMOTE=REMOTE
from colab_cli.common import state
from colab_cli.contents import ContentsClient
CLI=ROOT.parent/'improvement_v9/colab_cli_env/bin/colab'

def save(name,value):
    path=ROOT/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['allocate','setup','upload','run','collect','progress','release','export-upload','export-run']);parser.add_argument('stage',nargs='?',default='pilot');args=parser.parse_args()
    auth=json.loads((ROOT/'authorization.json').read_text())
    if args.action!='release':assert time.time()<auth['hard_stop_epoch'],'V23 elapsed ceiling reached'
    if args.action=='allocate':
        r=state.client.session.get('https://www.googleapis.com/oauth2/v2/userinfo',timeout=30);r.raise_for_status()
        identity=hashlib.sha256(r.json()['id'].encode()).hexdigest()
        assert identity==auth['account_identity_sha256'],'Selected account changed'
        active=state.client.list_assignments();assert not active,'Another runtime is active; preserve it'
        p=subprocess.run([str(CLI),'--auth','oauth2','new','-s',SESSION,'--gpu','T4'],capture_output=True,text=True,timeout=180)
        if p.returncode:
            save('transport_run/allocation.json',{'status':'failed','returncode':p.returncode})
            raise RuntimeError('Free T4 allocation failed; credential-bearing output suppressed')
        session=state.get_session(SESSION);assert session.accelerator.lower()=='t4'
        save('transport_run/allocation.json',{'status':'allocated','owned_session':SESSION,'epoch':time.time(),'paid':False})
        print('V23 owned free T4 allocated.',flush=True)
    elif args.action=='setup':
        impl.execute("""import sys,subprocess,json,time
from pathlib import Path
root=Path('/content/mla_v23');root.mkdir(exist_ok=True)
requirements=['torch==2.11.0','numpy==2.5.4','pandas==3.0.6','scikit-learn==1.9.1','scipy==1.18.1','einops','kditransform','nvtx','huggingface-hub','typing_extensions','joblib','psutil']
with (root/'setup.log').open('w') as log:
 p=subprocess.run([sys.executable,'-m','pip','install','-q',*requirements],stdout=log,stderr=subprocess.STDOUT,timeout=1000)
 assert p.returncode==0,'Dependency installation failed'
print('V23 isolated dependencies installed.',flush=True)
""",1100)
    elif args.action=='upload':
        archive=ROOT/'runtime_bundle.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
            for name in ['experiment.py','adapter.py','protocol.json','authorization.json','controls.npz','partitions.json','feature_train.csv','feature_test.csv','history_audit.json','source_frozen.json','PLAN.md','RESEARCH_SOURCES.md','research_sources/mlx_manifest.json']:
                bundle.write(ROOT/name,name)
            for folder in ['research_sources/LimiX']:
                for p in (ROOT/folder).rglob('*'):
                    if p.is_file() and p.suffix in ['.py','.json','.toml','.md','.txt','.yml']:bundle.write(p,str(p.relative_to(ROOT)))
            for name in ['source_manifest.json','checkpoint_manifest.json']:bundle.write(ROOT/'research_sources'/name,'research_sources/'+name)
        ContentsClient(state.get_session(SESSION)).upload(str(archive),REMOTE+'/runtime_bundle.zip')
        impl.execute("""import zipfile
from pathlib import Path
root=Path('/content/mla_v23')
with zipfile.ZipFile(root/'runtime_bundle.zip') as z:
 for n in z.namelist():assert not Path(n).is_absolute() and '..' not in Path(n).parts
 z.extractall(root)
history=root/'results/engineering_history';history.mkdir(parents=True,exist_ok=True)
for p in (root/'results').glob('*_failure.json'):
 p.replace(history/(str(__import__('time').time_ns())+'_'+p.name))
print('V23 verified source and data bundle transferred.',flush=True)
""",180)
        save('transport_run/transfer.json',{'uploaded':True,'explicit_user_permission':auth.get('dataset_export_authorized',False),
            'account_identity_sha256':auth['account_identity_sha256'],'owned_session':SESSION,
            'bundle_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'epoch':time.time()})
    elif args.action=='run':
        assert args.stage in ['resolve','pilot','screen','confirm','full']
        code="import os,sys,subprocess\nroot='/content/mla_v23'\nenv=os.environ.copy()\nenv.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')\nwith open(root+'/run.log','a',buffering=1) as log:\n p=subprocess.run([sys.executable,root+'/experiment.py',"+repr(args.stage)+"],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout="+str(int(min(3300,auth['training_stop_epoch']-time.time())))+")\nassert p.returncode==0,'V23 stage failed; collect measured status and run.log'\nprint('V23 stage complete: "+args.stage+"',flush=True)"
        impl.execute(code,min(3400,max(60,auth['training_stop_epoch']-time.time())))
    elif args.action=='export-upload':
        archive=ROOT/'export_bundle.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
            bundle.write(ROOT/'qualified_export.py','qualified_export.py')
            for name in ['scored_v14.csv','scored_v14.ipynb']:bundle.write(ROOT/'reference'/name,'reference/'+name)
            for name in ['Train_dataset.csv','Test_dataset.csv','Data_Dictionary.csv','Sample_Solution.csv']:bundle.write(ROOT.parent/name,'data/'+name)
        ContentsClient(state.get_session(SESSION)).upload(str(archive),REMOTE+'/export_bundle.zip')
        impl.execute("""import zipfile,subprocess,sys
from pathlib import Path
root=Path('/content/mla_v23')
with zipfile.ZipFile(root/'export_bundle.zip') as z:
 for n in z.namelist():assert not Path(n).is_absolute() and '..' not in Path(n).parts
 z.extractall(root)
subprocess.run([sys.executable,'-m','pip','install','-q','nbformat','nbclient'],check=True,timeout=120)
print('Qualified export inputs prepared; no inference performed.')
""",180)
    elif args.action=='export-run':
        remaining=int(auth['training_stop_epoch']-time.time());assert remaining>1
        code="import os,sys,subprocess\nroot='/content/mla_v23'\nenv=os.environ.copy()\nenv.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')\nwith open(root+'/export.log','a',buffering=1) as log:\n p=subprocess.run([sys.executable,root+'/qualified_export.py'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout="+str(remaining)+")\nassert p.returncode==0,'Qualified export failed; inspect export.log'\nprint('V23 fresh qualified export complete.')"
        impl.execute(code,remaining+30)
    elif args.action in ['collect','progress']:
        client=ContentsClient(state.get_session(SESSION))
        if args.action=='progress':
            for name in ['status.json','results/pilot_progress.json','results/pilot.json','results/decision.json']:
                try:print(json.dumps({'file':name,'value':json.loads(impl.read(client,REMOTE+'/'+name))}),flush=True)
                except FileNotFoundError:pass
        else:
            def copy_dir(remote,local):
                local.mkdir(parents=True,exist_ok=True)
                entries=client.list_dir(remote)['content'];names={x['name'] for x in entries}
                for p in local.glob('*_failure.json'):
                    if p.name not in names:
                        h=ROOT/'engineering_history/stale_snapshot';h.mkdir(parents=True,exist_ok=True)
                        p.replace(h/(str(time.time_ns())+'_'+p.name))
                for entry in entries:
                    name=entry['name']
                    if entry['type']=='directory':copy_dir(remote+'/'+name,local/name)
                    elif not name.endswith('.partial'):(local/name).write_bytes(impl.read(client,remote+'/'+name))
            names={e['name']:e for e in client.list_dir(REMOTE)['content']}
            for name in ['results','delivery','status.json','run.log','setup.log','resolved.json','export.log']:
                if name not in names:continue
                if names[name]['type']=='directory':copy_dir(REMOTE+'/'+name,ROOT/'colab_results'/name)
                else:
                    path=ROOT/'colab_results'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(impl.read(client,REMOTE+'/'+name))
            print('V23 evidence collected.',flush=True)
    elif args.action=='release':
        p=subprocess.run([str(CLI),'--auth','oauth2','stop','-s',SESSION],capture_output=True,text=True,timeout=120)
        assert p.returncode==0,'Owned runtime release failed; details suppressed'
        active=state.client.list_assignments()
        save('transport_run/release.json',{'owned_session':SESSION,'release_command_succeeded':True,'active_assignments_after_release':len(active),'epoch':time.time()})
        print(json.dumps({'owned_runtime_released':True,'active_assignments_after_release':len(active)}))

if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps({'status':'error','type':type(error).__name__,'detail':str(error) if isinstance(error,AssertionError) else 'See saved evidence; credential-bearing transport details suppressed.'}),flush=True)
        sys.exit(1)
