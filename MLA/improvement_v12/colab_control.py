"""Colab CLI library transport; authenticated URLs and credentials never printed."""
import argparse,json,logging,os,sys,zipfile
from pathlib import Path
from colab_cli.common import state
from colab_cli.contents import ContentsClient
from colab_cli.runtime import ColabRuntime
os.umask(0o077);logging.disable(logging.CRITICAL)
ROOT=Path(__file__).resolve().parent;REMOTE='/content/mla_v12';SESSION='v12-free-t4'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['setup','upload','foreground','execute','collect','mac','metadata']);parser.add_argument('value',nargs='?');args=parser.parse_args()
    s=state.get_session(SESSION);assert s.accelerator.lower()=='t4'
    def kernel(k):s.kernel_id=k;state.store.add(s)
    def started(k):s.session_id=k;state.store.add(s)
    runtime=ColabRuntime(s.url,s.token,kernel_id=s.kernel_id,session_id=s.session_id,on_kernel_started=kernel,on_session_started=started)
    def hook(x):
        if x.get('output_type')=='stream':print(x.get('text',''),end='',flush=True)
        if x.get('output_type')=='error':print(x.get('ename','Error')+': '+x.get('evalue',''),flush=True)
    def execute(code,timeout=180):
        result=runtime.execute_code(code,output_hook=hook,timeout=timeout)
        errors=[x for x in result if x.get('output_type')=='error']
        if errors:raise RuntimeError(errors[0].get('ename','Error')+': '+errors[0].get('evalue',''))
    try:
        if args.action=='setup':
            profile_path=ROOT/'colab_results/resolved_models.json'
            extra_pins=[]
            if profile_path.exists():
                frozen=json.loads(profile_path.read_text())
                extra_pins=[k+'=='+v for k,v in frozen.get('runtime',{}).get('auxiliary_dependencies',{}).items()]
            execute("""import os,sys,subprocess,json
from pathlib import Path
root=Path('/content/mla_v12');root.mkdir(exist_ok=True)
requirements=['numpy==2.5.4','pandas==3.0.6','scikit-learn==1.9.1','scipy==1.18.1','catboost==1.2.10','interpret-core==0.7.8','tabpfn==9.2.0','torch==2.11.0','pytabkit==1.7.3','nbformat','nbclient']+EXTRA_PINS
with (root/'setup.log').open('a') as log:
 p=subprocess.Popen([sys.executable,'-m','pip','install','-q',*requirements],stdout=log,stderr=subprocess.STDOUT)
 print('V12 pinned dependency installation started.',flush=True)
 assert p.wait()==0,'Dependency installation failed.'
print('V12 pinned installation complete; clean kernel restart follows.',flush=True)
""".replace('EXTRA_PINS',repr(extra_pins)),1800)
        elif args.action=='upload':
            import hashlib
            ready=json.loads((ROOT/'RUN_READY.json').read_text());assert ready['status']=='PASS'
            for name,want in ready['source_identity'].items():
                assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==want,'Source changed after preflight: '+name
            archive=ROOT/'runtime_bundle.zip'
            with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
                for folder in ['reference','provenance','results','transport']:
                    for p in (ROOT/folder).rglob('*'):
                        if p.is_file() and '.partial' not in p.name:z.write(p,str(p.relative_to(ROOT)))
                for p in ROOT.iterdir():
                    if p.is_file() and p.suffix in ['.py','.json','.md'] and p.name!='colab_control.py':z.write(p,p.name)
                for n in ['Train_dataset.csv','Test_dataset.csv','Data_Dictionary.csv','Sample_Solution.csv']:z.write(ROOT.parent/n,'data/'+n)
            ContentsClient(state.get_session(SESSION)).upload(str(archive),REMOTE+'/runtime_bundle.zip')
            execute("""import zipfile
from pathlib import Path
root=Path('/content/mla_v12')
with zipfile.ZipFile(root/'runtime_bundle.zip') as z:
 assert z.testzip() is None
 for n in z.namelist():assert not Path(n).is_absolute() and '..' not in Path(n).parts
 z.extractall(root)
print('V12 official inputs, frozen sources and verified baseline registry restored.')
""")
        elif args.action=='foreground':
            assert args.value in ['audit','pilot','screen','confirm','export']
            supervisor=(ROOT/'transport/stage_supervisor.py').read_text()
            execute("""import os,sys,json,subprocess
from pathlib import Path
root=Path('/content/mla_v12');(root/'transport').mkdir(exist_ok=True)
(root/'transport/stage_supervisor.py').write_text(SUPERVISOR)
env=os.environ.copy()
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LOKY_MAX_CPU_COUNT']:env[key]='2'
env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
with (root/'run.log').open('a') as log:
 p=subprocess.Popen([sys.executable,'-u',str(root/'transport/stage_supervisor.py'),STAGE],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=root)
 (root/'stage_pid.json').write_text(json.dumps({'stage':STAGE,'pid':p.pid}))
 print('V12 supervised foreground stage:',STAGE,'PID',p.pid,flush=True)
 assert p.wait()==0,'V12 supervised stage failed; inspect saved logs.'
""".replace('STAGE',repr(args.value)).replace('SUPERVISOR',repr(supervisor)),43200)
        elif args.action=='execute':execute(Path(args.value).read_text(),1800)
        elif args.action=='mac':
            contents=ContentsClient(state.get_session(SESSION))
            for name in [f'reference_v7_{seed}{suffix}' for seed in [71501,71502,71503] for suffix in ['.npz','.json']]+['mac_compute_receipt.json']:
                p=ROOT/'mac_results'/name
                if p.exists():contents.upload(str(p),REMOTE+'/results/'+name)
            print('Available original-Mac control artifacts uploaded.',flush=True)
        elif args.action=='metadata':
            contents=ContentsClient(state.get_session(SESSION))
            for name in ['build_delivery.py','PLAN.md','README.md','source_inventory.json','IMPLEMENTATION_PROGRESS.md']:
                if (ROOT/name).exists():contents.upload(str(ROOT/name),REMOTE+'/'+name)
        elif args.action=='collect':
            execute("""import zipfile,hashlib,json
from pathlib import Path
root=Path('/content/mla_v12')
from huggingface_hub.constants import HF_HUB_CACHE
inventory=[]
for repository,filename in [('Prior-Labs/TabPFN-v2-reg','tabpfn-v2-regressor.ckpt')]:
 base=Path(HF_HUB_CACHE)/('models--'+repository.replace('/','--'))/'snapshots'
 for p in base.glob('*/'+filename):
  inventory.append({'repo_id':repository,'revision':p.parent.name,'filename':filename,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(root/'results/checkpoint_inventory.json').write_text(json.dumps(inventory,indent=2)+'\\n')
with zipfile.ZipFile(root/'collected_results.zip','w',zipfile.ZIP_DEFLATED) as z:
 for folder in ['results','delivery']:
  for p in (root/folder).rglob('*'):
   if p.is_file() and '.partial' not in p.name:z.write(p,str(p.relative_to(root)))
 for n in ['run.log','setup.log','resolved_models.json','stage_pid.json']:
  p=root/n
  if p.exists():z.write(p,n)
print('V12 result archive prepared.',flush=True)
""",300)
            local=ROOT/'colab_results';local.mkdir(exist_ok=True);archive=local/'collected_results.zip'
            ContentsClient(state.get_session(SESSION)).download(REMOTE+'/collected_results.zip',str(archive))
            with zipfile.ZipFile(archive) as z:
                assert z.testzip() is None
                for n in z.namelist():assert not Path(n).is_absolute() and '..' not in Path(n).parts
                z.extractall(local)
            if (local/'resolved_models.json').exists():
                import shutil
                shutil.copy2(local/'resolved_models.json',ROOT/'resolved_models.json')
            print('Saved completed V12 artifacts locally.',flush=True)
    finally:runtime.stop()

if __name__=='__main__':
    try:main()
    except Exception as e:print('V12 transport failure:',type(e).__name__,'inspect the saved stage log; authenticated transport details are suppressed',flush=True);raise SystemExit(1)
