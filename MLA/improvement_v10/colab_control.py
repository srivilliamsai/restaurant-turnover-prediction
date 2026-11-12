"""Authorized free-T4 transport. Never prints credentials, proxy URLs or tokens."""
import argparse
import hashlib
import json
import logging
import os
import sys
import zipfile
from pathlib import Path
from colab_cli.common import state
from colab_cli.contents import ContentsClient
from colab_cli.runtime import ColabRuntime

os.umask(0o077)
logging.getLogger().setLevel(logging.CRITICAL)
ROOT=Path(__file__).resolve().parent
REMOTE='/content/mla_v10'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['setup','status','upload','restore','metadata','city','merge_mac','stage','foreground','collect','execute','release'])
    parser.add_argument('value',nargs='?')
    args=parser.parse_args()
    session=state.get_session('v10-free-t4')
    assert session.accelerator.lower()=='t4', 'Only the authorized free T4 is permitted.'
    def kernel(identifier):
        session.kernel_id=identifier;state.store.add(session)
    def started(identifier):
        session.session_id=identifier;state.store.add(session)
    runtime=ColabRuntime(session.url,session.token,kernel_id=session.kernel_id,
                         session_id=session.session_id,on_kernel_started=kernel,on_session_started=started)
    def output(item):
        if item.get('output_type')=='stream':print(item.get('text',''),end='',flush=True)
        if item.get('output_type')=='error':print(item.get('ename','Error')+': '+item.get('evalue',''),flush=True)
    def execute(code,timeout=180):
        result=runtime.execute_code(code,output_hook=output,timeout=timeout)
        errors=[x for x in result if x.get('output_type')=='error']
        if errors:raise RuntimeError(errors[0].get('ename','RemoteError')+': '+errors[0].get('evalue',''))
        return result
    try:
        if args.action=='setup':
            execute("""import os,sys,subprocess,json
from pathlib import Path
root=Path('/content/mla_v10');root.mkdir(exist_ok=True)
requirements=['numpy==2.5.4','pandas==3.0.6','scikit-learn==1.9.1','scipy==1.18.1','catboost==1.2.10','interpret-core==0.7.8','tabpfn==9.2.0','torch==2.11.0','nbformat','nbclient']
log=(root/'setup.log').open('a')
p=subprocess.Popen([sys.executable,'-m','pip','install','-q',*requirements],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'setup_pid.json').write_text(json.dumps({'pid':p.pid,'paid_resources_requested':False}))
print('V10 pinned dependency setup started; PID',p.pid)
assert p.wait()==0,'Pinned package installation failed; inspect setup.log.'
print('Pinned dependency installation complete.')
""",1800)
        elif args.action=='status':
            execute("""import os,json
from pathlib import Path
root=Path('/content/mla_v10')
for name in ['setup_pid.json','stage_pid.json']:
 p=root/name
 if p.exists():
  d=json.loads(p.read_text())
  try:os.kill(d['pid'],0);d['process_present']=True
  except ProcessLookupError:d['process_present']=False
  print(name,json.dumps(d))
for name in ['run.log']:
 p=root/name
 if p.exists():print(name);print(p.read_text()[-4000:])
for name in ['stage_status.json','pilot.json','screening_progress.json','screening_decision.json','candidate_lock.json','decision.json','compute_budget.json']:
 p=root/'results'/name
 if p.exists():
  d=json.loads(p.read_text())
  print(name,json.dumps({k:v for k,v in d.items() if k in ['status','stage','choice','qualified','complete','settings','configurations','settings_complete','planned','seconds','by_stage','repeat_difference_INR','decision','error']}))
""")
        elif args.action=='upload':
            contents=ContentsClient(state.get_session('v10-free-t4'))
            archive=ROOT/'runtime_bundle.zip'
            with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
                for p in ROOT.iterdir():
                    if p.is_file() and p.suffix in ['.py','.json','.md'] and p.name!='colab_control.py':z.write(p,p.name)
                for p in (ROOT/'reference').rglob('*'):
                    if p.is_file():z.write(p,str(p.relative_to(ROOT)))
                for name in ['Train_dataset.csv','Test_dataset.csv','Data_Dictionary.csv','Sample_Solution.csv']:
                    z.write(ROOT.parent/name,'data/'+name)
            contents.upload(str(archive),REMOTE+'/runtime_bundle.zip')
            execute("""import zipfile
from pathlib import Path
root=Path('/content/mla_v10')
with zipfile.ZipFile(root/'runtime_bundle.zip') as z:
 for n in z.namelist():assert not Path(n).is_absolute() and '..' not in Path(n).parts
 z.extractall(root)
print('V10 source/data/reference bundle uploaded and extracted.')
""")
        elif args.action=='restore':
            contents=ContentsClient(state.get_session('v10-free-t4'))
            archive=ROOT/'resume_checkpoints.zip'
            with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
                for folder in ['results','preflight_batch_failure']:
                    for p in (ROOT/'colab_results'/folder).rglob('*'):
                        if p.is_file() and '.partial' not in p.name:z.write(p,str(p.relative_to(ROOT/'colab_results')))
            contents.upload(str(archive),REMOTE+'/resume_checkpoints.zip')
            execute("""import zipfile
from pathlib import Path
root=Path('/content/mla_v10')
with zipfile.ZipFile(root/'resume_checkpoints.zip') as z:
 for n in z.namelist():assert not Path(n).is_absolute() and '..' not in Path(n).parts
 z.extractall(root)
print('Restored completed checkpoint pairs and cumulative recovery budget.')
""")
        elif args.action=='metadata':
            contents=ContentsClient(state.get_session('v10-free-t4'))
            for name in ['PLAN.md','README.md','IMPLEMENTATION_PROGRESS.md','build_delivery.py','adapter_verification.json',
                         'local_feature_verification.json','notebook_structure_verification.json','independent_review.json']:
                if (ROOT/name).exists():contents.upload(str(ROOT/name),REMOTE+'/'+name)
            print('Uploaded report/notebook packaging metadata; frozen model sources unchanged.')
        elif args.action=='city':
            contents=ContentsClient(state.get_session('v10-free-t4'))
            local=ROOT/'results';array=local/'reference_city_v7.npz';meta=local/'reference_city_v7.json'
            record=json.loads(meta.read_text())
            assert record['sha256']==hashlib.sha256(array.read_bytes()).hexdigest()
            assert record['lock_sha256']==hashlib.sha256((ROOT/'colab_results/results/candidate_lock.json').read_bytes()).hexdigest()
            for path in [array,meta]:contents.upload(str(path),REMOTE+'/results/'+path.name)
            receipt={'seconds':json.loads((local/'compute_budget.json').read_text())['seconds'],
                     'ledger_sha256':hashlib.sha256((local/'compute_budget.json').read_bytes()).hexdigest(),
                     'city_prediction_sha256':record['sha256'],'purpose':'Original-Mac city-baseline confirmation; charge once.'}
            path=local/'mac_compute_receipt.json';path.write_text(json.dumps(receipt,indent=2)+'\n')
            contents.upload(str(path),REMOTE+'/results/mac_compute_receipt.json')
            print('Verified original-Mac city baseline and compute receipt uploaded.')
        elif args.action=='merge_mac':
            execute("""import json
from pathlib import Path
root=Path('/content/mla_v10/results');receipt=json.loads((root/'mac_compute_receipt.json').read_text())
ledger=root/'compute_budget.json';state=json.loads(ledger.read_text());key=receipt['ledger_sha256']
if key not in state.get('merged_mac_receipts',[]):
 state['seconds']+=receipt['seconds'];state['by_stage']['confirm']=state['by_stage'].get('confirm',0)+receipt['seconds']
 state.setdefault('merged_mac_receipts',[]).append(key)
 tmp=ledger.with_suffix('.partial');tmp.write_text(json.dumps(state,indent=2)+'\\n');tmp.replace(ledger)
print('Original-Mac confirmation charge recorded once:',receipt['seconds'],'seconds')
""")
        elif args.action in ['stage','foreground']:
            assert args.value in ['audit','pilot','screen','confirm','export','all']
            execute("""import subprocess,sys,os,json
from pathlib import Path
root=Path('/content/mla_v10')
environment=os.environ.copy()
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LOKY_MAX_CPU_COUNT']:environment[key]='2'
environment['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
environment['PYTHONDONTWRITEBYTECODE']='1'
environment['HF_HUB_DISABLE_TELEMETRY']='1'
previous=root/'stage_pid.json'
if previous.exists():
 pid=json.loads(previous.read_text())['pid'];stat=Path('/proc')/str(pid)/'stat'
 assert not stat.exists() or stat.read_text().split()[2]=='Z','Previous V10 supervisor is still active.'
(root/'transport').mkdir(exist_ok=True)
code=SUPERVISOR
(root/'transport/stage_supervisor.py').write_text(code)
log=(root/'run.log').open('a')
p=subprocess.Popen([sys.executable,'-u',str(root/'transport/stage_supervisor.py'),STAGE],env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=root)
(root/'stage_pid.json').write_text(json.dumps({'stage':STAGE,'pid':p.pid,'supervised':True}))
print('Budget-supervised V10 stage started:',STAGE,'PID',p.pid)
""".replace('STAGE',repr(args.value)).replace('SUPERVISOR',repr((ROOT/'transport/stage_supervisor.py').read_text()))+
                    ("\nassert p.wait()==0,'V10 supervised process failed; check saved logs.'\n" if args.action=='foreground' else ''),
                    64800 if args.action=='foreground' else 180)
        elif args.action=='collect':
            execute("""import zipfile
from pathlib import Path
root=Path('/content/mla_v10')
with zipfile.ZipFile(root/'collected_results.zip','w',zipfile.ZIP_DEFLATED) as z:
 for folder in ['results','delivery','preflight_batch_failure']:
  for p in (root/folder).rglob('*'):
   if p.is_file() and '.partial' not in p.name:z.write(p,str(p.relative_to(root)))
 for name in ['run.log','setup.log','stage_pid.json','tabpfn_checkpoint.json']:
  p=root/name
  if p.exists():z.write(p,name)
print('Checkpoint backup prepared.')
""",600)
            local=ROOT/'colab_results';local.mkdir(exist_ok=True)
            archive=local/'collected_results.zip'
            ContentsClient(state.get_session('v10-free-t4')).download(REMOTE+'/collected_results.zip',str(archive))
            with zipfile.ZipFile(archive) as z:
                assert z.testzip() is None
                for n in z.namelist():assert not Path(n).is_absolute() and '..' not in Path(n).parts
                z.extractall(local)
            print('Saved local checkpoints:',local)
        elif args.action=='execute':
            assert args.value
            execute(Path(args.value).read_text(),64800)
        elif args.action=='release':
            import subprocess
            subprocess.run([str(ROOT.parent/'improvement_v9/colab_cli_env/bin/colab'),'--auth','oauth2','stop','-s','v10-free-t4'],check=True)
    finally:runtime.stop()


if __name__=='__main__':
    try:main()
    except Exception as error:
        # Network exception strings can contain authenticated proxy URLs.
        if type(error).__module__.startswith(('requests','urllib3')):
            print('Colab transport error:',type(error).__name__,flush=True)
        else:print('V10 transport error:',type(error).__name__,str(error),flush=True)
        sys.exit(1)
