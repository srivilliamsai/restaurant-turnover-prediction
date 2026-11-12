"""Finite free-T4 V11 runner with an eight-hour separate task ledger."""
import json,os,signal,subprocess,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'transport_run';RUN.mkdir(exist_ok=True)
CLI=ROOT.parent/'improvement_v9/colab_cli_env/bin/colab';CONTROL=ROOT/'colab_control.py'
CPU='~/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'
started=time.monotonic();mac_seconds=0.;allocated=False;guard=None;watch=None;mac=None;confirmation=None
ledger=ROOT/'colab_results/results/compute_budget.json'
previous=json.loads(ledger.read_text())['seconds'] if ledger.exists() else 0.
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1')
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LOKY_MAX_CPU_COUNT']:env[key]='2'
cpuenv=env.copy();cpuenv['PYTHONPATH']=str(ROOT.parent/'improvement_v9/runtime_deps')+':'+str(ROOT.parent/'.runtime_deps')
def save(p,v):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);t=p.with_name(p.name+'.partial');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def status(phase,**extra):
 d={'phase':phase,'pid':os.getpid(),'elapsed_seconds':time.monotonic()-started,'maximum_hours':8,'paid_resources':False,**extra};save(RUN/'status.json',d);print(json.dumps(d),flush=True)
def remaining():return max(0,28800-previous-(time.monotonic()-started)-mac_seconds)
def command(args,timeout=None,environment=env,check=True):
 limit=min(remaining(),timeout or remaining())
 if limit<=0:raise TimeoutError('V11 eight-hour allowance exhausted')
 p=subprocess.Popen([str(x) for x in args],env=environment,stdin=subprocess.DEVNULL,stdout=sys.stdout,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT.parent)
 try:code=p.wait(timeout=limit)
 except subprocess.TimeoutExpired:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  raise TimeoutError('V11 subprocess time allowance exhausted')
 if check and code:raise RuntimeError('V11 stage failed; inspect preserved logs')
 return code
def control(action,value=None,timeout=None):return command([sys.executable,CONTROL,action,*([value] if value is not None else [])],timeout)
def stop(p):
 if p is None or p.poll() is not None:return
 try:os.killpg(p.pid,signal.SIGTERM)
 except ProcessLookupError:return
 try:p.wait(timeout=15)
 except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
def final_account():
 d=json.loads(ledger.read_text()) if ledger.exists() else {'seconds':0.,'by_stage':{}}
 d['seconds']=max(d['seconds'],previous+time.monotonic()-started+mac_seconds);d['maximum_hours']=8
 assert d['seconds']<=28800,'V11 budget exceeded';save(ledger,d);return d
try:
 status('requesting_free_t4');guard=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 command([CLI,'--auth','oauth2','usage'],120)
 for attempt in range(1,4):
  try:p=subprocess.run([str(CLI),'--auth','oauth2','new','--gpu','T4','-s','v11-free-t4'],stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=min(180,remaining()),env=env)
  except subprocess.TimeoutExpired:
   status('allocation_retry',attempt=attempt,request_timeout=True)
   if attempt==3:raise
   time.sleep(60*attempt);continue
  if p.returncode==0:allocated=True;break
  error=p.stdout+p.stderr;unavailable='503' in error or 'Service Unavailable' in error
  status('allocation_retry',attempt=attempt,service_unavailable=unavailable)
  if not unavailable or attempt==3:raise RuntimeError('Free T4 unavailable; no paid fallback')
  time.sleep(60*attempt)
 status('installing_pinned_dependencies');control('setup',timeout=1800)
 command([CLI,'--auth','oauth2','restart-kernel','-s','v11-free-t4'],120)
 status('preparing_verified_experiment_bundle')
 ready_deadline=time.monotonic()+600
 while not (ROOT/'RUN_READY.json').exists():
  if time.monotonic()>ready_deadline:raise TimeoutError('Implementation preparation allowance exhausted')
  time.sleep(3)
 backup=ROOT/'colab_results/results'
 if backup.exists():
  # Restore only complete, hash-verified pairs; interrupted writes stay local.
  for p in backup.rglob('*'):
   if not p.is_file() or '.partial' in p.name:continue
   if p.suffix=='.npz' and p.parent.name=='cache':
    import hashlib
    meta=p.with_suffix('.json');assert meta.exists(),'Incomplete cached fold'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==json.loads(meta.read_text())['sha256'],'Corrupt cached fold'
   destination=ROOT/'results'/p.relative_to(backup);destination.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(p,destination)
 if (ROOT/'colab_results/resolved_models.json').exists():shutil.copy2(ROOT/'colab_results/resolved_models.json',ROOT/'resolved_models.json')
 control('upload',timeout=600)
 charge=RUN/'charge_setup.py';charge.write_text("import json\nfrom pathlib import Path\np=Path('/content/mla_v11/results/compute_budget.json')\nd=json.loads(p.read_text()) if p.exists() else {'seconds':0.,'by_stage':{}}\nd['seconds']+="+repr(time.monotonic()-started)+"\nd['by_stage']['setup']=d['by_stage'].get('setup',0)+"+repr(time.monotonic()-started)+"\nd['maximum_hours']=8\np.write_text(json.dumps(d,indent=2)+'\\n')\n")
 control('execute',str(charge),120)
 log=(RUN/'watcher.log').open('a');watch=subprocess.Popen([sys.executable,str(ROOT/'transport/watch_backups.py')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT.parent)
 status('auditing');control('foreground','audit')
 pilot=ROOT/'results/pilot.json'
 if pilot.exists():
  # Keep the engineering-only decision frozen; do not resolve new upstream weights
  # or repeat already passing model fits after an unrelated reporting interruption.
  resume=RUN/'verify_resume.py';resume.write_text("""import sys,json
from pathlib import Path
root=Path('/content/mla_v11');sys.path.insert(0,str(root))
from core import Engine,sha,save
import modeling,workflow
e=Engine(root/'data',root/'results');p=json.loads((e.out/'pilot.json').read_text())
assert p['status'] in ['PASS','NO_FEASIBLE_FAMILY']
assert p['sources']==e.sources and p['inputs']==e.inputs
assert p['model_contract_sha256']==sha(root/'resolved_models.json')
assert p['model_contract']['runtime']==modeling.runtime()
assert callable(workflow.diagnostics) and callable(workflow.uncertainty)
for family in {name[0] for name in p['screen_settings']}:
 assert p['families'][family]['status']=='PASS'
 modeling.local_checkpoint(p['model_contract']['models'][family]['checkpoint'])
save(e.out/'resume_verification.json',{'status':'PASS','pilot_reused_without_retuning':True,'sources':e.sources,'inputs':e.inputs,'model_contract_sha256':p['model_contract_sha256'],'entrypoint':'import workflow; workflow.main()'})
print('Frozen pilot, runtime, checkpoint and fully loaded reporting functions verified; no model refit for resume.')
""");status('verifying_frozen_pilot_for_resume');control('execute',str(resume),600)
 else:
  status('piloting_new_backbones');control('foreground','pilot')
 control('collect',timeout=600)
 status('screening_feasible_fixed_settings');control('foreground','screen');control('collect',timeout=600)
 out=ROOT/'colab_results/results';lock=json.loads((out/'candidate_lock.json').read_text());status('finalist_frozen',choice=lock['choice'])
 if lock['choice'] is not None:
  maclog=(RUN/'mac_controls.log').open('a');mac_started=time.monotonic()
  mac=subprocess.Popen([CPU,str(ROOT/'refit_controls.py')],env=cpuenv,stdin=subprocess.DEVNULL,stdout=maclog,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT.parent)
  clog=(RUN/'confirmation_transport.log').open('a')
  confirmation=subprocess.Popen([sys.executable,str(CONTROL),'foreground','confirm'],env=env,stdin=subprocess.DEVNULL,stdout=clog,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT.parent)
  status('confirming_frozen_finalist');uploaded=set()
  while confirmation.poll() is None:
   if remaining()<=7200:raise TimeoutError('Preserving final delivery reserve')
   if mac.poll() is not None and mac.returncode:raise RuntimeError('Original-Mac controls failed')
   for seed in [61301,61302,61303]:
    path=ROOT/f'mac_results/reference_v7_{seed}.json'
    if path.exists() and seed not in uploaded:control('mac',timeout=120);uploaded.add(seed)
   if time.monotonic()-mac_started>=3600 and mac.poll() is None:raise TimeoutError('Mac comparison allowance reached')
   time.sleep(5)
  if confirmation.returncode:raise RuntimeError('Confirmation transport failed')
  if mac is not None:
   while mac.poll() is None:
    if time.monotonic()-mac_started>=3600 or remaining()<=7200:raise TimeoutError('Mac comparison allowance reached')
    time.sleep(5)
   if mac.returncode:raise RuntimeError('Mac comparison failed')
   mac_seconds=float(json.loads((ROOT/'mac_results/mac_compute_receipt.json').read_text())['seconds']);control('mac',timeout=120)
 control('collect',timeout=600)
 if not (out/'decision.json').exists():control('foreground','confirm');control('collect',timeout=600)
 decision=json.loads((out/'decision.json').read_text());assert decision['complete']
 status('independent_verification',qualified=decision['qualified'])
 command([CPU,str(ROOT/'transport/verify_results.py')],600,environment=cpuenv)
 control('metadata',timeout=120)
 status('reproduction_and_delivery',qualified=decision['qualified']);control('foreground','export')
 stop(watch);watch=None;control('collect',timeout=600)
 d=final_account()
 command([CPU,str(ROOT/'transport/finalize.py')],300,environment=cpuenv)
 status('complete',qualified=decision['qualified'],decision=decision['decision'],charged_hours=d['seconds']/3600,portal_uploads=0)
except Exception as error:
 status('stopped_with_preserved_v9',error_type=type(error).__name__,message=str(error)[:200],qualified_submission_recommended=False)
 if allocated:
  try:control('collect',timeout=180)
  except Exception:pass
 stop(watch);watch=None
 try:final_account()
 except Exception:pass
finally:
 stop(confirmation);stop(mac);stop(watch)
 if allocated:
  try:subprocess.run([str(CLI),'--auth','oauth2','stop','-s','v11-free-t4'],stdin=subprocess.DEVNULL,stdout=sys.stdout,stderr=subprocess.STDOUT,timeout=120)
  except Exception:pass
 if guard is not None:guard.terminate()
