"""Finite, detached V10 continuation. Exact experiments; free T4; global 24h cap."""
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'transport_run';RUN.mkdir(exist_ok=True)
OUT=ROOT/'colab_results/results'
CLI=ROOT.parent/'improvement_v9/colab_cli_env/bin/colab'
CONTROL=ROOT/'colab_control.py'
CPU='~/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'
amendment=json.loads((OUT/'budget_amendment.json').read_text())
assert amendment['authorization']=='Raise total cap to 24 hours and finish V10'
assert amendment['global_cap_hours']==24 and amendment['free_compute_only']
prior=float(amendment['prior_charged_seconds'])
epoch_at_start=float(json.loads((OUT/'compute_budget.json').read_text())['seconds'])
started=time.monotonic();deadline=started+24*3600-prior-epoch_at_start
mac=None;watch=None;guard=None;confirmation=None;allocated=False;city_uploaded=False;mac_seconds=0.
environment=os.environ.copy()
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LOKY_MAX_CPU_COUNT']:environment[key]='2'
environment['PYTHONDONTWRITEBYTECODE']='1'
cpu_environment=environment.copy()
cpu_environment['PYTHONPATH']=str(ROOT.parent/'improvement_v9/runtime_deps')+':'+str(ROOT.parent/'.runtime_deps')

def save(path,value):
    path=Path(path);temp=path.with_name(path.name+'.partial')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def status(phase,**extra):
    value={'phase':phase,'pid':os.getpid(),'elapsed_seconds':time.monotonic()-started,
           'global_cap_hours':24,'prior_charged_seconds':prior,'paid_resources':False,**extra}
    save(RUN/'status.json',value);print(json.dumps(value),flush=True)

def remaining():
    return max(0,deadline-time.monotonic()-mac_seconds)

def command(argv,timeout=None,env=environment,check=True):
    limit=min(remaining(),timeout or remaining())
    if limit<=0:raise TimeoutError('Global authorized time exhausted')
    p=subprocess.Popen([str(v) for v in argv],env=env,stdin=subprocess.DEVNULL,stdout=sys.stdout,
                       stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT.parent)
    try:code=p.wait(timeout=limit)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        raise TimeoutError('Finite V10 process allowance exhausted')
    if check and code:raise RuntimeError('V10 subprocess failed; inspect saved logs')
    return code

def control(action,value=None,timeout=None):
    return command([sys.executable,CONTROL,action,*([value] if value is not None else [])],timeout)

def stop_process(p):
    if p is None or p.poll() is not None:return
    p.terminate()
    try:p.wait(timeout=15)
    except subprocess.TimeoutExpired:p.kill();p.wait()

def verify_global_ledger():
    path=OUT/'compute_budget.json';ledger=json.loads(path.read_text())
    # Count transport/setup/local verification wall time as well as the GPU ledger.
    ledger['seconds']=max(float(ledger['seconds']),epoch_at_start+time.monotonic()-started+mac_seconds)
    ledger['global_charged_seconds']=prior+ledger['seconds']
    ledger['global_cap_hours']=24
    assert ledger['global_charged_seconds']<=24*3600,'Global 24-hour cap exhausted'
    save(path,ledger)
    return ledger

try:
    status('starting_free_t4')
    guard=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())],stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    command([CLI,'--auth','oauth2','usage'],120)
    for attempt in range(1,4):
        request=subprocess.run([str(CLI),'--auth','oauth2','new','--gpu','T4','-s','v10-free-t4'],
                               stdin=subprocess.DEVNULL,capture_output=True,text=True,env=environment,
                               timeout=min(180,remaining()))
        if request.returncode==0:
            allocated=True;print('Free T4 allocation READY.',flush=True);break
        response=request.stdout+request.stderr
        unavailable='Service Unavailable' in response or '503' in response
        quota=any(word in response.lower() for word in ['quota exceeded','usage limit','too many requests'])
        status('waiting_for_free_t4',attempt=attempt,service_unavailable=unavailable,explicit_limit=quota)
        if quota or not unavailable or attempt==3:raise RuntimeError('Free T4 allocation unavailable; no paid fallback')
        time.sleep(min(60*attempt,max(0,remaining())))
    control('setup',timeout=1800)
    # Colab's initial kernel can preload an older NumPy before pip replaces it.
    # Start a clean kernel before importing the frozen experiment dependencies.
    command([CLI,'--auth','oauth2','restart-kernel','-s','v10-free-t4'],120)
    control('upload',timeout=600);control('restore',timeout=600)
    control('execute',str(ROOT/'transport/resume_checks.py'),timeout=600)
    setup_seconds=time.monotonic()-started
    setup_script=RUN/'setup_charge_remote.py'
    setup_script.write_text("import json\nfrom pathlib import Path\np=Path('/content/mla_v10/results/compute_budget.json')\nd=json.loads(p.read_text())\nd['seconds']+="+repr(setup_seconds)+"\nd['by_stage']['resumed_setup']="+repr(setup_seconds)+"\np.write_text(json.dumps(d,indent=2)+'\\n')\nprint('Resumed setup charge recorded.')\n")
    control('execute',str(setup_script),timeout=120)
    log=(RUN/'watcher.log').open('a')
    watch=subprocess.Popen([sys.executable,str(ROOT/'transport/watch_backups.py')],env=environment,
                           stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT.parent)
    status('screening_from_verified_checkpoints',completed_folds=26)
    control('foreground','screen');control('collect',timeout=600)
    lock=json.loads((OUT/'candidate_lock.json').read_text())
    assert lock['complete'] and lock['settings']==6 and lock['configurations']==24
    status('finalist_frozen',choice=lock['choice'])
    if lock['choice'] is not None:
        mac_log=(RUN/'mac_city.log').open('a')
        mac_started=time.monotonic()
        mac=subprocess.Popen([CPU,str(ROOT/'refit_city.py')],env=cpu_environment,stdin=subprocess.DEVNULL,
                             stdout=mac_log,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT.parent)
    confirm_log=(RUN/'confirmation_transport.log').open('a')
    confirmation=subprocess.Popen([sys.executable,str(CONTROL),'foreground','confirm'],env=environment,
                                  stdin=subprocess.DEVNULL,stdout=confirm_log,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT.parent)
    status('confirming_frozen_finalist')
    while confirmation.poll() is None:
        if remaining()<=0:stop_process(confirmation);raise TimeoutError('Global budget exhausted')
        if mac is not None:
            if time.monotonic()-mac_started>=5400 and mac.poll() is None:
                os.killpg(mac.pid,signal.SIGTERM);raise TimeoutError('Original-Mac city baseline exceeded its allowance')
            if mac.poll() is not None and not city_uploaded:
                if mac.returncode:raise RuntimeError('Original-Mac city baseline failed')
                mac_seconds=float(json.loads((ROOT/'results/compute_budget.json').read_text())['seconds'])
                control('city',timeout=120);city_uploaded=True
                status('confirming_with_verified_mac_city_baseline',mac_seconds=mac_seconds)
        time.sleep(5)
    if confirmation.returncode:raise RuntimeError('Frozen confirmation failed; inspect transport and model logs')
    control('collect',timeout=600)
    if mac is not None:
        while mac.poll() is None:
            if time.monotonic()-mac_started>=5400 or remaining()<=0:
                os.killpg(mac.pid,signal.SIGTERM);raise TimeoutError('Original-Mac city baseline allowance exhausted')
            time.sleep(5)
        if mac.returncode:raise RuntimeError('Original-Mac city baseline failed')
        mac_seconds=float(json.loads((ROOT/'results/compute_budget.json').read_text())['seconds'])
        if not city_uploaded:control('city',timeout=120);city_uploaded=True
        control('merge_mac',timeout=120)
        if not (OUT/'decision.json').exists():control('foreground','confirm')
        control('collect',timeout=600)
    decision=json.loads((OUT/'decision.json').read_text());assert decision['complete']
    status('independent_verification',qualified=decision['qualified'])
    command([CPU,str(ROOT/'transport/verify_results.py')],600,env=cpu_environment)
    control('execute',str(ROOT/'transport/check_pruned_models.py'),timeout=3600)
    control('metadata',timeout=300)
    status('fresh_notebook_reproduction_and_delivery',qualified=decision['qualified'])
    control('foreground','export')
    # Stop mirror writes before final local accounting so an older remote ledger
    # cannot replace the final transport-inclusive receipt.
    stop_process(watch);watch=None
    control('collect',timeout=1200)
    ledger=verify_global_ledger()
    finalize=RUN/'finalize_local.py'
    finalize.write_text('import sys\nfrom pathlib import Path\nroot=Path('+repr(str(ROOT))+')\nsys.path.insert(0,str(root))\nfrom core import Engine\nimport build_delivery\ne=Engine(root.parent,root/"colab_results/results")\nbuild_delivery.finalize_receipt(e,root/"colab_results/delivery")\n')
    command([CPU,finalize],600,env=cpu_environment)
    status('complete',qualified=decision['qualified'],decision=decision['decision'],
           global_charged_hours=ledger['global_charged_seconds']/3600,portal_uploads=0,
           delivery=str(ROOT/'V10_Delivery.zip'))
except Exception as error:
    status('stopped_with_preserved_v9',error_type=type(error).__name__,qualified_submission_recommended=False)
    if allocated:
        try:control('collect',timeout=300)
        except Exception:pass
    try:verify_global_ledger()
    except Exception:pass
    report=RUN/'NO_NEW_UPLOAD.md'
    report.write_text('V10 continuation stopped before complete delivery verification. Preserve scored V9. Inspect status.json and saved diagnostics. No new portal upload is recommended.\n')
finally:
    stop_process(confirmation)
    if mac is not None and mac.poll() is None:
        try:os.killpg(mac.pid,signal.SIGTERM)
        except ProcessLookupError:pass
    stop_process(watch)
    if allocated:
        try:subprocess.run([str(CLI),'--auth','oauth2','stop','-s','v10-free-t4'],stdin=subprocess.DEVNULL,
                           stdout=sys.stdout,stderr=subprocess.STDOUT,timeout=120,check=False)
        except Exception:pass
    stop_process(guard)
