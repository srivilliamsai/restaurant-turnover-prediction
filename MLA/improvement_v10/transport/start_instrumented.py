"""Remote transport wrapper: timed prediction logs and a bounded pilot process."""
import json
import os
import subprocess
import sys
from pathlib import Path

root=Path('/content/mla_v10')
(root/'transport').mkdir(exist_ok=True)
worker=r'''import faulthandler,json,runpy,sys,time
from pathlib import Path
faulthandler.enable()
faulthandler.dump_traceback_later(600,repeat=True)
root=Path('/content/mla_v10');sys.path.insert(0,str(root))
import modeling
original=modeling.predict
calls=0
def timed_predict(fitted,raw):
 global calls
 calls+=1;started=time.monotonic()
 faulthandler.cancel_dump_traceback_later()
 faulthandler.dump_traceback_later(600,repeat=True)
 print('PREDICT_START',calls,fitted['name'],len(raw),time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),flush=True)
 result=original(fitted,raw)
 faulthandler.cancel_dump_traceback_later()
 print('PREDICT_END',calls,'seconds',round(time.monotonic()-started,3),flush=True)
 return result
modeling.predict=timed_predict
sys.argv=[str(root/'workflow.py'),'pilot','--data-dir',str(root/'data'),'--output-dir',str(root/'results')]
runpy.run_path(str(root/'workflow.py'),run_name='__main__')
'''
(root/'transport/timed_worker.py').write_text(worker)
supervisor=r'''import json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path('/content/mla_v10');started=time.monotonic()
ledger=json.loads((root/'results/compute_budget.json').read_text())
maximum=min(3100,3600-ledger['by_stage'].get('pilot',0))
assert maximum>0,'Pilot allowance exhausted.'
log=(root/'run.log').open('a')
p=subprocess.Popen([sys.executable,'-u',str(root/'transport/timed_worker.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=root)
while p.poll() is None:
 elapsed=time.monotonic()-started
 path=root/'results/runtime_heartbeat.json';tmp=path.with_suffix('.partial')
 tmp.write_text(json.dumps({'stage':'pilot','elapsed_seconds':elapsed,'maximum_seconds':maximum,'pid':p.pid}));tmp.replace(path)
 if elapsed>=maximum:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=30)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  (root/'results/stage_status.json').write_text(json.dumps({'stage':'pilot','status':'failed','error':'Monitored pilot time allowance exhausted; preserve V9.'}))
  break
 time.sleep(10)
print('PILOT_PROCESS_EXIT',p.returncode,'elapsed',time.monotonic()-started,flush=True)
raise SystemExit(p.returncode)
'''
(root/'transport/pilot_supervisor.py').write_text(supervisor)
environment=os.environ.copy()
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LOKY_MAX_CPU_COUNT']:environment[key]='2'
environment.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',HF_HUB_DISABLE_TELEMETRY='1')
log=(root/'run.log').open('a')
p=subprocess.Popen([sys.executable,'-u',str(root/'transport/pilot_supervisor.py')],env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=root)
(root/'stage_pid.json').write_text(json.dumps({'stage':'pilot','pid':p.pid,'supervised':True}))
print('Bounded, instrumented pilot started:',p.pid)
assert p.wait()==0,'Pilot supervisor failed; inspect saved logs.'
