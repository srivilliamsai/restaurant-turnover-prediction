"""Remote process budget guard. Does not alter model settings or predictions."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

root=Path('/content/mla_v10');out=root/'results';stage=sys.argv[1]
ledger=out/'compute_budget.json'
def read():
    return json.loads(ledger.read_text()) if ledger.exists() else {'seconds':0.,'by_stage':{}}
def save(path,value):
    temp=path.with_name(path.name+'.partial')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
initial=read();maximum_hours=initial.get('maximum_hours',18)
caps={'audit':1,'pilot':1,'screen':6,'confirm':7,'export':3,'all':18}
total_limit=maximum_hours*3600-(0 if stage in ['export','all'] else min(3,maximum_hours/6)*3600)
maximum=min(total_limit-initial['seconds'],caps[stage]*3600-initial['by_stage'].get(stage,0))
if stage=='confirm' and not initial.get('merged_mac_receipts'):
    lock=out/'candidate_lock.json'
    if lock.exists() and json.loads(lock.read_text()).get('choice') is not None:
        # Bound the simultaneous original-Mac comparison to 90 minutes.
        # Reserve that charge until its actual receipt is merged at a safe boundary.
        maximum-=5400
if maximum<=0:
    save(out/'stage_status.json',{'stage':stage,'status':'failed','error':'Stage/total budget exhausted before launch; preserve V9.'})
    raise SystemExit(1)
started=time.monotonic()
p=subprocess.Popen([sys.executable,'-u',str(root/'workflow.py'),stage,'--data-dir',str(root/'data'),
                    '--output-dir',str(out),'--budget-hours',str(maximum_hours)],start_new_session=True,cwd=root)
expired=False
while p.poll() is None:
    elapsed=time.monotonic()-started
    save(out/'runtime_heartbeat.json',{'stage':stage,'elapsed_seconds':elapsed,'maximum_seconds':maximum,'pid':p.pid})
    if elapsed>=maximum:
        expired=True;os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        break
    time.sleep(5)
elapsed=time.monotonic()-started
latest=read();unrecorded=max(0,initial['seconds']+elapsed-latest['seconds'])
latest['seconds']+=unrecorded
latest['by_stage'][stage]=latest['by_stage'].get(stage,0)+unrecorded
latest['maximum_hours']=maximum_hours
latest['measurement']='Accumulated stage wall time plus conservatively charged unobserved interruption interval.'
save(ledger,latest)
if expired:
    save(out/'stage_status.json',{'stage':stage,'status':'failed','error':'Supervised stage/total time allowance exhausted; incomplete candidates cannot be exported.'})
print('STAGE_PROCESS_EXIT',stage,p.returncode,'elapsed_seconds',round(elapsed,3),'charged_hours',round(latest['seconds']/3600,4),flush=True)
raise SystemExit(1 if expired else p.returncode)
