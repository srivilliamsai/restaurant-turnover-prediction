import os,signal,subprocess,json,time
from pathlib import Path
r=Path(__file__).resolve().parents[1];x=json.loads((r/'transport_run/status.json').read_text());pid=x['pid']
assert x['phase']=='installing_dependencies'
row=subprocess.run(['/bin/ps','-p',str(pid),'-o','command='],capture_output=True,text=True,check=True).stdout
assert 'improvement_v15_retry1/run_colab.py' in row
children=[]
for line in subprocess.run(['/bin/ps','-Ao','pid=,ppid='],capture_output=True,text=True,check=True).stdout.splitlines():
    parts=line.split()
    if len(parts)==2 and int(parts[1])==pid:children.append(int(parts[0]))
os.kill(pid,signal.SIGKILL)
for child in children:
    try:os.killpg(child,signal.SIGTERM)
    except ProcessLookupError:pass
record={'reason':'Server kernel idle after completed installation; local websocket client did not receive completion.','old_coordinator_stopped':True,'remote_runtime_released':False,'owned_session':'v15-retry1-free-t4','original_retry_budget_preserved':True,'numerical_sources_changed':False,'new_accuracy_results_seen':False,'epoch':time.time()}
(r/'transport_run/setup_recovery.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
