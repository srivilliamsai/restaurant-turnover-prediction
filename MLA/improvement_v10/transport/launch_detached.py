"""Start one finite V10 coordinator outside the chat tool's process group."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
root=Path(__file__).resolve().parents[1];run=root/'transport_run';run.mkdir(exist_ok=True)
os.umask(0o077)
if args.smoke:
    code="import os,time,json;from pathlib import Path;p=Path("+repr(str(run/'detached_smoke.json'))+");p.write_text(json.dumps({'pid':os.getpid(),'ppid':os.getppid(),'started':True}));time.sleep(60)"
    argv=[sys.executable,'-c',code];name='smoke'
else:
    argv=[sys.executable,str(root/'transport/continue_finite.py')];name='coordinator'
    path=run/'coordinator_pid.json'
    if path.exists():
        prior=json.loads(path.read_text())
        try:os.kill(prior['pid'],0)
        except ProcessLookupError:pass
        else:raise RuntimeError('Existing coordinator is present; inspect before launching another.')
with (run/(name+'.log')).open('a') as output:
    p=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
                       start_new_session=True,close_fds=True,cwd=root.parent)
(run/(name+'_pid.json')).write_text(json.dumps({'pid':p.pid,'finite':True,'paid_compute':False})+'\n')
print(json.dumps({'started':name,'pid':p.pid,'log':str(run/(name+'.log'))}))
