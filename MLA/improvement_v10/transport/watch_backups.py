"""Back up every completed remote fold locally; no credentials in logs/artifacts."""
import base64
import hashlib
import json
import os
import time
import requests
from pathlib import Path
from colab_cli.common import state
from colab_cli.contents import ContentsClient

os.umask(0o077)
# The upstream contents client has no request timeout. Bound transport waits
# without editing its package or printing authenticated URLs on failure.
_request=requests.request
def bounded_request(*args,**kwargs):
    kwargs.setdefault('timeout',(10,45))
    return _request(*args,**kwargs)
requests.request=bounded_request
ROOT=Path(__file__).resolve().parents[1]
REMOTE='/content/mla_v10'
LOCAL=ROOT/'colab_results'
LOCAL.mkdir(exist_ok=True)


def read(contents,path):
    result=contents._request('GET',path,params={'content':'1'})
    content=result['content']
    return base64.b64decode(content) if result.get('format')=='base64' else str(content).encode()


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.backup_partial');temp.write_bytes(value);temp.replace(path)


deadline=time.monotonic()+18*3600
failures=0
while time.monotonic()<deadline:
    try:
        contents=ContentsClient(state.get_session('v10-free-t4'))
        for name in ['run.log','results/stage_status.json','results/compute_budget.json','results/pilot.json',
                     'results/runtime_heartbeat.json','results/experiment_plan.json','results/audit.json',
                     'results/feature_verification.json','results/baseline_inference_differences.json',
                     'results/screening_progress.json','results/screening.csv','results/component_diagnostics.csv',
                     'results/development_segments.csv','results/development_folds.csv','results/development_concentration.json',
                     'results/confirmation.csv','results/confirmation_progress.json','results/confirmation_segments.csv',
                     'results/confirmation_folds.csv','results/confirmation_concentration.json',
                     'results/candidate_lock.json','results/decision.json']:
            try:put(LOCAL/name,read(contents,REMOTE+'/'+name))
            except FileNotFoundError:pass
        try:listing=contents.list_dir(REMOTE+'/results/cache')['content']
        except FileNotFoundError:listing=[]
        names={entry['name'] for entry in listing}
        for name in sorted(names):
            if not name.endswith('.json') or name[:-5]+'.npz' not in names:continue
            npz=name[:-5]+'.npz'
            meta_path=LOCAL/'results/cache'/name;data_path=meta_path.with_suffix('.npz')
            if meta_path.exists() and data_path.exists():continue
            metadata_bytes=read(contents,REMOTE+'/results/cache/'+name)
            metadata=json.loads(metadata_bytes)
            data=read(contents,REMOTE+'/results/cache/'+npz)
            assert hashlib.sha256(data).hexdigest()==metadata['sha256']
            put(data_path,data);put(meta_path,metadata_bytes)
            print('BACKED_UP',npz,flush=True)
        status_path=LOCAL/'results/stage_status.json'
        status=json.loads(status_path.read_text()) if status_path.exists() else {}
        if status.get('status') in ['failed','waiting_city_baseline'] or (status.get('stage')=='export' and status.get('status')=='complete'):
            print('WATCH_END',json.dumps(status),flush=True);break
        failures=0
    except BaseException as error:
        if isinstance(error,KeyboardInterrupt):raise
        failures+=1
        print('Backup transport retry:',type(error).__name__,flush=True)
        if failures>=3:
            print('WATCH_END: runtime unavailable after three attempts; preserved local backups.',flush=True)
            break
    time.sleep(min(60,15*max(1,failures)))
