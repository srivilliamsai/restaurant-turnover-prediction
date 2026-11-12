"""Move disposable download caches after verified inference, before packaging."""
import json,time,os
from pathlib import Path
root=Path('/content/mla_v14')
checks=root/'delivery/fresh_execution/fresh_execution_checks.json'
auth=json.loads((root/'authorization.json').read_text())
limit=min(time.time()+3600,auth['started_epoch']+8*3600)
while not checks.exists():
    if time.time()>limit:raise SystemExit('No verified inference before cleanup deadline; no files moved.')
    status=root/'results/stage_status.json'
    if status.exists() and json.loads(status.read_text()).get('status')=='failed':raise SystemExit('Execution failed; no files moved.')
    time.sleep(.1)
record=json.loads(checks.read_text())
assert record['status']=='PASS' and record['maximum_final_difference_INR']<.01
moves=[]
for source in (root/'delivery/fresh_execution').glob('v14_fresh_*/model_weights'):
    destination=root/'model_weights/verified_notebook_cache'/source.parent.name
    destination.parent.mkdir(parents=True,exist_ok=True)
    assert not destination.exists()
    sizes={p.name:p.stat().st_size for p in source.iterdir() if p.is_file()}
    source.rename(destination)
    moves.append({'from':str(source),'to':str(destination),'files_bytes':sizes})
result={'status':'PASS','method':'Reversible cache relocation after fresh inference passed; numerical sources unchanged.','epoch':time.time(),'moves':moves}
(root/'results/packaging_cache_relocation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
