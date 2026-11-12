import json
from pathlib import Path
p=Path('/content/mla_v12/results/compute_budget.json')
d=json.loads(p.read_text()) if p.exists() else {'seconds':0.,'by_stage':{}}
d['seconds']+=55.92097579199617
d['by_stage']['setup']=d['by_stage'].get('setup',0)+55.92098658399482
d['maximum_hours']=18
p.write_text(json.dumps(d,indent=2)+'\n')
