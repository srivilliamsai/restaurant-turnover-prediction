import json
from pathlib import Path
p=Path('/content/mla_v11/results/compute_budget.json')
d=json.loads(p.read_text()) if p.exists() else {'seconds':0.,'by_stage':{}}
d['seconds']+=54.449220624999725
d['by_stage']['setup']=d['by_stage'].get('setup',0)+54.44922854199831
d['maximum_hours']=8
p.write_text(json.dumps(d,indent=2)+'\n')
