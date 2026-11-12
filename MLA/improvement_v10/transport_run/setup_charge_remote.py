import json
from pathlib import Path
p=Path('/content/mla_v10/results/compute_budget.json')
d=json.loads(p.read_text())
d['seconds']+=78.38645145800001
d['by_stage']['resumed_setup']=78.38645145800001
p.write_text(json.dumps(d,indent=2)+'\n')
print('Resumed setup charge recorded.')
