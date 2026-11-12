import json
import shutil
from pathlib import Path
root=Path('/content/mla_v10')
status=json.loads((root/'results/stage_status.json').read_text())
assert status['stage']=='pilot' and status['status']=='failed'
archive=root/'preflight_batch_failure'
assert not archive.exists()
(root/'results').rename(archive)
for name in ['modeling.py','core.py','workflow.py','verify.py']:
    shutil.copy2(root/name,archive/('original_'+name))
(root/'results').mkdir()
shutil.copy2(archive/'compute_budget.json',root/'results/compute_budget.json')
shutil.copy2(root/'run.log',archive/'run.log')
(root/'run.log').write_text('Restarting numerical feasibility check with fixed internal prediction batches.\n')
print('Preserved failed pilot and sources; carried its compute time into the new ledger.')
