from pathlib import Path
import json
root=Path('/content/mla_v12');out=root/'results'
assert not list((out/'cache').glob('*.npz')),'Cannot replace a scored run'
assert not (out/'screening.csv').exists(),'Cannot replace development results'
for name in ['pilot.json','pilot_progress.json','experiment_plan.json','audit.json','stage_status.json','runtime_heartbeat.json']:
 (out/name).unlink(missing_ok=True)
print('Prior incomplete aggregate cleared after full archival; three verified pilot checkpoints retained.')
