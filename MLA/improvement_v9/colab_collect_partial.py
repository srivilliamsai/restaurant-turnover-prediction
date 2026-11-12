"""Back up completed, hash-verified V9 fold checkpoints without executing code."""
import hashlib
import json
import os
from pathlib import Path

from colab_cli.common import state
from colab_cli.contents import ContentsClient

os.umask(0o077)
root = Path(__file__).resolve().parent
run = root / 'colab_cli_runs' / '20261026T160148Z'
remote = 'content/v9_cli_20261026T160148Z/v9_fresh__h2_j7gd'
target = run / 'partial_backup'
target.mkdir(exist_ok=True)
contents = ContentsClient(state.get_session('v9-cli-t4'))
record = {}

def download(relative):
    local = target / relative
    local.parent.mkdir(parents=True, exist_ok=True)
    contents.download(remote + '/' + relative, str(local))
    record[relative] = hashlib.sha256(local.read_bytes()).hexdigest()
    return local

for name in ['audit.json', 'experiment_plan.json', 'pilot.json']:
    download('results/' + name)
download('tabpfn_checkpoint.json')
results_listing = contents.list_dir(remote + '/results')
result_names = {p.get('name') for p in results_listing.get('content', [])}
for optional in ['screening.csv', 'screening_decision.json', 'development.csv',
                 'candidate_lock.json', 'confirmation.csv', 'segments.csv',
                 'folds.csv', 'decision.json', 'compute_budget.json']:
    if optional in result_names:
        download('results/' + optional)
listing = contents.list_dir(remote + '/results/cache')
names = {item['name'] for item in listing.get('content', []) if item.get('type') == 'file'}
completed = 0
for name in sorted(names):
    if not name.startswith(('tabpfn_', 'v7_')) or not name.endswith('.json'):
        continue
    array = name[:-5] + '.npz'
    if array not in names:
        continue
    metadata = target / 'results/cache' / name
    prediction = metadata.with_suffix('.npz')
    if metadata.exists() and prediction.exists():
        expected = json.loads(metadata.read_text())['sha256']
        if hashlib.sha256(prediction.read_bytes()).hexdigest() == expected:
            completed += 1
            record['results/cache/' + name] = hashlib.sha256(metadata.read_bytes()).hexdigest()
            record['results/cache/' + array] = expected
            continue
    metadata = download('results/cache/' + name)
    prediction = download('results/cache/' + array)
    assert hashlib.sha256(prediction.read_bytes()).hexdigest() == json.loads(metadata.read_text())['sha256']
    completed += 1
(target / 'backup_manifest.json').write_text(json.dumps({'verified_completed_folds': completed,
    'files': record, 'remote_predictions_unmodified': True}, indent=2) + '\n')
print('Completed V9 fold checkpoints backed up:', completed)
