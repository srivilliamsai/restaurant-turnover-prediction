"""Read-only, credential-free summary of saved V15 progress."""
import csv
import json
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
results = root / 'colab_results' / 'results'

def read(path):
    return json.loads(path.read_text()) if path.exists() else {}

auth = read(root / 'authorization.json')
elapsed = time.time() - auth['started_epoch']
status = read(root / 'transport_run' / 'status.json')
heartbeat = read(results / 'runtime_heartbeat.json')
pilots = read(results / 'pilot_progress.json')
summary = {
    'phase': status.get('phase'),
    'elapsed_minutes': round(elapsed / 60, 2),
    'remaining_minutes': round(max(0, 4 * 3600 - elapsed) / 60, 2),
    'stage': heartbeat.get('stage'),
    'heartbeat_age_seconds': round(time.time() - heartbeat['epoch']) if heartbeat else None,
    'pilots': {name: result['status'] for name, result in pilots.items()},
    'complete_new_fold_backups': len(list((results / 'cache').glob('*.json'))),
}
if len(pilots) == 5:
    screening = results / 'screening.csv'
    if screening.exists():
        with screening.open() as handle:
            rows = list(csv.DictReader(handle))
        summary['development_comparisons'] = len(rows)
        if rows:
            best = max(rows, key=lambda row: float(row['mean_relative_gain']))
            summary['best_development'] = {k: best[k] for k in ['name', 'weight', 'rmse_42', 'rmse_71', 'mean_relative_gain']}
    lock = read(results / 'candidate_lock.json')
    if lock:
        summary['frozen_candidates'] = lock.get('choices')
    decision = read(results / 'decision.json')
    if decision:
        summary['decision'] = decision.get('decision')
        summary['qualified'] = decision.get('qualified')
print(json.dumps(summary, indent=2))
