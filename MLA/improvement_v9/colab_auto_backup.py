"""Copy completed checkpoints for this one active V9 run until it finishes."""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.umask(0o077)
root = Path(__file__).resolve().parent
run = root / 'colab_cli_runs' / '20261026T160148Z'
started = time.monotonic()
while time.monotonic()-started < 18*3600:
    driver = json.loads((run/'execution.json').read_text())
    if driver['status'] in ['completed', 'failed']:
        break
    result = subprocess.run([sys.executable, str(root/'colab_collect_partial.py')], timeout=90)
    status = {'last_attempt_utc':datetime.now(timezone.utc).isoformat(),
              'last_exit_code':result.returncode, 'training_status':driver['status']}
    manifest = run/'partial_backup/backup_manifest.json'
    if manifest.exists():
        status['verified_completed_folds'] = json.loads(manifest.read_text())['verified_completed_folds']
    (run/'auto_backup.json').write_text(json.dumps(status,indent=2)+'\n')
    time.sleep(90)
print('Automatic checkpoint copy ended.',flush=True)
