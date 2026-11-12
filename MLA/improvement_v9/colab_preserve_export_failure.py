"""Preserve the incomplete run; never mark it as an accepted delivery."""
import hashlib
import json
from pathlib import Path
from colab_cli.common import state
from colab_cli.contents import ContentsClient

root = Path(__file__).resolve().parent
run = root / 'colab_cli_runs' / '20261026T160148Z'
contents = ContentsClient(state.get_session('v9-cli-t4'))
remote = '/content/v9_cli_20261026T160148Z/v9_fresh__h2_j7gd/export_diagnostics/'
target = run / 'export_diagnostics'
target.mkdir(exist_ok=True)
# Only these two files, produced by our diagnostic, are in scope. No directory
# traversal, folder archive, model pickle, environment or authentication files.
allowed = ('failure.json', 'failed_v7_reproduction.npz')
files = {}
for name in allowed:
    path = target / name
    contents.download(remote + name, str(path))
    files[name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                   'bytes': path.stat().st_size}
record = {'status': 'preserved_incomplete', 'release_ready': False,
          'failed_gate': 'scored V7 reproduction within INR 0.01',
          'files': files, 'new_model_fits': 0, 'portal_uploads': 0}
(run / 'incomplete_preservation.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2))
