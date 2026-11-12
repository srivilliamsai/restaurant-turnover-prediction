"""Run the frozen V9 notebook through Google's CLI runtime transport.

Only transport, storage paths and frontend download handling differ. Model code,
features, settings, validation assignments and acceptance gates stay unchanged.
Stops at the first notebook error and saves each completed cell locally.
"""
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import nbformat
import requests
from colab_cli.common import state
from colab_cli.contents import ContentsClient
from colab_cli.runtime import ColabRuntime

os.umask(0o077)
ROOT = Path(__file__).resolve().parent
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
local = ROOT / 'colab_cli_runs' / stamp
local.mkdir(parents=True, exist_ok=False)
remote = '/content/v9_cli_' + stamp
original = ROOT / 'V9_Research_Colab_Lite.ipynb'
book = nbformat.read(original, as_version=4)
executed = local / 'V9_Research_Colab_CLI_executed.ipynb'
record = {'started_utc': stamp, 'source_notebook_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
          'session': os.environ.get('V9_COLAB_SESSION', 'v9-existing-t4'), 'remote_output_dir': remote,
          'status': 'starting', 'completed_cells': 0,
          'transport_changes_only': ['CLI execution', 'explicit input/output paths',
                                     'download artifacts via Contents API'],
          'paid_resources_requested': False}

def persist():
    (local / 'execution.json').write_text(json.dumps(record, indent=2) + '\n')
    nbformat.write(book, executed)

def output(item):
    if item.get('output_type') == 'stream':
        print(item.get('text', ''), end='', flush=True)
    elif item.get('output_type') == 'error':
        print(item.get('ename', 'Error') + ': ' + item.get('evalue', ''), flush=True)

def execute(code, timeout=64800):
    result = runtime.execute_code(code, output_hook=output, timeout=timeout)
    errors = [o for o in result if o.get('output_type') == 'error']
    if errors:
        raise RuntimeError(errors[0].get('ename', 'Error') + ': ' + errors[0].get('evalue', ''))
    return result

def normalize(outputs):
    normalized = []
    for item in outputs:
        allowed = {k: v for k, v in item.items()
                   if k in ['output_type', 'name', 'text', 'data', 'metadata',
                            'execution_count', 'ename', 'evalue', 'traceback']}
        normalized.append(nbformat.from_dict(allowed))
    return normalized

session = state.get_session(record['session'])
response = requests.get(session.url.rstrip('/') + '/api/kernels',
    params={'authuser': '0', 'colab-runtime-proxy-token': session.token}, timeout=30)
if not response.ok:
    raise RuntimeError(f'Kernel inspection HTTP {response.status_code}')
kernels = response.json()
assert len(kernels) <= 1, 'Several kernels exist; inspect before selecting.'
if kernels:
    assert kernels[0]['execution_state'] == 'idle', 'Existing notebook is busy; do not duplicate its work.'
    if session.kernel_id:
        assert kernels[0]['id'] == session.kernel_id
    session.kernel_id = kernels[0]['id']

def kernel_started(identifier):
    session.kernel_id = identifier
    state.store.add(session)

def session_started(identifier):
    session.session_id = identifier
    state.store.add(session)

def connect_runtime():
    return ColabRuntime(session.url, session.token, kernel_id=session.kernel_id,
        session_id=session.session_id, on_kernel_started=kernel_started,
        on_session_started=session_started)

runtime = connect_runtime()
prelude = (
    "import os\n"
    "os.chdir('/content')\n"
    "os.environ['MLA_DATA_DIR'] = '/content'\n"
    f"os.environ['MLA_OUTPUT_DIR'] = {remote!r}\n"
    "os.environ.pop('V9_CPU_ONLY', None)\n"
    "from google.colab import files as _v9_files\n"
    "_v9_files.download = lambda path: print('CLI_ARTIFACT_READY:', path)\n"
)
persist()
started = time.monotonic()
try:
    execute(prelude, 90)
    codes = [c for c in book.cells if c.cell_type == 'code']
    print('V9_CLI_RUN:', str(local), flush=True)
    print('Installing pinned packages before a clean kernel restart.', flush=True)
    execute(codes[0].source, 1800)
    runtime.restart(timeout=120)
    runtime.stop()
    runtime = connect_runtime()
    execute(prelude, 90)
    record['package_setup_kernel_restart'] = True
    for number, cell in enumerate(codes, 1):
        record['status'] = 'running'
        record['current_cell'] = number
        persist()
        print(f'V9_CELL {number}/{len(codes)} START', flush=True)
        cell.outputs = normalize(execute(cell.source))
        cell.execution_count = number
        record['completed_cells'] = number
        persist()
        print(f'V9_CELL {number}/{len(codes)} COMPLETE', flush=True)
    archive = local / 'V9_research_results.zip'
    ContentsClient(state.get_session(record['session'])).download(remote + '/V9_research_results.zip', str(archive))
    record.update(status='completed', archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                  executed_notebook_sha256=hashlib.sha256(executed.read_bytes()).hexdigest())
    print('V9_RESULTS_DOWNLOADED:', str(archive), flush=True)
except Exception as error:
    record['status'] = 'failed'
    record['error'] = f'{type(error).__name__}: {error}'
    print('V9_CLI_ERROR:', record['error'], flush=True)
    raise
finally:
    record['elapsed_seconds'] = time.monotonic() - started
    persist()
    runtime.stop()
