"""Finish the active V9 job once, without another browser/manual upload step.

This is the delivery continuation of the running training job, not a scheduler.
It never starts extra experiments, uploads to Olympus, or buys compute.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from colab_cli.common import state
from colab_cli.contents import ContentsClient

os.umask(0o077)
root = Path(__file__).resolve().parent
run = root / 'colab_cli_runs' / '20261026T160148Z'
session_name = 'v9-cli-t4'
expected_endpoint = state.get_session(session_name).endpoint
colab = root / 'colab_cli_env/bin/colab'
python = Path('~/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3')
record = {'status': 'waiting_for_training', 'session': session_name,
          'new_experiments': 0, 'portal_uploads': 0, 'paid_resources_requested': False}

def save():
    (run / 'delivery_progress.json').write_text(json.dumps(record, indent=2) + '\n')

def command(argv, **kwargs):
    print('DELIVERY_STEP:', Path(argv[0]).name, Path(argv[-1]).name, flush=True)
    subprocess.run([str(a) for a in argv], check=True, **kwargs)

sleep_guard = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())])
save()
started = time.monotonic()
try:
    while True:
        driver = json.loads((run / 'execution.json').read_text())
        if driver['status'] == 'completed':
            break
        if driver['status'] == 'failed':
            raise RuntimeError('Training driver failed; preserve its evidence for review.')
        if time.monotonic()-started > 18*3600:
            raise RuntimeError('Delivery wait exceeded the 18-hour allowance; review required.')
        time.sleep(20)
    time.sleep(2)
    record['status'] = 'recording_environment'
    save()
    command([colab, '--auth', 'oauth2', 'exec', '-s', session_name,
             '--timeout', '180', '-f', root/'colab_freeze_environment.py'], timeout=240)
    current = state.get_session(session_name)
    assert current.endpoint == expected_endpoint, 'Runtime changed; do not operate on another session.'
    contents = ContentsClient(current)
    contents.download(driver['remote_output_dir'] + '/colab_environment_manifest.json',
                      str(run/'colab_environment_manifest.json'))
    record['status'] = 'independent_review'
    save()
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(root/'runtime_deps')+':'+str(root.parent/'.runtime_deps'),
        PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        MKL_NUM_THREADS='2', LOKY_MAX_CPU_COUNT='2')
    command([python, root/'review_colab_v9.py', run/'V9_research_results.zip',
             '--data-dir', root.parent], env=environment, timeout=1800)
    record['status'] = 'packaging'
    save()
    command([sys.executable, root/'package_cli_v9.py', run], timeout=600)
    record.update(status='completed',
        package=str(root/'V9_CLI_Delivery.zip'),
        package_sha256=hashlib.sha256((root/'V9_CLI_Delivery.zip').read_bytes()).hexdigest(),
        report=str(run/'V9_Colab_Report.md'))
    save()
    # This runtime was created for this run. Only release that exact assignment.
    current = state.get_session(session_name)
    assert current.endpoint == expected_endpoint
    command([colab, '--auth', 'oauth2', 'stop', '-s', session_name], timeout=180)
    record['owned_runtime_released'] = True
    save()
    print('V9_DELIVERY_COMPLETE:', record['report'], flush=True)
except Exception as error:
    record['status'] = 'needs_review'
    record['error_type'] = type(error).__name__
    record['error'] = str(error)
    save()
    (root/'CLI_PROGRESS.md').write_text('# V9 CLI run — review needed\n\n'
        'A training or delivery step needs review. No new portal upload was performed. '
        'Keep scored V7. Details are in `colab_cli_runs/20261026T160148Z/delivery_progress.json` '
        'and `colab_finish_v9.log`.\n')
    raise
finally:
    sleep_guard.terminate()
