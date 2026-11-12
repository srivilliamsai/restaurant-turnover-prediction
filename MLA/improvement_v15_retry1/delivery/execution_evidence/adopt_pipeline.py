"""Maintain the existing foreground pipeline while renewing expiring proxy access."""
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

logging.disable(logging.CRITICAL)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import colab_control as control

RUN = ROOT / 'transport_run'
AUTH = json.loads((ROOT / 'authorization.json').read_text())
END = min(AUTH['started_epoch'] + 4 * 3600, AUTH['hard_stop_epoch'])

def save(name, value):
    path = RUN / name
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)

def status(phase, **details):
    record = dict(phase=phase, pid=os.getpid(), maximum_hours=4,
                  task_elapsed_seconds=time.time()-AUTH['started_epoch'], **details)
    save('status.json', record)
    print(json.dumps(record), flush=True)

def connect(session):
    runtime = control.ColabRuntime(session.url, session.token, kernel_id=session.kernel_id)
    client = runtime.kernel_client._manager.client
    assert client.connection_ready.is_set(), 'Replacement foreground connection not ready'
    assert client.kernel_socket.sock.connected, 'Replacement foreground socket disconnected'
    return runtime

def invoke(action, timeout=300):
    result = subprocess.run([sys.executable, str(ROOT/'colab_control.py'), action],
                            stdin=subprocess.DEVNULL, capture_output=True, text=True,
                            timeout=timeout)
    print(result.stdout, end='', flush=True)
    assert result.returncode == 0, 'Owned V15 transport operation failed: '+action

def main():
    old = json.loads((RUN/'status.json').read_text())
    assert old['phase'] == 'running_fixed_pipeline'
    old_pid = old['pid']
    command = subprocess.run(['/bin/ps', '-p', str(old_pid), '-o', 'command='],
                             capture_output=True, text=True, check=True).stdout
    assert 'improvement_v15_retry1/run_colab.py' in command
    session = control.state.get_session(control.SESSION)
    contents = control.ContentsClient(session)
    markers = []
    for entry in contents.list_dir(control.REMOTE+'/transport_markers')['content']:
        if entry['name'].endswith('.json'):
            path = control.REMOTE+'/transport_markers/'+entry['name']
            record = json.loads(control.read(contents, path))
            if record.get('status') == 'running':
                assert entry['name'] == record['id']+'.json'
                markers.append((path, record['id']))
    assert len(markers) == 1, 'Expected exactly one existing foreground pipeline'
    marker, identity = markers[0]
    runtime = connect(session)
    token = session.token
    children = []
    for line in subprocess.run(['/bin/ps', '-Ao', 'pid=,ppid='], capture_output=True,
                               text=True, check=True).stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and int(parts[1]) == old_pid:
            children.append(int(parts[0]))
    # A replacement socket is connected before the old local clients close.
    os.kill(old_pid, signal.SIGKILL)
    for child in children:
        try:
            os.killpg(child, signal.SIGTERM)
        except ProcessLookupError:
            try:
                os.kill(child, signal.SIGTERM)
            except ProcessLookupError:
                pass
    save('connection_adoption.json', dict(epoch=time.time(),
         existing_remote_pipeline_preserved=True, numerical_sources_changed=False,
         budget_reset=False, replacement_socket_verified_before_old_client_closed=True))
    status('monitoring_existing_pipeline')
    complete = False
    released = False
    last_backup = 0.
    failures = 0
    try:
        while time.time() < END:
            session = control.state.get_session(control.SESSION)
            if session.token != token:
                replacement = connect(session)
                runtime.stop()
                runtime = replacement
                token = session.token
                print('Foreground proxy access renewed; existing computation preserved.', flush=True)
            contents = control.ContentsClient(session)
            try:
                record = json.loads(control.read(contents, marker))
            except json.JSONDecodeError:
                time.sleep(1)
                continue
            assert record['id'] == identity
            assert record['status'] != 'error', 'Remote pipeline failed: '+record.get('error_type','unknown')
            if record['status'] == 'ok':
                invoke('snapshot', 180)
                invoke('collect', 600)
                complete = True
                break
            if time.monotonic()-last_backup >= 60:
                try:
                    invoke('snapshot', 90)
                    failures = 0
                except Exception:
                    failures += 1
                last_backup = time.monotonic()
                heartbeat = ROOT/'colab_results/results/runtime_heartbeat.json'
                if heartbeat.exists():
                    value = json.loads(heartbeat.read_text())
                    assert value['status'] != 'running' or time.time()-value['epoch'] <= 300, 'Remote heartbeat stale'
                assert failures < 3, 'Three consecutive backup failures'
            time.sleep(5)
        assert complete, 'Original retry allowance expired; no further compute authorized'
    except Exception as error:
        status('stopped', error_type=type(error).__name__)
        try:
            invoke('snapshot', 180)
            invoke('collect', 600)
        except Exception:
            pass
        raise
    finally:
        try:
            invoke('release', 150)
            released = True
        finally:
            runtime.stop()
            save('runtime_release.json', dict(owned_session=control.SESSION,
                 release_command_succeeded=released, epoch=time.time()))
            status('complete' if complete else 'stopped', artifacts_collected=complete,
                   owned_runtime_released=released)

if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('V15 foreground monitor stopped:', type(error).__name__, flush=True)
        raise SystemExit(1)
