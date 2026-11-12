"""Install isolated, pinned V19 candidate and frozen-control dependencies."""
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(os.environ.get('MLA_V19_SETUP_ROOT', '/content/mla_v19'))
ROOT.mkdir(exist_ok=True)
started = time.time()
records = {}


def save():
    (ROOT / 'setup_status.json').write_text(json.dumps(records, indent=2) + '\n')


def run(name, command, timeout=1200):
    begin = time.time()
    with (ROOT / 'setup.log').open('a') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    records[name] = {'returncode': result.returncode, 'seconds': time.time() - begin}
    save()
    print(json.dumps({'setup_step': name, **records[name]}), flush=True)
    return result.returncode == 0


baseline = ['numpy==2.5.4', 'pandas==3.0.6', 'scikit-learn==1.9.1',
            'scipy==1.18.1', 'catboost==1.2.10', 'interpret-core==0.7.8',
            'tabpfn==9.2.0', 'tabicl==2.2.0', 'torch==2.11.0',
            'skrub==0.10.1', 'pyarrow==23.0.1', 'nbformat', 'nbclient',
            'psutil', 'safetensors>=0.8', 'huggingface-hub']
lock = json.loads((ROOT / 'runtime_lock.json').read_text()) if (ROOT / 'runtime_lock.json').exists() else None
if lock:
    baseline = [name + '==' + version.split('+')[0] for name, version in lock['baseline'].items()]
assert run('frozen_control_dependencies', [sys.executable, '-m', 'pip', 'install', '-q', *baseline], 1500)
assert run('candidate_venv', [sys.executable, '-m', 'venv', '--without-pip', '--system-site-packages', str(ROOT / 'candidate_env')], 120)
candidate = str(ROOT / 'candidate_env/bin/python')
candidate_requirements = [name + '==' + version.split('+')[0] for name, version in lock['candidates'].items()] if lock else ['torch==2.13.0', 'causilo==1.0.3', 'numba==0.68.0']
run('causilo_torch213_environment', [candidate, '-m', 'pip', 'install', '-q', *candidate_requirements], 1500)
assert run('tabdpt_dependencies', [candidate, '-m', 'pip', 'install', '-q', 'tabdpt==1.3.1'], 600)
commit = '2b1696f4fdc22004cc757c44703a69cf326c0a11'
url = 'https://codeload.github.com/NVIDIA/structured-data-models/tar.gz/' + commit
archive = ROOT / 'kumo_source.tar.gz'
with urllib.request.urlopen(url, timeout=90) as response:
    archive.write_bytes(response.read())
records['kumo_source'] = {'commit': commit, 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
folder = ROOT / 'engineering_sources'
folder.mkdir(exist_ok=True)
with tarfile.open(archive) as bundle:
    bundle.extractall(folder, filter='data')
source = folder / ('structured-data-models-' + commit)
assert run('kumo_source_install', [candidate, '-m', 'pip', 'install', '-q', str(source)], 600)
for name, python in [('baseline', sys.executable), ('candidates', candidate)]:
    script = "import importlib.metadata,json,torch;print(json.dumps({'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'total_gpu_bytes':torch.cuda.get_device_properties(0).total_memory}))"
    raw = subprocess.check_output([python, '-c', script], timeout=90, text=True)
    records[name] = json.loads(raw)
assert records['baseline']['torch'] == '2.11.0+cu130'
assert records['baseline']['gpu'] == 'Tesla T4'
records['setup_complete'] = True
records['seconds'] = time.time() - started
save()
print(json.dumps({'setup_complete': True, 'baseline_torch': records['baseline']['torch'],
                  'candidate_torch': records['candidates']['torch'], 'gpu': records['baseline']['gpu']}), flush=True)
