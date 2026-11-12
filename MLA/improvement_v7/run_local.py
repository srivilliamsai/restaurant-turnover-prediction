"""Use the compatible desktop Python and workspace packages for all v7 stages."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT.parent)
    parser.add_argument('--python', default=os.environ.get('MLA_PYTHON'))
    args = parser.parse_args()
    bundled = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'
    executable = args.python or (str(bundled) if bundled.exists() else sys.executable)
    environment = os.environ.copy()
    dependencies = ROOT.parent / '.runtime_deps'
    if dependencies.exists():
        environment['PYTHONPATH'] = str(dependencies) + os.pathsep + environment.get('PYTHONPATH', '')
        omp = dependencies / 'sklearn/.dylibs'
        if omp.exists():
            environment['DYLD_LIBRARY_PATH'] = str(omp) + os.pathsep + environment.get('DYLD_LIBRARY_PATH', '')
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    environment['LOKY_MAX_CPU_COUNT'] = '2'
    stages = [['prepare_resume.py'], ['workflow.py', 'all', '--data-dir', str(args.data_dir.resolve())],
              ['deliver.py', 'build'],
              ['execute_notebooks.py', '--data-dir', str(args.data_dir.resolve())],
              ['deliver.py', 'finalize']]
    for stage in stages:
        print('Running:', ' '.join(stage), flush=True)
        subprocess.run([executable, str(ROOT / stage[0]), *stage[1:]], cwd=ROOT, env=environment, check=True)
