"""Launch with the compatible desktop Python and existing workspace dependencies."""
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
    dependency_root = ROOT.parent / '.runtime_deps'
    if dependency_root.exists():
        environment['PYTHONPATH'] = str(dependency_root) + os.pathsep + environment.get('PYTHONPATH', '')
        omp = dependency_root / 'sklearn/.dylibs'
        if omp.exists():
            environment['DYLD_LIBRARY_PATH'] = str(omp) + os.pathsep + environment.get('DYLD_LIBRARY_PATH', '')
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    steps = [
        ['prepare_reference.py'],
        ['workflow.py', 'all', '--data-dir', str(args.data_dir.resolve())],
        ['deliver.py', 'build'],
        ['execute_notebook.py', '--data-dir', str(args.data_dir.resolve())],
        ['deliver.py', 'finalize'],
    ]
    for step in steps:
        print('Running:', ' '.join(step), flush=True)
        subprocess.run([executable, str(ROOT / step[0]), *step[1:]], env=environment, cwd=ROOT, check=True)
