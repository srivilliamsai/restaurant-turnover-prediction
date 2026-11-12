"""Launch the isolated workflow with the compatible local CPU environment."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', nargs='?', default='all', choices=['audit', 'pilot', 'screen', 'refine', 'confirm', 'export', 'all'])
    p.add_argument('--data-dir', type=Path, default=ROOT.parent)
    p.add_argument('--out-dir', type=Path, default=ROOT / 'results')
    p.add_argument('--budget-hours', type=float, default=18)
    args = p.parse_args()
    bundled = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'
    python = str(bundled) if bundled.exists() else sys.executable
    environment = os.environ.copy()
    deps = ROOT.parent / '.runtime_deps'
    if deps.exists():
        environment['PYTHONPATH'] = str(ROOT / 'runtime_deps') + os.pathsep + str(deps) + os.pathsep + environment.get('PYTHONPATH', '')
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'LOKY_MAX_CPU_COUNT']:
        environment[key] = '2'
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    subprocess.run([python, str(ROOT / 'workflow.py'), args.stage, '--data-dir', str(args.data_dir.resolve()),
                    '--out-dir', str(args.out_dir.resolve()), '--budget-hours', str(args.budget_hours)],
                   env=environment, check=True)
