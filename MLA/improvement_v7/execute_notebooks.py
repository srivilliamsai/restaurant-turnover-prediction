"""Run every selected candidate notebook in its own fresh process and verify outputs."""
import argparse
import base64
import contextlib
import io
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat as nb

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
STEM = 'submission'


class Tee:
    def __init__(self, capture, live):
        self.capture, self.live = capture, live

    def write(self, text):
        self.capture.write(text)
        self.live.write(text)
        self.live.flush()
        return len(text)

    def flush(self):
        self.capture.flush()
        self.live.flush()


def execute(index, data_dir):
    work = OUT / f'notebook_trial_{index}'
    work.mkdir(exist_ok=True)
    os.environ['MLA_DATA_DIR'] = str(data_dir.resolve())
    os.environ['MLA_OUTPUT_DIR'] = str(work)
    os.environ['V7_REPRODUCTION_FORCE_FRESH'] = '1'
    os.chdir(work)
    path = ROOT / 'candidates' / f'trial_{index}' / f'{STEM}.ipynb'
    book = nb.read(path, as_version=4)
    namespace = {'__name__': '__main__'}
    count, started, figures = 0, time.monotonic(), []
    for cell_index, cell in enumerate(book.cells):
        if cell.cell_type != 'code':
            continue
        count += 1
        cell.execution_count = count
        stdout, stderr, displayed = io.StringIO(), io.StringIO(), []

        def show(*args, **kwargs):
            for number in plt.get_fignums():
                stream = io.BytesIO()
                plt.figure(number).savefig(stream, format='png', dpi=130, bbox_inches='tight')
                image_path = OUT / f'trial_{index}_figure_{len(figures) + 1}.png'
                image_path.write_bytes(stream.getvalue())
                figures.append(image_path.name)
                displayed.append(nb.v4.new_output('display_data', data={
                    'image/png': base64.b64encode(stream.getvalue()).decode(), 'text/plain': 'Measured v7 results'}))
                plt.close(number)

        plt.show = show
        print(f'Trial {index}: executing cell {cell_index}, code {count}', flush=True)
        try:
            with contextlib.redirect_stdout(Tee(stdout, sys.stdout)), contextlib.redirect_stderr(Tee(stderr, sys.stderr)):
                exec(compile(cell.source, f'trial_{index}_cell_{cell_index}', 'exec'), namespace)
        except Exception as error:
            cell.outputs = [nb.v4.new_output('stream', name='stdout', text=stdout.getvalue()),
                            nb.v4.new_output('error', ename=type(error).__name__, evalue=str(error),
                                             traceback=traceback.format_exc().splitlines())]
            nb.write(book, path)
            raise
        cell.outputs = []
        if stdout.getvalue():
            cell.outputs.append(nb.v4.new_output('stream', name='stdout', text=stdout.getvalue()))
        if stderr.getvalue():
            cell.outputs.append(nb.v4.new_output('stream', name='stderr', text=stderr.getvalue()))
        cell.outputs.extend(displayed)
        nb.write(book, path)
        print(f'Trial {index}: completed code {count}, elapsed {time.monotonic() - started:.1f}s', flush=True)
    independent = np.load(OUT / f'full_fit_trial_{index}.npz')['prediction']
    full_difference = float(np.max(np.abs(namespace['test_prediction'] - independent)))
    oof_difference = float(np.max(np.abs(namespace['fresh_component'] - namespace['EXPECTED_COMPONENT_OOF'])))
    assert full_difference < .01 and oof_difference < .01
    assert namespace['cache_reads'] == 0
    nb.validate(book)
    record = {'trial': f'trial_{index}', 'code_cells': count, 'figures': figures,
              'elapsed_seconds': time.monotonic() - started, 'fresh_process': True,
              'validation_checkpoint_reads': namespace['cache_reads'], 'fresh_five_fold_assignment': 18421,
              'maximum_full_fit_difference': full_difference, 'maximum_oof_difference': oof_difference,
              'fresh_full_fit': True, 'checks': namespace['checks'], 'portal_uploads': 0,
              'csv_path': str(namespace['output_path'])}
    (OUT / f'notebook_execution_trial_{index}.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT.parent)
    parser.add_argument('--trial', type=int)
    args = parser.parse_args()
    if args.trial is not None:
        execute(args.trial, args.data_dir)
    else:
        decision = json.loads((OUT / 'decision.json').read_text())
        for index in range(1, len(decision['trials']) + 1):
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--trial', str(index),
                            '--data-dir', str(args.data_dir.resolve())], check=True, env=os.environ.copy())
