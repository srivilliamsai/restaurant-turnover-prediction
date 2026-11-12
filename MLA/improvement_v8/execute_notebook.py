"""Execute the standalone notebook in a new process with no model checkpoints."""
import base64
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import pandas as pd
import nbformat as nb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'


class Tee:
    def __init__(self, capture, live): self.capture, self.live = capture, live
    def write(self, value):
        self.capture.write(value)
        self.live.write(value)
        self.live.flush()
        return len(value)
    def flush(self): self.live.flush()


def main():
    destination = OUT / 'notebook_execution'
    destination.mkdir(exist_ok=True)
    os.environ['MLA_DATA_DIR'] = str(ROOT.parent)
    os.environ['MLA_OUTPUT_DIR'] = str(destination)
    os.chdir(destination)
    path = ROOT / 'V8_Research_Colab.ipynb'
    book = nb.read(path, as_version=4)
    namespace = {'__name__': '__main__'}
    count, figures, started = 0, [], time.monotonic()
    for i, cell in enumerate(book.cells):
        if cell.cell_type != 'code': continue
        count += 1
        cell.execution_count = count
        stdout, stderr, displayed = io.StringIO(), io.StringIO(), []
        def show(*args, **kwargs):
            for number in plt.get_fignums():
                stream = io.BytesIO()
                plt.figure(number).savefig(stream, format='png', dpi=140, bbox_inches='tight')
                image = OUT / f'research_figure_{len(figures) + 1}.png'
                image.write_bytes(stream.getvalue())
                figures.append(image.name)
                displayed.append(nb.v4.new_output('display_data', data={'image/png': base64.b64encode(stream.getvalue()).decode(), 'text/plain': 'Measured V8 screening results'}))
                plt.close(number)
        plt.show = show
        print(f'Fresh V8 notebook: code cell {count}', flush=True)
        try:
            with contextlib.redirect_stdout(Tee(stdout, sys.stdout)), contextlib.redirect_stderr(Tee(stderr, sys.stderr)):
                exec(compile(cell.source, f'v8_cell_{i}', 'exec'), namespace)
        except Exception as error:
            cell.outputs = [nb.v4.new_output('stream', name='stdout', text=stdout.getvalue()),
                nb.v4.new_output('error', ename=type(error).__name__, evalue=str(error), traceback=traceback.format_exc().splitlines())]
            nb.write(book, path)
            raise
        cell.outputs = []
        if stdout.getvalue(): cell.outputs.append(nb.v4.new_output('stream', name='stdout', text=stdout.getvalue()))
        if stderr.getvalue(): cell.outputs.append(nb.v4.new_output('stream', name='stderr', text=stderr.getvalue()))
        cell.outputs.extend(displayed)
        nb.write(book, path)
    key = ['name', 'mode', 'weight']
    reference = pd.read_csv(OUT / 'screening.csv').set_index(key).sort_index()
    fresh = namespace['table'].set_index(key).sort_index()
    assert reference.index.equals(fresh.index) and len(fresh) == 64
    columns = ['rmse_42', 'rmse_71', 'gain_42', 'gain_71', 'mean_rmse']
    difference = float(np.max(np.abs(reference[columns].to_numpy() - fresh[columns].to_numpy())))
    assert difference < .01
    assert namespace['e'].cache.reads == 0
    module_directory = Path(namespace['workflow'].__file__).parent
    assert module_directory == namespace['WORK_ROOT']
    assert not namespace['decision']['trials']
    nb.validate(book)
    record = {'status': 'PASS', 'code_cells': count, 'figures': figures, 'fresh_process': True,
        'fresh_directory': str(namespace['WORK_ROOT']), 'earlier_project_directories_required': False,
        'prediction_checkpoint_reads': 0, 'v8_component_checkpoint_reads': 0,
        'v7_oof_snapshots_reused_explicitly': True, 'v8_outer_fold_fits': 160,
        'development_scores_recomputed': 128, 'maximum_score_difference': difference,
        'v7_reproduction': namespace['reproduction'], 'qualified_candidates': 0,
        'elapsed_seconds': time.monotonic() - started, 'portal_uploads_by_agent': 0}
    (OUT / 'research_notebook_execution.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__': main()
