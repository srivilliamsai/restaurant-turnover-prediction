"""Execute the standalone notebook in a fresh Python process, without OOF caches."""
import argparse
import base64
import contextlib
import io
import json
import os
import time
import traceback
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat as nb

ROOT = Path(__file__).resolve().parent
STEM = 'submission'


def main(data_dir):
    work = ROOT / 'results/notebook_run'
    work.mkdir(exist_ok=True)
    os.environ['MLA_DATA_DIR'] = str(Path(data_dir).resolve())
    os.environ['MLA_OUTPUT_DIR'] = str(work)
    os.chdir(work)
    book = nb.read(ROOT / f'{STEM}.ipynb', as_version=4)
    namespace = {'__name__': '__main__'}
    started, count, figures = time.monotonic(), 0, []
    for index, cell in enumerate(book.cells):
        if cell.cell_type != 'code':
            continue
        count += 1
        cell.execution_count = count
        stdout, stderr, displayed = io.StringIO(), io.StringIO(), []

        def show(*args, **kwargs):
            for number in plt.get_fignums():
                stream = io.BytesIO()
                plt.figure(number).savefig(stream, format='png', dpi=130, bbox_inches='tight')
                path = ROOT / 'results' / f'notebook_figure_{len(figures) + 1}.png'
                path.write_bytes(stream.getvalue())
                figures.append(path.name)
                displayed.append(nb.v4.new_output('display_data', data={
                    'image/png': base64.b64encode(stream.getvalue()).decode(), 'text/plain': 'Measured experiment figure'}))
                plt.close(number)

        plt.show = show
        print(f'Executing notebook cell {index} ({count})', flush=True)
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(compile(cell.source, f'notebook_cell_{index}', 'exec'), namespace)
        except Exception as error:
            cell.outputs = [nb.v4.new_output('stream', name='stdout', text=stdout.getvalue()),
                            nb.v4.new_output('error', ename=type(error).__name__, evalue=str(error),
                                             traceback=traceback.format_exc().splitlines())]
            nb.write(book, ROOT / f'{STEM}.ipynb')
            raise
        cell.outputs = []
        if stdout.getvalue():
            cell.outputs.append(nb.v4.new_output('stream', name='stdout', text=stdout.getvalue()))
        if stderr.getvalue():
            cell.outputs.append(nb.v4.new_output('stream', name='stderr', text=stderr.getvalue()))
        cell.outputs.extend(displayed)
        nb.write(book, ROOT / f'{STEM}.ipynb')
        print(f'Completed {count}; elapsed {time.monotonic() - started:.1f}s', flush=True)
    expected = np.load(ROOT / 'results/full_fit.npz')['prediction']
    difference = float(np.max(np.abs(namespace['test_prediction'] - expected)))
    assert difference < .01
    assert np.max(np.abs(namespace['fresh_selected'] - namespace['EXPECTED_SELECTED_OOF'])) < .01
    nb.validate(book)
    summary = {'code_cells': count, 'figures': figures, 'elapsed_seconds': time.monotonic() - started,
               'fresh_process': True, 'validation_cache_used': False, 'fresh_five_fold_assignment': 18421,
               'maximum_full_fit_difference': difference,
               'maximum_oof_difference': float(np.max(np.abs(namespace['fresh_selected'] - namespace['EXPECTED_SELECTED_OOF']))),
               'full_fit_reproduced': True, 'checks': namespace['checks'],
               'export_path': str(namespace['output_path'])}
    (ROOT / 'results/notebook_execution.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT.parent)
    main(parser.parse_args().data_dir)
