"""Inspect the failed export without starting a model fit or changing a gate."""
import json
from pathlib import Path
from colab_cli.common import state
from colab_cli.runtime import ColabRuntime

root = Path(__file__).resolve().parent
run = root / 'colab_cli_runs' / '20261026T160148Z'
session = state.get_session('v9-cli-t4')
runtime = ColabRuntime(session.url, session.token, kernel_id=session.kernel_id,
                       session_id=session.session_id)
source = '''
import sys, json, traceback, pickle, hashlib
from pathlib import Path
ip = get_ipython()
error = getattr(getattr(ip, 'last_execution_result', None), 'error_in_exec', None)
tb = getattr(error, '__traceback__', None) or getattr(sys, 'last_traceback', None)
frames = []
while tb is not None:
    frame = tb.tb_frame
    frames.append({'file': Path(frame.f_code.co_filename).name,
                   'function': frame.f_code.co_name, 'line': tb.tb_lineno})
    if frame.f_code.co_name == '_export':
        values = frame.f_locals
        report = {'maximum_difference_INR': float(values['difference']),
                  'rms_difference_INR': float(np.sqrt(np.mean((values['b']-values['expected'])**2))),
                  'first_fresh': values['b'][:5].tolist(),
                  'first_scored': values['expected'][:5].tolist()}
        diagnostic_dir = WORK_ROOT / 'export_diagnostics'
        diagnostic_dir.mkdir(exist_ok=True)
        np.savez_compressed(diagnostic_dir / 'failed_v7_reproduction.npz',
                            fresh=values['b'], expected=values['expected'])
        with (diagnostic_dir / 'fresh_v7_fit.pkl').open('wb') as f:
            pickle.dump(values['fitted_base'], f)
        report['saved_fit_sha256'] = hashlib.sha256((diagnostic_dir / 'fresh_v7_fit.pkl').read_bytes()).hexdigest()
        (diagnostic_dir / 'failure.json').write_text(json.dumps(report, indent=2)+'\\n')
        print('EXPORT_DIAGNOSTIC', json.dumps(report))
    tb = tb.tb_next
print('ERROR_FRAMES', json.dumps(frames))
print('KERNEL_AVAILABLE', str(WORK_ROOT))
'''
try:
    outputs = runtime.execute_code(source, timeout=120)
    for item in outputs:
        if item.get('output_type') == 'stream':
            print(item.get('text', ''), end='')
        elif item.get('output_type') == 'error':
            print(item.get('ename'), item.get('evalue'))
    (run / 'export_diagnostic_outputs.json').write_text(json.dumps(outputs, indent=2)+'\n')
finally:
    runtime.stop()
