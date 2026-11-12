"""Verify original V7 fitted-state inference in the existing Colab environment."""
import json
from pathlib import Path
from colab_cli.common import state
from colab_cli.contents import ContentsClient
from colab_cli.runtime import ColabRuntime
root=Path(__file__).resolve().parent
run=root/'colab_cli_runs/20261026T160148Z'
session=state.get_session('v9-cli-t4')
runtime=ColabRuntime(session.url,session.token,kernel_id=session.kernel_id,session_id=session.session_id)
try:
    outputs=runtime.execute_code('''
import pickle, hashlib, gc, baseline
from engine import check_behavior
checkpoint=WORK_ROOT/'export_diagnostics/local_v7_fit.pkl'
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()=='63ab212c7249ab857fc52004a97341662c5b110db7e21f19ae6b20dcdfc1090d'
with checkpoint.open('rb') as f: portable_v7=pickle.load(f)
portable_prediction=baseline.predict(portable_v7,e.test)
expected=pd.read_csv(WORK_ROOT/'scored_v7_backup/submission.csv')['Annual Turnover'].to_numpy(float)
difference=float(np.max(np.abs(portable_prediction-expected)))
checks=check_behavior(lambda raw:baseline.predict(portable_v7,raw),e.test,portable_prediction)
record={'maximum_difference_from_scored_V7_INR':difference,'checks':checks,
        'fitting_platform':'verified fresh Darwin arm64 fit',
        'inference_platform':'Linux x86_64 Colab',
        'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'fresh_Colab_V7_retraining_claimed':False}
(WORK_ROOT/'export_diagnostics/portable_inference.json').write_text(json.dumps(record,indent=2)+'\\n')
assert difference<.01
print('PORTABLE_V7_INFERENCE_PASS',difference)
del repeat_component
gc.collect()
import torch
torch.cuda.empty_cache()
''',timeout=180)
    for x in outputs:
        if x.get('output_type')=='stream':print(x.get('text',''),end='')
        if x.get('output_type')=='error':raise RuntimeError(x.get('ename','Error')+': '+x.get('evalue',''))
    ContentsClient(state.get_session('v9-cli-t4')).download('/content/v9_cli_20261026T160148Z/v9_fresh__h2_j7gd/export_diagnostics/portable_inference.json',str(run/'export_diagnostics/portable_inference.json'))
finally:runtime.stop()
