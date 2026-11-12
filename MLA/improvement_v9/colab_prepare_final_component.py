"""Prepare only the frozen selected TabPFN full fit and verify a repeated fit."""
import json
from pathlib import Path
from colab_cli.common import state
from colab_cli.contents import ContentsClient
from colab_cli.runtime import ColabRuntime

root=Path(__file__).resolve().parent
run=root/'colab_cli_runs/20261026T160148Z'
session=state.get_session('v9-cli-t4')
runtime=ColabRuntime(session.url,session.token,kernel_id=session.kernel_id,session_id=session.session_id)
source='''
import json, time, hashlib
import modeling
from engine import check_behavior
decision=json.loads((e.out/'decision.json').read_text())
assert len(decision['trials'])==1
selected=decision['trials'][0]
assert selected['name']=='tabpfn_raw_v7' and selected['mode']=='ensemble16' and selected['weight']==.5
assert hashlib.sha256((e.out/'candidate_lock.json').read_bytes()).hexdigest()==decision['lock_sha256']
final_component_dir=WORK_ROOT/'final_component'
final_component_dir.mkdir(exist_ok=True)
started=time.monotonic()
final_component=modeling.fit_model(e.train,e.y,modeling.SPECS[selected['name']],selected['mode'],print)
final_component_prediction=modeling.predict_model(final_component,e.test)
component_checks=check_behavior(lambda raw:modeling.predict_model(final_component,raw),e.test,final_component_prediction)
del final_component
repeat_component=modeling.fit_model(e.train,e.y,modeling.SPECS[selected['name']],selected['mode'],print)
repeated_prediction=modeling.predict_model(repeat_component,e.test)
maximum_repeat_difference=float(np.max(np.abs(final_component_prediction-repeated_prediction)))
np.savez_compressed(final_component_dir/'tabpfn_full_fit.npz',prediction=final_component_prediction,repeated=repeated_prediction,ids=e.test['Registration Number'].to_numpy())
record={'selected':{k:selected[k] for k in ['name','mode','weight']},
        'inputs':e.inputs,'sources':e.sources,'execution_contract':modeling.execution_contract(),
        'maximum_repeat_difference_INR':maximum_repeat_difference,
        'checks':component_checks,'seconds':time.monotonic()-started,
        'prediction_sha256':hashlib.sha256((final_component_dir/'tabpfn_full_fit.npz').read_bytes()).hexdigest(),
        'baseline_reproduction_still_required':True}
(final_component_dir/'checks.json').write_text(json.dumps(record,indent=2)+'\\n')
print('FULL_TABPFN_REPEAT_DIFFERENCE_INR',maximum_repeat_difference)
assert maximum_repeat_difference<.01
print('FULL_TABPFN_COMPONENT_CHECKS_PASS')
'''
try:
    outputs=runtime.execute_code(source,timeout=1800,output_hook=lambda x:print(x.get('text',''),end='',flush=True) if x.get('output_type')=='stream' else None)
    for x in outputs:
        if x.get('output_type')=='error':raise RuntimeError(x.get('ename','Error')+': '+x.get('evalue',''))
    contents=ContentsClient(state.get_session('v9-cli-t4'))
    target=run/'final_component';target.mkdir(exist_ok=True)
    remote='/content/v9_cli_20261026T160148Z/v9_fresh__h2_j7gd/final_component/'
    for name in ['tabpfn_full_fit.npz','checks.json']:contents.download(remote+name,str(target/name))
    print('Frozen full component backed up locally.',flush=True)
finally:runtime.stop()
